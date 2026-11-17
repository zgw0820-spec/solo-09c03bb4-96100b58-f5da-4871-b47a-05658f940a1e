"""Power-cut drill scenarios: one node down at a time, ready-count impact,
reservation release, first-commit-wins interleavings and restart durability."""

from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402


def make_client(db: Path) -> TestClient:
    return TestClient(create_app(db))


def topology(nodes: list[dict], containers: list[dict]) -> dict:
    return {"nodes": nodes, "containers": containers}


def node(state: dict, node_id: str) -> dict:
    return next(n for n in state["nodes"] if n["id"] == node_id)


def service(state: dict, name: str) -> dict:
    return next(s for s in state["services"] if s["name"] == name)


def test_power_off_drops_ready_counts_and_flags_services(tmp_path):
    client = make_client(tmp_path / "t.db")
    payload = topology(
        nodes=[
            {"id": "n1", "cpu_capacity": 8},
            {"id": "n2", "cpu_capacity": 8},
        ],
        containers=[
            {"id": "c1", "service": "svc", "min_ready": 2, "node_id": "n1", "cpu_request": 2},
            {"id": "c2", "service": "svc", "min_ready": 2, "node_id": "n2", "cpu_request": 2},
            {"id": "c3", "service": "other", "min_ready": 1, "node_id": "n1", "cpu_request": 1},
        ],
    )
    with client:
        client.post("/api/topology", json=payload)
        state = client.post("/api/nodes/n2/power-off").json()

        assert state["powered_off_node_id"] == "n2"
        assert node(state, "n2")["powered_on"] is False
        # c2 no longer counts: svc drops to 1/2 and is flagged unhealthy.
        svc = service(state, "svc")
        assert svc["ready"] == 1
        assert svc["min_ready"] == 2
        assert svc["healthy"] is False
        assert svc["offline_replicas"] == 1
        other = service(state, "other")
        assert other["ready"] == 1 and other["healthy"] is True
        # The replica is still placed on n2; capacity accounting is unchanged.
        assert node(state, "n2")["serving_containers"] == 1
        assert node(state, "n2")["used_cpu"] == 2

        state = client.post("/api/nodes/n2/power-on").json()
        assert state["powered_off_node_id"] is None
        svc = service(state, "svc")
        assert svc["ready"] == 2 and svc["healthy"] is True
        assert svc["offline_replicas"] == 0


def test_only_one_node_powered_off_at_a_time(tmp_path):
    client = make_client(tmp_path / "t.db")
    payload = topology(
        nodes=[{"id": "n1", "cpu_capacity": 4}, {"id": "n2", "cpu_capacity": 4}],
        containers=[],
    )
    with client:
        client.post("/api/topology", json=payload)
        assert client.post("/api/nodes/n1/power-off").status_code == 200
        second = client.post("/api/nodes/n2/power-off")
        assert second.status_code == 409
        assert second.json()["error"]["code"] == "ANOTHER_NODE_POWERED_OFF"
        # Repeating the same cut is also rejected.
        again = client.post("/api/nodes/n1/power-off")
        assert again.status_code == 409
        assert again.json()["error"]["code"] == "NODE_ALREADY_POWERED_OFF"
        # After restoring, the other node may be cut instead.
        assert client.post("/api/nodes/n1/power-on").status_code == 200
        assert client.post("/api/nodes/n2/power-off").status_code == 200
        # Power-on on a live node is a no-op error.
        assert client.post("/api/nodes/n1/power-on").status_code == 409
        assert client.post("/api/nodes/ghost/power-off").status_code == 404
        assert client.post("/api/nodes/ghost/power-on").status_code == 404


def test_power_off_fails_inbound_pending_and_releases_reservation(tmp_path):
    client = make_client(tmp_path / "t.db")
    payload = topology(
        nodes=[
            {"id": "n1", "cpu_capacity": 8},
            {"id": "n2", "cpu_capacity": 8},
        ],
        containers=[
            {"id": "c1", "service": "svc", "min_ready": 1, "node_id": "n1", "cpu_request": 3},
        ],
    )
    with client:
        client.post("/api/topology", json=payload)
        state = client.post("/api/nodes/n1/drain").json()
        migration = state["open_drains"][0]["pending_decisions"][0]
        assert migration["target_node_id"] == "n2"
        assert node(state, "n2")["reserved_cpu"] == 3

        state = client.post("/api/nodes/n2/power-off").json()
        # Reservation released, source replica untouched and still serving.
        assert node(state, "n2")["reserved_cpu"] == 0
        assert node(state, "n2")["remaining_cpu"] == 8
        assert node(state, "n1")["serving_containers"] == 1
        c1 = next(c for c in state["containers"] if c["id"] == "c1")
        assert c1["node_id"] == "n1"
        # Drain turned incomplete with the power-cut reason surfaced.
        assert state["open_drains"][0]["status"] == "incomplete"
        blocked = state["blocked_containers"]
        assert len(blocked) == 1
        assert blocked[0]["container_id"] == "c1"
        assert blocked[0]["reason_code"] == "TARGET_POWERED_OFF"
        # The failed migration is no longer pending; resolving it is stale.
        resp = client.post(
            f"/api/migrations/{migration['id']}/resolve", json={"outcome": "ready"}
        )
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "MIGRATION_NOT_PENDING"


def test_pending_starts_on_other_targets_stay_decidable(tmp_path):
    client = make_client(tmp_path / "t.db")
    payload = topology(
        nodes=[
            {"id": "n1", "cpu_capacity": 8},
            {"id": "n2", "cpu_capacity": 8},
            {"id": "n3", "cpu_capacity": 8},
        ],
        containers=[
            {"id": "c1", "service": "svc", "min_ready": 1, "node_id": "n1", "cpu_request": 2},
            {"id": "c2", "service": "svc", "min_ready": 1, "node_id": "n1", "cpu_request": 2},
        ],
    )
    with client:
        client.post("/api/topology", json=payload)
        state = client.post("/api/nodes/n1/drain").json()
        pending = state["open_drains"][0]["pending_decisions"]
        assert len(pending) == 2
        by_target = {m["target_node_id"]: m for m in pending}
        doomed = by_target["n2"]
        survivor = by_target["n3"]

        state = client.post("/api/nodes/n2/power-off").json()
        # Only the n2-bound start failed; the n3-bound one is still pending.
        remaining = state["open_drains"][0]["pending_decisions"]
        assert [m["id"] for m in remaining] == [survivor["id"]]
        state = client.post(
            f"/api/migrations/{survivor['id']}/resolve", json={"outcome": "ready"}
        ).json()
        moved = next(c for c in state["containers"] if c["id"] == survivor["container_id"])
        assert moved["node_id"] == "n3"
        # The doomed migration's source replica never left n1.
        stuck = next(c for c in state["containers"] if c["id"] == doomed["container_id"])
        assert stuck["node_id"] == "n1"
        assert state["open_drains"][0]["status"] == "incomplete"


def test_powered_off_node_is_not_a_migration_target(tmp_path):
    client = make_client(tmp_path / "t.db")
    payload = topology(
        nodes=[
            {"id": "n1", "cpu_capacity": 8},
            {"id": "n2", "cpu_capacity": 4},
            {"id": "n3", "cpu_capacity": 8},
        ],
        containers=[
            {"id": "c1", "service": "svc", "min_ready": 1, "node_id": "n1", "cpu_request": 2},
        ],
    )
    with client:
        client.post("/api/topology", json=payload)
        client.post("/api/nodes/n3/power-off")
        state = client.post("/api/nodes/n1/drain").json()
        migration = state["open_drains"][0]["pending_decisions"][0]
        assert migration["target_node_id"] == "n2"


def test_powered_off_node_cannot_drain_or_resume(tmp_path):
    client = make_client(tmp_path / "t.db")
    payload = topology(
        nodes=[
            {"id": "n1", "cpu_capacity": 8},
            {"id": "n2", "cpu_capacity": 8},
        ],
        containers=[
            {"id": "c1", "service": "svc", "min_ready": 1, "node_id": "n1", "cpu_request": 2},
        ],
    )
    with client:
        client.post("/api/topology", json=payload)
        state = client.post("/api/nodes/n1/drain").json()
        mid = state["open_drains"][0]["pending_decisions"][0]["id"]
        state = client.post(
            f"/api/migrations/{mid}/resolve", json={"outcome": "failed"}
        ).json()
        assert state["open_drains"][0]["status"] == "incomplete"

        client.post("/api/nodes/n1/power-off")
        state = client.get("/api/state").json()
        n1 = node(state, "n1")
        assert n1["can_drain"] is False
        assert n1["can_resume"] is False
        resp = client.post("/api/nodes/n1/drain")
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "NODE_POWERED_OFF"
        resp = client.post("/api/nodes/n1/resume")
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "NODE_POWERED_OFF"

        # After power returns the drain can be resumed again.
        client.post("/api/nodes/n1/power-on")
        state = client.post("/api/nodes/n1/resume").json()
        assert state["open_drains"][0]["status"] == "active"
        assert len(state["open_drains"][0]["pending_decisions"]) == 1


def test_power_on_recounts_only_replicas_still_on_node(tmp_path):
    client = make_client(tmp_path / "t.db")
    payload = topology(
        nodes=[
            {"id": "n1", "cpu_capacity": 8},
            {"id": "n2", "cpu_capacity": 8},
        ],
        containers=[
            {"id": "c1", "service": "svc", "min_ready": 1, "node_id": "n1", "cpu_request": 2},
            {"id": "c2", "service": "svc", "min_ready": 1, "node_id": "n1", "cpu_request": 2},
        ],
    )
    with client:
        client.post("/api/topology", json=payload)
        # Drain n1 so both replicas get outbound starts, then cut n1's power:
        # the starts target n2 (live) and stay decidable during the outage.
        state = client.post("/api/nodes/n1/drain").json()
        client.post("/api/nodes/n1/power-off")
        assert service(client.get("/api/state").json(), "svc")["ready"] == 0

        # One start completes during the outage: its replica moves to n2.
        pending = client.get("/api/state").json()["open_drains"][0]["pending_decisions"]
        first = next(m for m in pending if m["container_id"] == "c1")
        client.post(f"/api/migrations/{first['id']}/resolve", json={"outcome": "ready"})
        state = client.get("/api/state").json()
        assert service(state, "svc")["ready"] == 1  # only the moved replica

        # Power returns: c2 (still on n1) counts again, c1 stays on n2.
        state = client.post("/api/nodes/n1/power-on").json()
        svc = service(state, "svc")
        assert svc["ready"] == 2
        c1 = next(c for c in state["containers"] if c["id"] == "c1")
        c2 = next(c for c in state["containers"] if c["id"] == "c2")
        assert c1["node_id"] == "n2"
        assert c2["node_id"] == "n1"


def test_interleaving_resolve_first_then_power_cut(tmp_path):
    client = make_client(tmp_path / "t.db")
    payload = topology(
        nodes=[
            {"id": "n1", "cpu_capacity": 8},
            {"id": "n2", "cpu_capacity": 8},
        ],
        containers=[
            {"id": "c1", "service": "svc", "min_ready": 1, "node_id": "n1", "cpu_request": 2},
        ],
    )
    with client:
        client.post("/api/topology", json=payload)
        state = client.post("/api/nodes/n1/drain").json()
        mid = state["open_drains"][0]["pending_decisions"][0]["id"]

        # The ready decision commits first: the replica lands on n2.
        state = client.post(
            f"/api/migrations/{mid}/resolve", json={"outcome": "ready"}
        ).json()
        assert next(c for c in state["containers"] if c["id"] == "c1")["node_id"] == "n2"
        # Cutting n2 now simply takes that replica out of the ready count.
        state = client.post("/api/nodes/n2/power-off").json()
        svc = service(state, "svc")
        assert svc["ready"] == 0
        assert svc["offline_replicas"] == 1
        assert next(c for c in state["containers"] if c["id"] == "c1")["node_id"] == "n2"


def test_interleaving_power_cut_first_then_resolve_is_stale(tmp_path):
    client = make_client(tmp_path / "t.db")
    payload = topology(
        nodes=[
            {"id": "n1", "cpu_capacity": 8},
            {"id": "n2", "cpu_capacity": 8},
        ],
        containers=[
            {"id": "c1", "service": "svc", "min_ready": 1, "node_id": "n1", "cpu_request": 2},
        ],
    )
    with client:
        client.post("/api/topology", json=payload)
        state = client.post("/api/nodes/n1/drain").json()
        mid = state["open_drains"][0]["pending_decisions"][0]["id"]

        # The power cut commits first: the pending start dies with the node.
        client.post("/api/nodes/n2/power-off")
        resp = client.post(f"/api/migrations/{mid}/resolve", json={"outcome": "ready"})
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "MIGRATION_NOT_PENDING"
        state = client.get("/api/state").json()
        # Capacity and placement follow the committed power cut: reservation
        # released, source replica still on n1.
        assert node(state, "n2")["reserved_cpu"] == 0
        assert next(c for c in state["containers"] if c["id"] == "c1")["node_id"] == "n1"


def test_restart_preserves_power_and_drain_state(tmp_path):
    db = tmp_path / "restart.db"
    payload = topology(
        nodes=[
            {"id": "n1", "cpu_capacity": 8},
            {"id": "n2", "cpu_capacity": 8},
        ],
        containers=[
            {"id": "c1", "service": "svc", "min_ready": 1, "node_id": "n1", "cpu_request": 2},
        ],
    )
    first = make_client(db)
    with first:
        first.post("/api/topology", json=payload)
        state = first.post("/api/nodes/n1/drain").json()
        mid = state["open_drains"][0]["pending_decisions"][0]["id"]
        first.post(f"/api/migrations/{mid}/resolve", json={"outcome": "failed"})
        first.post("/api/nodes/n1/power-off")

    second = make_client(db)
    with second:
        state = second.get("/api/state").json()
        assert state["powered_off_node_id"] == "n1"
        assert node(state, "n1")["powered_on"] is False
        assert state["open_drains"][0]["status"] == "incomplete"
        assert service(state, "svc")["ready"] == 0
        # Still cannot continue the drain while the node is down.
        resp = second.post("/api/nodes/n1/resume")
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "NODE_POWERED_OFF"
        # Restore power, then the drain resumes and completes.
        second.post("/api/nodes/n1/power-on")
        state = second.post("/api/nodes/n1/resume").json()
        mid = state["open_drains"][0]["pending_decisions"][0]["id"]
        state = second.post(
            f"/api/migrations/{mid}/resolve", json={"outcome": "ready"}
        ).json()
        assert state["drains"][0]["status"] == "completed"


def test_restart_after_cut_keeps_released_reservation(tmp_path):
    db = tmp_path / "restart2.db"
    payload = topology(
        nodes=[
            {"id": "n1", "cpu_capacity": 8},
            {"id": "n2", "cpu_capacity": 8},
        ],
        containers=[
            {"id": "c1", "service": "svc", "min_ready": 1, "node_id": "n1", "cpu_request": 3},
        ],
    )
    first = make_client(db)
    with first:
        first.post("/api/topology", json=payload)
        first.post("/api/nodes/n1/drain")
        first.post("/api/nodes/n2/power-off")

    second = make_client(db)
    with second:
        state = second.get("/api/state").json()
        assert node(state, "n2")["reserved_cpu"] == 0
        assert node(state, "n2")["remaining_cpu"] == 8
        assert state["blocked_containers"][0]["reason_code"] == "TARGET_POWERED_OFF"


def test_resolve_ready_guard_against_dead_target(tmp_path):
    """Defence in depth: if a starting migration ever points at a node that
    is already off (only constructible by bypassing the API), resolving it
    ready must fail the migration instead of cutting over, and the response
    must reflect the committed state."""
    import sqlite3

    db = tmp_path / "guard.db"
    client = make_client(db)
    payload = topology(
        nodes=[
            {"id": "n1", "cpu_capacity": 8},
            {"id": "n2", "cpu_capacity": 8},
        ],
        containers=[
            {"id": "c1", "service": "svc", "min_ready": 1, "node_id": "n1", "cpu_request": 2},
        ],
    )
    with client:
        client.post("/api/topology", json=payload)
        state = client.post("/api/nodes/n1/drain").json()
        mid = state["open_drains"][0]["pending_decisions"][0]["id"]

    # Bypass the API: flip the target's power without failing its inbound
    # reservations, creating the state the guard exists for.
    conn = sqlite3.connect(db)
    conn.execute("UPDATE nodes SET powered_on = 0 WHERE id = 'n2'")
    conn.commit()
    conn.close()

    with client:
        state = client.post(
            f"/api/migrations/{mid}/resolve", json={"outcome": "ready"}
        ).json()
        # The response itself (not just a later read) shows the failure.
        assert state["open_drains"][0]["status"] == "incomplete"
        assert state["blocked_containers"][0]["reason_code"] == "TARGET_POWERED_OFF"
        assert node(state, "n2")["reserved_cpu"] == 0
        c1 = next(c for c in state["containers"] if c["id"] == "c1")
        assert c1["node_id"] == "n1"


def test_import_and_reset_clear_power_state(tmp_path):
    client = make_client(tmp_path / "t.db")
    payload = topology(
        nodes=[{"id": "n1", "cpu_capacity": 4}, {"id": "n2", "cpu_capacity": 4}],
        containers=[
            {"id": "c1", "service": "svc", "min_ready": 1, "node_id": "n1", "cpu_request": 1},
        ],
    )
    with client:
        client.post("/api/topology", json=payload)
        client.post("/api/nodes/n1/power-off")
        state = client.post("/api/topology", json=payload).json()
        assert state["powered_off_node_id"] is None
        assert all(n["powered_on"] for n in state["nodes"])
        assert service(state, "svc")["ready"] == 1

        client.post("/api/nodes/n2/power-off")
        state = client.post("/api/reset").json()
        assert state["topology_loaded"] is False
        assert state["powered_off_node_id"] is None
