from __future__ import annotations

import concurrent.futures
import random
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from app.main import create_app  # noqa: E402


def _client(db: Path) -> TestClient:
    return TestClient(create_app(db))


def test_two_power_cuts_race_and_only_one_wins(tmp_path):
    db = tmp_path / "power-race.db"
    payload = {
        "nodes": [
            {"id": "n1", "cpu_capacity": 8},
            {"id": "n2", "cpu_capacity": 8},
        ],
        "containers": [],
    }
    with _client(db) as client:
        assert client.post("/api/topology", json=payload).status_code == 200

    def cut(node_id: str) -> int:
        with _client(db) as other:
            return other.post(f"/api/nodes/{node_id}/power-off").status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(cut, ["n1", "n2"]))
    assert sorted(statuses) == [200, 409]

    with _client(db) as client:
        state = client.get("/api/state").json()
    off_nodes = [n for n in state["nodes"] if not n["powered_on"]]
    assert len(off_nodes) == 1
    assert state["powered_off_node_id"] == off_nodes[0]["id"]


def test_power_cut_and_ready_decision_race_stays_coherent(tmp_path):
    db = tmp_path / "cut-vs-ready.db"
    payload = {
        "nodes": [
            {"id": "n1", "cpu_capacity": 8},
            {"id": "n2", "cpu_capacity": 8},
        ],
        "containers": [
            {"id": "c1", "service": "svc", "min_ready": 1, "node_id": "n1", "cpu_request": 2},
        ],
    }
    with _client(db) as client:
        assert client.post("/api/topology", json=payload).status_code == 200
        state = client.post("/api/nodes/n1/drain").json()
        mid = state["open_drains"][0]["pending_decisions"][0]["id"]

    def cut() -> int:
        with _client(db) as other:
            return other.post("/api/nodes/n2/power-off").status_code

    def resolve() -> int:
        with _client(db) as other:
            return other.post(
                f"/api/migrations/{mid}/resolve", json={"outcome": "ready"}
            ).status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda fn: fn(), [cut, resolve]))
    cut_status, resolve_status = results

    with _client(db) as client:
        state = client.get("/api/state").json()
    n2 = next(n for n in state["nodes"] if n["id"] == "n2")
    container = next(c for c in state["containers"] if c["id"] == "c1")
    svc = next(s for s in state["services"] if s["name"] == "svc")

    assert cut_status == 200
    assert n2["powered_on"] is False
    # Whichever committed first decides placement and capacity; the loser
    # gets a coherent error instead of a stale write.
    if resolve_status == 200:
        # Ready committed first: the replica landed on n2 before the cut.
        assert container["node_id"] == "n2"
        assert svc["ready"] == 0  # powered off, so not counted
        assert svc["offline_replicas"] == 1
    else:
        # The cut committed first: the start died, the source replica stayed.
        assert resolve_status == 409
        assert container["node_id"] == "n1"
        assert n2["reserved_cpu"] == 0
        assert svc["ready"] == 1  # n1 is live; the old replica still serves


def test_two_nodes_drained_at_the_same_time_never_overcommit(tmp_path):
    db = tmp_path / "concurrent.db"
    payload = {
        "nodes": [
            {"id": "n1", "cpu_capacity": 10},
            {"id": "n2", "cpu_capacity": 10},
            {"id": "n3", "cpu_capacity": 10},
        ],
        "containers": [
            {"id": f"c{i}", "service": "s", "min_ready": 1, "node_id": "n1", "cpu_request": 3}
            for i in range(3)
        ]
        + [
            {"id": f"d{i}", "service": "t", "min_ready": 1, "node_id": "n2", "cpu_request": 3}
            for i in range(3)
        ],
    }
    with _client(db) as client:
        assert client.post("/api/topology", json=payload).status_code == 200

    def drain(node_id: str) -> int:
        # Separate apps/connections: the SQLite write lock is the only
        # coordination between them.
        with _client(db) as other:
            return other.post(f"/api/nodes/{node_id}/drain").status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(drain, ["n1", "n2"]))
    assert statuses == [200, 200]

    with _client(db) as client:
        state = client.get("/api/state").json()

    for node in state["nodes"]:
        assert node["used_cpu"] <= node["cpu_capacity"]
        assert node["used_cpu"] == node["serving_cpu"] + node["reserved_cpu"]
    # Every logical replica's CPU is committed exactly once as "serving" on
    # its source, plus a second, separate reservation per in-flight replica on
    # its target (that overlap is what makes cutover safe).
    assert sum(node["serving_cpu"] for node in state["nodes"]) == 18
    pending = [m for drain in state["open_drains"] for m in drain["pending_decisions"]]
    assert sum(node["reserved_cpu"] for node in state["nodes"]) == len(pending) * 3


def _random_topology(rng: random.Random) -> dict:
    nodes = [
        {"id": f"n{i}", "cpu_capacity": rng.choice([4, 6, 8, 12])}
        for i in range(rng.randint(2, 4))
    ]
    services = [f"svc{i}" for i in range(rng.randint(1, 3))]
    containers = [
        {
            "id": f"c{i}",
            "service": rng.choice(services),
            "min_ready": 1,
            "node_id": rng.choice(nodes)["id"],
            "cpu_request": rng.randint(1, 4),
        }
        for i in range(rng.randint(1, 8))
    ]
    return {"nodes": nodes, "containers": containers}


@pytest.mark.parametrize("seed", [3, 11])
def test_random_workload_keeps_invariants(tmp_path, seed):
    rng = random.Random(seed)
    db = tmp_path / f"prop-{seed}.db"

    with _client(db) as client:
        accepted = None
        for _ in range(200):
            candidate = _random_topology(rng)
            response = client.post("/api/topology", json=candidate)
            if response.status_code == 200:
                accepted = response.json()
                break
        assert accepted is not None

        for _ in range(30):
            state = client.get("/api/state").json()
            for node in state["nodes"]:
                assert node["used_cpu"] == node["serving_cpu"] + node["reserved_cpu"]
                assert node["used_cpu"] <= node["cpu_capacity"]
                assert node["remaining_cpu"] == node["cpu_capacity"] - node["used_cpu"]
                if node["is_empty"]:
                    assert node["serving_containers"] == 0
                    assert node["reserved_cpu"] == 0
                    assert node["inbound_reservations"] == []
            for service in state["services"]:
                assert service["ready"] >= 0
                assert service["starting_replicas"] >= 0
                assert service["offline_replicas"] >= 0
            # At most one node is ever powered off, and nothing in flight
            # may target it.
            off_nodes = [n for n in state["nodes"] if not n["powered_on"]]
            assert len(off_nodes) <= 1
            assert state["powered_off_node_id"] == (
                off_nodes[0]["id"] if off_nodes else None
            )
            for off in off_nodes:
                assert off["inbound_reservations"] == []
                assert off["can_drain"] is False
                assert off["can_resume"] is False
            pending_targets = {
                m["target_node_id"]
                for drain in state["open_drains"]
                for m in drain["pending_decisions"]
            }
            assert not ({n["id"] for n in off_nodes} & pending_targets)

            action = rng.choice(
                ["drain", "resolve", "resolve", "resume", "restore",
                 "poweroff", "poweron"]
            )
            if action == "drain":
                candidates = [n["id"] for n in state["nodes"] if n["can_drain"]]
                if candidates:
                    client.post(f"/api/nodes/{rng.choice(candidates)}/drain")
            elif action == "resume":
                candidates = [n["id"] for n in state["nodes"] if n["can_resume"]]
                if candidates:
                    response = client.post(f"/api/nodes/{rng.choice(candidates)}/resume")
                    assert response.status_code == 200, response.text
            elif action == "restore":
                candidates = [n["id"] for n in state["nodes"] if n["can_restore"]]
                if candidates:
                    response = client.post(f"/api/nodes/{rng.choice(candidates)}/restore")
                    assert response.status_code == 200, response.text
            elif action == "poweroff":
                candidates = [n["id"] for n in state["nodes"] if n["can_power_off"]]
                if candidates:
                    response = client.post(
                        f"/api/nodes/{rng.choice(candidates)}/power-off"
                    )
                    assert response.status_code == 200, response.text
            elif action == "poweron":
                candidates = [n["id"] for n in state["nodes"] if n["can_power_on"]]
                if candidates:
                    response = client.post(
                        f"/api/nodes/{rng.choice(candidates)}/power-on"
                    )
                    assert response.status_code == 200, response.text
            else:
                pending = [
                    m for drain in state["open_drains"] for m in drain["pending_decisions"]
                ]
                if pending:
                    migration = rng.choice(pending)
                    response = client.post(
                        f"/api/migrations/{migration['id']}/resolve",
                        json={"outcome": rng.choice(["ready", "failed"])},
                    )
                    assert response.status_code == 200, response.text

    # Restarting must not change any of the capacity accounting.
    with _client(db) as reopened:
        state = reopened.get("/api/state").json()
        for node in state["nodes"]:
            assert node["used_cpu"] <= node["cpu_capacity"]
            if node["is_empty"]:
                assert node["serving_containers"] == 0
        off_nodes = [n for n in state["nodes"] if not n["powered_on"]]
        assert len(off_nodes) <= 1
        pending_ids = {
            m["id"]
            for drain in state["open_drains"]
            for m in drain["pending_decisions"]
        }
        injected = [
            m for drain in state["drains"] for m in drain["pending_decisions"]
        ]
        # Closed drains never expose pending decisions.
        assert pending_ids == {m["id"] for m in injected}
