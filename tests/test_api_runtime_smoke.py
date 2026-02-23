from fastapi.testclient import TestClient

from webapi.app import create_app


def _deploy_payload():
    return {
        "players": [
            {"player": "RED", "agent_type": "human", "agent_config": {}},
            {"player": "BLUE", "agent_type": "mcts", "agent_config": {"difficulty": "easy"}},
            {"player": "GREEN", "agent_type": "mcts", "agent_config": {"difficulty": "medium"}},
            {"player": "YELLOW", "agent_type": "mcts", "agent_config": {"difficulty": "hard"}}
        ],
        "auto_start": True
    }


def test_deploy_runtime_health_and_game_creation():
    app = create_app(profile="deploy", include_research_routes=False)
    client = TestClient(app)

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json().get("ok") is True
    assert health_resp.json().get("profile") == "deploy"

    create_resp = client.post("/api/games", json=_deploy_payload())
    assert create_resp.status_code == 200
    payload = create_resp.json()
    assert "game_id" in payload
    assert payload["game_state"]["current_player"] == "RED"


def test_deploy_runtime_rejects_invalid_config():
    app = create_app(profile="deploy", include_research_routes=False)
    client = TestClient(app)

    invalid_payload = {
        "players": [
            {"player": "RED", "agent_type": "human", "agent_config": {}},
            {"player": "BLUE", "agent_type": "random", "agent_config": {}},
            {"player": "GREEN", "agent_type": "mcts", "agent_config": {"difficulty": "medium"}},
            {"player": "YELLOW", "agent_type": "mcts", "agent_config": {"difficulty": "hard"}}
        ],
        "auto_start": True
    }

    resp = client.post("/api/games", json=invalid_payload)
    assert resp.status_code == 400
    assert "only supports agent_type values" in str(resp.json())


def test_deploy_runtime_move_route_smoke():
    app = create_app(profile="deploy", include_research_routes=False)
    client = TestClient(app)

    create_resp = client.post("/api/games", json=_deploy_payload())
    assert create_resp.status_code == 200
    game_state = create_resp.json()["game_state"]
    game_id = create_resp.json()["game_id"]
    legal_moves = game_state.get("legal_moves", [])
    assert len(legal_moves) > 0

    first_move = legal_moves[0]
    move_resp = client.post(
        f"/api/games/{game_id}/move",
        json={
            "player": "RED",
            "move": {
                "piece_id": first_move["piece_id"],
                "orientation": first_move["orientation"],
                "anchor_row": first_move["anchor_row"],
                "anchor_col": first_move["anchor_col"]
            }
        },
    )
    assert move_resp.status_code == 200
    assert move_resp.json().get("success") is True
