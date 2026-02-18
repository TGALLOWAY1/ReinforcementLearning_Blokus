#!/usr/bin/env python3
import argparse
import json
import time
import urllib.request


def _post(url: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload or {}).encode('utf-8')
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode('utf-8'))


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode('utf-8'))


def create_game(api_url: str, budget_ms: int) -> str:
    payload = {
        "players": [
            {"player": "RED", "agent_type": "mcts", "agent_config": {"time_budget_ms": budget_ms}},
            {"player": "BLUE", "agent_type": "random", "agent_config": {}},
        ],
        "auto_start": True,
    }
    data = _post(f"{api_url}/api/games", payload)
    return data["game_id"]


def wait_or_stop(api_url: str, game_id: str, target_moves: int = 20, timeout_s: int = 45):
    start = time.time()
    while time.time() - start < timeout_s:
        data = _get(f"{api_url}/api/games/{game_id}")
        if data.get("game_over") or data.get("move_count", 0) >= target_moves:
            break
        time.sleep(1.0)

    _post(f"{api_url}/api/games/{game_id}/finish")
    return _get(f"{api_url}/api/games/{game_id}")


def main(api_url: str):
    results = []
    for budget in [1000, 3000, 5000]:
        game_id = create_game(api_url, budget)
        state = wait_or_stop(api_url, game_id)
        results.append((budget, game_id, state.get("winner"), state.get("move_count")))
        print(f"budget={budget} game_id={game_id} winner={state.get('winner')} moves={state.get('move_count')}")
    print("Completed sample games:")
    for row in results:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    main(args.api_url)
