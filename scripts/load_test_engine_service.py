#!/usr/bin/env python3
import argparse
import json
import statistics
import time
import urllib.request

SAMPLE_PAYLOAD = {
    "gameState": {
        "board": [[0 for _ in range(20)] for _ in range(20)],
        "pieces_used": {"RED": [], "BLUE": [], "GREEN": [], "YELLOW": []},
        "current_player": "RED",
        "move_count": 0,
    },
    "legalMoves": [{"piece_id": 1, "orientation": 0, "anchor_row": 0, "anchor_col": 0}],
    "timeBudgetMs": 1000,
}


def percentile(data, pct):
    if not data:
        return 0
    data = sorted(data)
    idx = int((len(data) - 1) * pct)
    return data[idx]


def run(engine_url, requests, budgets):
    for budget in budgets:
        latencies = []
        errors = 0
        for _ in range(requests):
            payload = dict(SAMPLE_PAYLOAD)
            payload["timeBudgetMs"] = budget
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(f"{engine_url.rstrip('/')}/think", data=body, method="POST", headers={"Content-Type": "application/json"})
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=max(2, budget / 1000 + 2)) as resp:
                    _ = resp.read()
                    latencies.append((time.perf_counter() - start) * 1000)
            except Exception:
                errors += 1

        print(f"budget={budget}ms count={requests} errors={errors} error_rate={errors/requests:.3f}")
        if latencies:
            print(f"  p50={percentile(latencies,0.5):.2f}ms p95={percentile(latencies,0.95):.2f}ms avg={statistics.mean(latencies):.2f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-url", required=True)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1000, 3000, 5000])
    args = parser.parse_args()
    run(args.engine_url, args.requests, args.budgets)
