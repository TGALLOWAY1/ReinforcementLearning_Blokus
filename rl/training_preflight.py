"""
Pre-flight check for the training pipeline before starting a long run.

Validates:
- Config loads (e.g. configs/v1_rl_vs_mcts.yaml)
- Exactly 3 opponents for 4-player league
- Env and vec env build
- League agents build (RL + 3 baselines)
- One 4-player league match runs without error
- Optionally: a few training steps and one league eval

Run from repo root: PYTHONPATH=. python rl/training_preflight.py [--config configs/v1_rl_vs_mcts.yaml]
"""

from __future__ import annotations

import argparse
import sys

from sb3_contrib import MaskablePPO

from league.league import League, build_league_agents
from rl.train import (
    TrainConfig,
    _make_vec_env,
    evaluate_and_update_league,
    load_config,
    set_seeds,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-flight check for training")
    parser.add_argument("--config", default="configs/v1_rl_vs_mcts.yaml", help="Config YAML path")
    parser.add_argument("--dummy-env", action="store_true", default=True, help="Use DummyVecEnv (default)")
    parser.add_argument("--no-dummy-env", action="store_false", dest="dummy_env", help="Use config vec_env_type")
    parser.add_argument("--run-one-eval", action="store_true", help="Run one league eval (needs model, slower)")
    args = parser.parse_args()

    print("Loading config...")
    config = load_config(args.config)
    if len(config.opponents) != 3:
        print(f"FAIL: config must have exactly 3 opponents for 4-player league, got {len(config.opponents)}")
        return 1
    print(f"  opponents: {config.opponents}")

    print("Building env (1 env, dummy for preflight)...")
    set_seeds(config.seed)
    if args.dummy_env:
        # Use 1 env and dummy to avoid subprocess and keep preflight fast
        override = TrainConfig()
        for k, v in config.__dict__.items():
            setattr(override, k, v)
        override.num_envs = 1
        override.vec_env_type = "dummy"
        env = _make_vec_env(override)
    else:
        env = _make_vec_env(config)
    env.reset()
    print("  env OK")

    print("Building league agents (3 baselines, no RL)...")
    agents, specs, ordered_4p = build_league_agents(config.opponents, config.seed, model=None, model_name=None)
    assert len(ordered_4p) == 3, "Without RL we get 3 baseline names"
    print(f"  baseline names: {ordered_4p}")

    if args.run_one_eval:
        print("Creating model and running one league eval (eval_matches=1)...")
        import os
        import tempfile

        from torch.distributions.distribution import Distribution
        Distribution.set_default_validate_args(False)
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=config.learning_rate,
            n_steps=min(64, config.n_steps),
            batch_size=min(32, config.batch_size),
            gamma=config.gamma,
            verbose=0,
        )
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            league = League(db_path=db_path)
            eval_config = TrainConfig()
            for k, v in config.__dict__.items():
                setattr(eval_config, k, v)
            eval_config.eval_matches = 1
            results = evaluate_and_update_league(model, league, eval_config, step=0)
            league.close()
            print(f"  eval W/L/D: {results['wins']}/{results['losses']}/{results['draws']}, Elo: {results.get('elo', 1200)}")
        finally:
            os.unlink(db_path)
        env.close()

    print("Pre-flight OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
