"""Tests for progressive league sampling schedule."""

import json
from pathlib import Path

import numpy as np

from league.pdl import LeagueCheckpoint, LeagueSamplingConfig, WindowScheduleConfig, CheckpointOpponentSampler


def _write_registry(path: Path, steps):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for step in steps:
            entry = LeagueCheckpoint(
                checkpoint_id=f"ckpt_{step}",
                path=f"/tmp/ckpt_{step}.zip",
                step=step,
                timestamp="2025-01-01T00:00:00Z",
            )
            handle.write(json.dumps(entry.to_dict()) + "\n")


def _sample_mean_step(registry_path: Path, state_path: Path, step: int, samples: int = 500) -> float:
    sampling = LeagueSamplingConfig(
        recent_band_pct=0.7,
        mid_band_pct=0.25,
        old_band_pct=0.05,
        recent_window_frac=0.2,
        mid_window_frac=0.3,
        old_window_frac=0.5,
    ).normalize()
    schedule = WindowScheduleConfig(
        schedule_type="linear",
        start_window_frac=1.0,
        end_window_frac=0.2,
        schedule_steps=100,
    )
    sampler = CheckpointOpponentSampler(
        registry_path=registry_path,
        state_path=state_path,
        sampling_config=sampling,
        window_schedule=schedule,
        seed=7,
    )
    sampler.refresh()
    sampler.override_step(step)
    picks = [sampler.sample_entries(1)[0].step for _ in range(samples)]
    return float(np.mean(picks))


def test_league_sampling_shifts_recent(tmp_path: Path):
    registry_path = tmp_path / "league_registry.jsonl"
    state_path = tmp_path / "league_state.json"
    steps = list(range(0, 101))
    _write_registry(registry_path, steps)

    early_mean = _sample_mean_step(registry_path, state_path, step=10)
    late_mean = _sample_mean_step(registry_path, state_path, step=90)

    assert late_mean > early_mean + 5.0, (
        f"Expected later sampling to skew recent: early_mean={early_mean}, late_mean={late_mean}"
    )
