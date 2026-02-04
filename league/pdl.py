"""
Progressive Difficulty League (PDL) utilities for checkpoint self-play.
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sb3_contrib import MaskablePPO
except Exception:  # pragma: no cover - optional at import time
    MaskablePPO = None  # type: ignore

from agents.registry import AgentProtocol

logger = logging.getLogger(__name__)


@dataclass
class LeagueSamplingConfig:
    recent_band_pct: float = 0.7
    mid_band_pct: float = 0.25
    old_band_pct: float = 0.05

    recent_window_frac: float = 0.2
    mid_window_frac: float = 0.3
    old_window_frac: float = 0.5

    allow_duplicates: bool = False

    def normalize(self) -> "LeagueSamplingConfig":
        total_pct = self.recent_band_pct + self.mid_band_pct + self.old_band_pct
        if total_pct <= 0:
            raise ValueError("Band pct sum must be > 0")
        self.recent_band_pct /= total_pct
        self.mid_band_pct /= total_pct
        self.old_band_pct /= total_pct

        total_window = self.recent_window_frac + self.mid_window_frac + self.old_window_frac
        if total_window <= 0:
            raise ValueError("Window frac sum must be > 0")
        self.recent_window_frac /= total_window
        self.mid_window_frac /= total_window
        self.old_window_frac /= total_window
        return self


@dataclass
class WindowScheduleConfig:
    schedule_type: str = "linear"  # linear | fixed
    start_window_frac: float = 1.0
    end_window_frac: float = 0.3
    schedule_steps: int = 500_000
    min_window_frac: float = 0.1
    max_window_frac: float = 1.0

    def current_window_frac(self, step: int) -> float:
        if self.schedule_type == "fixed":
            frac = self.start_window_frac
        else:
            if self.schedule_steps <= 0:
                frac = self.end_window_frac
            else:
                progress = min(max(step / float(self.schedule_steps), 0.0), 1.0)
                frac = self.start_window_frac + (self.end_window_frac - self.start_window_frac) * progress
        frac = max(self.min_window_frac, min(self.max_window_frac, frac))
        return float(frac)


@dataclass
class Stage3LeagueConfig:
    enabled: bool = True
    league_dir: Optional[str] = None
    registry_filename: str = "league_registry.jsonl"
    state_filename: str = "league_state.json"
    save_every_steps: int = 50_000
    max_checkpoints_to_keep: int = 50
    keep_every_k: Optional[int] = None
    min_checkpoints: int = 3
    lru_cache_size: int = 4
    opponent_device: str = "auto"
    discover_on_start: bool = True
    eval_pool_size: int = 3
    eval_pool_strategy: str = "old_mid_recent"
    sampling: LeagueSamplingConfig = field(default_factory=LeagueSamplingConfig)
    window_schedule: WindowScheduleConfig = field(default_factory=WindowScheduleConfig)

    def resolve_league_dir(self, checkpoint_dir: str) -> Path:
        return Path(self.league_dir or checkpoint_dir)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Stage3LeagueConfig":
        config = cls()
        for key, value in raw.items():
            if key == "sampling" and isinstance(value, dict):
                config.sampling = LeagueSamplingConfig(**value).normalize()
            elif key == "window_schedule" and isinstance(value, dict):
                config.window_schedule = WindowScheduleConfig(**value)
            elif hasattr(config, key):
                setattr(config, key, value)
        # Ensure normalization if sampling set directly
        config.sampling.normalize()
        return config


@dataclass
class LeagueCheckpoint:
    checkpoint_id: str
    path: str
    step: int
    timestamp: str
    elo: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "path": self.path,
            "step": self.step,
            "timestamp": self.timestamp,
            "elo": self.elo,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "LeagueCheckpoint":
        return cls(
            checkpoint_id=raw["checkpoint_id"],
            path=raw["path"],
            step=int(raw["step"]),
            timestamp=raw.get("timestamp", ""),
            elo=raw.get("elo"),
            metadata=raw.get("metadata", {}),
        )


class CheckpointPolicyCache:
    def __init__(self, max_size: int = 4, device: str = "auto") -> None:
        self.max_size = max_size
        self.device = device
        self._cache: "OrderedDict[str, Any]" = OrderedDict()

    def get(self, path: str):
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        if MaskablePPO is None:
            raise RuntimeError("MaskablePPO not available for checkpoint loading")
        model = MaskablePPO.load(path, device=self.device)
        if hasattr(model, "policy"):
            model.policy.set_training_mode(False)
        self._cache[path] = model
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        return model


class CheckpointOpponentPolicy(AgentProtocol):
    def __init__(self, checkpoint_path: str, cache: CheckpointPolicyCache) -> None:
        self.checkpoint_path = checkpoint_path
        self._cache = cache

    def act(self, observation: np.ndarray, legal_mask: np.ndarray, env=None) -> Optional[int]:
        if legal_mask is None or not legal_mask.any():
            return None
        model = self._cache.get(self.checkpoint_path)
        action, _ = model.predict(observation, action_masks=legal_mask, deterministic=True)
        return int(action)


class LeagueRegistry:
    def __init__(self, registry_path: Path) -> None:
        self.registry_path = registry_path
        self.entries: List[LeagueCheckpoint] = []
        self._path_index: Dict[str, LeagueCheckpoint] = {}
        self._load()

    def _load(self) -> None:
        self.entries = []
        self._path_index = {}
        if not self.registry_path.exists():
            return
        with open(self.registry_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entry = LeagueCheckpoint.from_dict(data)
                except json.JSONDecodeError:
                    continue
                self._path_index[entry.path] = entry
                self.entries.append(entry)
        self.entries.sort(key=lambda e: e.step)

    def _append(self, entry: LeagueCheckpoint) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_dict()) + "\n")

    def _rewrite(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.registry_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(json.dumps(entry.to_dict()) + "\n")
        os.replace(tmp_path, self.registry_path)

    def add(self, entry: LeagueCheckpoint) -> None:
        if entry.path in self._path_index:
            return
        self.entries.append(entry)
        self.entries.sort(key=lambda e: e.step)
        self._path_index[entry.path] = entry
        self._append(entry)

    def prune(self, max_keep: int, keep_every_k: Optional[int] = None) -> List[LeagueCheckpoint]:
        if max_keep <= 0:
            return []
        if len(self.entries) <= max_keep:
            return []
        if keep_every_k:
            filtered = [e for e in self.entries if e.step % keep_every_k == 0]
        else:
            filtered = list(self.entries)
        if self.entries:
            latest = max(self.entries, key=lambda e: e.step)
            if latest.path not in {e.path for e in filtered}:
                filtered.append(latest)
        filtered.sort(key=lambda e: e.step)
        if len(filtered) > max_keep:
            filtered = filtered[-max_keep:]
        removed = [e for e in self.entries if e.path not in {f.path for f in filtered}]
        self.entries = filtered
        self._path_index = {e.path: e for e in self.entries}
        self._rewrite()
        return removed


class LeagueManager:
    def __init__(self, config: Stage3LeagueConfig, checkpoint_dir: str) -> None:
        self.config = config
        self.league_dir = config.resolve_league_dir(checkpoint_dir)
        self.registry_path = self.league_dir / config.registry_filename
        self.state_path = self.league_dir / config.state_filename
        self.registry = LeagueRegistry(self.registry_path)
        self._cache = CheckpointPolicyCache(max_size=config.lru_cache_size, device=config.opponent_device)

    def discover_checkpoints(self) -> int:
        if not self.league_dir.exists():
            return 0
        candidates = list(self.league_dir.glob("**/*.zip"))
        added = 0
        for path in candidates:
            if path.name.startswith("checkpoint_") or path.name.startswith("ep"):
                entry = self._build_entry_from_path(path)
                if entry and entry.path not in {e.path for e in self.registry.entries}:
                    self.registry.add(entry)
                    added += 1
        if added:
            logger.info("Discovered %s checkpoints for league", added)
        return added

    def register_checkpoint(
        self,
        checkpoint_path: Path,
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
        elo: Optional[float] = None,
    ) -> LeagueCheckpoint:
        entry = LeagueCheckpoint(
            checkpoint_id=f"step_{step}_{int(datetime.utcnow().timestamp())}",
            path=str(checkpoint_path),
            step=int(step),
            timestamp=datetime.utcnow().isoformat(),
            elo=elo,
            metadata=metadata or {},
        )
        self.registry.add(entry)
        removed = self.registry.prune(self.config.max_checkpoints_to_keep, self.config.keep_every_k)
        for entry in removed:
            try:
                entry_path = Path(entry.path)
                if _is_within(entry_path, self.league_dir):
                    if entry_path.exists():
                        os.remove(entry.path)
                    rng_path = str(entry_path.with_suffix(".rng.pkl"))
                    if os.path.exists(rng_path):
                        os.remove(rng_path)
                    meta_path = str(entry_path.with_suffix(".meta.json"))
                    if os.path.exists(meta_path):
                        os.remove(meta_path)
            except OSError:
                logger.warning("Failed to remove old checkpoint %s", entry.path)
        return entry

    def write_state(self, step: int, total_timesteps: int) -> None:
        state = {
            "step": int(step),
            "total_timesteps": int(total_timesteps),
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle)
        os.replace(tmp_path, self.state_path)

    def sample_opponents(self, num_opponents: int, step: int, seed: Optional[int] = None) -> List[LeagueCheckpoint]:
        sampler = CheckpointOpponentSampler(
            registry_path=self.registry_path,
            state_path=self.state_path,
            sampling_config=self.config.sampling,
            window_schedule=self.config.window_schedule,
            lru_cache_size=self.config.lru_cache_size,
            opponent_device=self.config.opponent_device,
            seed=seed,
        )
        sampler.refresh()
        sampler.override_step(step)
        return sampler.sample_entries(num_opponents)

    def build_opponent_agents(self, entries: Sequence[LeagueCheckpoint]) -> List[CheckpointOpponentPolicy]:
        return [CheckpointOpponentPolicy(entry.path, self._cache) for entry in entries]

    def _build_entry_from_path(self, path: Path) -> Optional[LeagueCheckpoint]:
        step = _infer_step_from_checkpoint(path)
        if step is None:
            return None
        return LeagueCheckpoint(
            checkpoint_id=f"step_{step}_{path.stem}",
            path=str(path),
            step=int(step),
            timestamp=datetime.utcnow().isoformat(),
            metadata={},
        )


class CheckpointOpponentSampler:
    def __init__(
        self,
        registry_path: Path,
        state_path: Path,
        sampling_config: LeagueSamplingConfig,
        window_schedule: WindowScheduleConfig,
        lru_cache_size: int = 4,
        opponent_device: str = "auto",
        seed: Optional[int] = None,
    ) -> None:
        self.registry_path = registry_path
        self.state_path = state_path
        self.sampling_config = sampling_config.normalize()
        self.window_schedule = window_schedule
        self.lru_cache_size = lru_cache_size
        self.opponent_device = opponent_device
        self.seed = seed

        self._registry_mtime: Optional[float] = None
        self._state_mtime: Optional[float] = None
        self._entries: List[LeagueCheckpoint] = []
        self._training_step: int = 0
        self._total_timesteps: int = 0
        self._sample_count: int = 0
        self._cache = CheckpointPolicyCache(max_size=lru_cache_size, device=opponent_device)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = CheckpointPolicyCache(max_size=self.lru_cache_size, device=self.opponent_device)
        return state

    def refresh(self) -> None:
        self._refresh_registry()
        self._refresh_state()

    def override_step(self, step: int) -> None:
        self._training_step = int(step)

    def build_opponents(self, num_opponents: int) -> Dict[str, AgentProtocol]:
        entries = self.sample_entries(num_opponents)
        agents = [CheckpointOpponentPolicy(entry.path, self._cache) for entry in entries]
        return {f"player_{i+1}": agents[i] for i in range(len(agents))}

    def __call__(self, seed: Optional[int] = None) -> Dict[str, AgentProtocol]:
        if seed is not None:
            self.seed = seed
        self.refresh()
        return self.build_opponents(3)

    def sample_entries(self, num_opponents: int) -> List[LeagueCheckpoint]:
        if not self._entries:
            raise RuntimeError("No checkpoints available for league sampling")
        base_seed = self.seed or 0
        rng = random.Random(base_seed + self._sample_count)
        self._sample_count += 1
        entries = list(self._entries)
        if not self.sampling_config.allow_duplicates and len(entries) >= num_opponents:
            chosen: List[LeagueCheckpoint] = []
            candidates = entries[:]
            for _ in range(num_opponents):
                entry = _sample_entry(candidates, self._training_step, self._total_timesteps, rng, self.window_schedule, self.sampling_config)
                chosen.append(entry)
                candidates = [e for e in candidates if e.path != entry.path]
            return chosen
        return [
            _sample_entry(entries, self._training_step, self._total_timesteps, rng, self.window_schedule, self.sampling_config)
            for _ in range(num_opponents)
        ]

    def _refresh_registry(self) -> None:
        if not self.registry_path.exists():
            return
        mtime = self.registry_path.stat().st_mtime
        if self._registry_mtime is not None and mtime == self._registry_mtime:
            return
        self._registry_mtime = mtime
        entries = []
        with open(self.registry_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entry = LeagueCheckpoint.from_dict(data)
                entries.append(entry)
        entries.sort(key=lambda e: e.step)
        self._entries = entries

    def _refresh_state(self) -> None:
        if not self.state_path.exists():
            return
        mtime = self.state_path.stat().st_mtime
        if self._state_mtime is not None and mtime == self._state_mtime:
            return
        self._state_mtime = mtime
        with open(self.state_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        self._training_step = int(data.get("step", 0))
        self._total_timesteps = int(data.get("total_timesteps", 0))


def _infer_step_from_checkpoint(path: Path) -> Optional[int]:
    name = path.stem
    if name.startswith("checkpoint_"):
        suffix = name.replace("checkpoint_", "")
        if suffix.isdigit():
            return int(suffix)
    if name.startswith("ep"):
        suffix = name.replace("ep", "")
        if suffix.isdigit():
            return int(suffix)
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            if "step" in meta:
                return int(meta["step"])
        except Exception:
            return None
    return None


def _window_entries(entries: Sequence[LeagueCheckpoint], step: int, schedule: WindowScheduleConfig) -> List[LeagueCheckpoint]:
    if not entries:
        return []
    latest_step = max(e.step for e in entries)
    target_step = max(step, latest_step)
    window_frac = schedule.current_window_frac(target_step)
    window_start = max(0, int(target_step * (1.0 - window_frac)))
    windowed = [e for e in entries if e.step >= window_start]
    if not windowed:
        return list(entries)
    return sorted(windowed, key=lambda e: e.step)


def _split_bands(entries: Sequence[LeagueCheckpoint], config: LeagueSamplingConfig) -> Tuple[List[LeagueCheckpoint], List[LeagueCheckpoint], List[LeagueCheckpoint]]:
    entries = list(entries)
    n = len(entries)
    if n == 0:
        return [], [], []

    old_count = int(round(n * config.old_window_frac))
    mid_count = int(round(n * config.mid_window_frac))
    # Ensure at least 1 in recent when possible
    recent_count = max(n - old_count - mid_count, 1)
    if old_count + mid_count + recent_count > n:
        overflow = old_count + mid_count + recent_count - n
        if mid_count >= overflow:
            mid_count -= overflow
        elif old_count >= overflow:
            old_count -= overflow
        else:
            recent_count = max(n - old_count - mid_count, 1)
    if old_count < 0:
        old_count = 0
    if mid_count < 0:
        mid_count = 0

    old_band = entries[:old_count]
    mid_band = entries[old_count:old_count + mid_count]
    recent_band = entries[old_count + mid_count:]
    if not recent_band and entries:
        recent_band = [entries[-1]]
    return old_band, mid_band, recent_band


def _sample_entry(
    entries: Sequence[LeagueCheckpoint],
    step: int,
    total_timesteps: int,
    rng: random.Random,
    schedule: WindowScheduleConfig,
    config: LeagueSamplingConfig,
) -> LeagueCheckpoint:
    windowed = _window_entries(entries, step, schedule)
    old_band, mid_band, recent_band = _split_bands(windowed, config)

    band_choices = [
        ("recent", config.recent_band_pct, recent_band),
        ("mid", config.mid_band_pct, mid_band),
        ("old", config.old_band_pct, old_band),
    ]
    pick = rng.random()
    cumulative = 0.0
    for _, weight, band in band_choices:
        cumulative += weight
        if pick <= cumulative:
            if band:
                return rng.choice(band)
            break
    # Fallbacks: prefer recent, then mid, then old
    if recent_band:
        return rng.choice(recent_band)
    if mid_band:
        return rng.choice(mid_band)
    if old_band:
        return rng.choice(old_band)
    return rng.choice(list(entries))


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path = path.resolve()
        parent = parent.resolve()
    except Exception:
        return False
    return str(path).startswith(str(parent))
