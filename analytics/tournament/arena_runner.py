"""Reproducible arena harness for multi-agent Blokus experiments."""

from __future__ import annotations

# Audit version tag: all results produced after the mcts audit remediation
# include this version so pre-fix and post-fix results can be distinguished.
AUDIT_VERSION = "v1_2026-03-28"

import csv
import hashlib
import json
import random
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from analytics.logging.logger import StrategyLogger
from analytics.tournament.arena_stats import compute_summary, render_summary_markdown
from analytics.winprob.features import (
    SNAPSHOT_FEATURE_COLUMNS,
    build_snapshot_runtime_context,
    coerce_feature_dict,
    extract_player_snapshot_features,
)
from engine.board import STATE_SCHEMA_VERSION, Player
from engine.game import SCORING_MODE_STANDARD, VALID_SCORING_MODES, BlokusGame
from engine.move_generator import ACTION_SCHEMA_VERSION, LegalMoveGenerator, Move
from engine.pieces import PieceGenerator
from mcts.champion_profile import CHALLENGE_CHAMPION_PROFILE, build_mcts_kwargs, load_challenge_champion_profile
from mcts.mcts_agent import MCTSAgent
from mcts.state_evaluator import BlokusStateEvaluator
from schemas.game_state import AgentType
from webapi.gameplay_agent_factory import build_deploy_gameplay_agent

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency path
    pd = None


DEFAULT_OUTPUT_ROOT = "arena_runs"

# Piece size (number of squares) by piece_id, cached — used by the per-game
# play-quality diagnostics (piece usage by size). Lookup is a dict hit after the
# first miss, so the per-move overhead is negligible.
_PIECE_SIZE_CACHE: Dict[int, int] = {}


def _piece_size(piece_id: int) -> int:
    """Number of squares in the piece with this id (0 if unknown), cached."""
    size = _PIECE_SIZE_CACHE.get(piece_id)
    if size is None:
        piece = PieceGenerator.get_piece_by_id(piece_id)
        size = piece.size if piece else 0
        _PIECE_SIZE_CACHE[piece_id] = size
    return size

# Shared evaluator instance for snapshot se_ feature extraction (default weights; raw features only)
_SE_EVALUATOR = BlokusStateEvaluator()
DEFAULT_MAX_TURNS = 2500
SUPPORTED_SEAT_POLICIES = {"randomized", "round_robin", "all_permutations"}
DEFAULT_SNAPSHOT_PLYS = [8, 16, 24, 32, 40, 48, 56, 64]


def _default_snapshot_plys() -> List[int]:
    return list(DEFAULT_SNAPSHOT_PLYS)


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for a single arena agent."""

    name: str
    type: str
    thinking_time_ms: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, item: Mapping[str, Any]) -> AgentConfig:
        if "name" not in item:
            raise ValueError("Agent entries must include 'name'")
        if "type" not in item:
            raise ValueError(f"Agent '{item['name']}' is missing required field 'type'")
        params = dict(item.get("params") or {})
        for key, value in item.items():
            if key in {"name", "type", "thinking_time_ms", "params"}:
                continue
            params[key] = value
        thinking_time = item.get("thinking_time_ms")
        thinking_time_ms = int(thinking_time) if thinking_time is not None else None
        return cls(
            name=str(item["name"]),
            type=str(item["type"]),
            thinking_time_ms=thinking_time_ms,
            params=params,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "thinking_time_ms": self.thinking_time_ms,
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class SnapshotConfig:
    """Snapshot logging configuration for ML dataset generation."""

    enabled: bool = False
    strategy: str = "fixed_ply"
    checkpoints: List[int] = field(default_factory=_default_snapshot_plys)

    @classmethod
    def from_dict(cls, item: Mapping[str, Any]) -> SnapshotConfig:
        strategy = str(item.get("strategy", "fixed_ply"))
        checkpoints_raw = item.get("checkpoints", _default_snapshot_plys())
        if not isinstance(checkpoints_raw, list):
            raise ValueError("snapshots.checkpoints must be a list of integers")
        checkpoints = sorted({int(value) for value in checkpoints_raw if int(value) >= 0})
        return cls(
            enabled=bool(item.get("enabled", False)),
            strategy=strategy,
            checkpoints=checkpoints,
        )

    def validate(self) -> None:
        if self.strategy != "fixed_ply":
            raise ValueError(
                f"Unsupported snapshot strategy '{self.strategy}'. Supported values: ['fixed_ply']."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "strategy": self.strategy,
            "checkpoints": list(self.checkpoints),
        }


@dataclass(frozen=True)
class RunConfig:
    """Top-level experiment configuration."""

    agents: List[AgentConfig]
    num_games: int
    seed: int
    seat_policy: str = "randomized"
    output_root: str = DEFAULT_OUTPUT_ROOT
    max_turns: int = DEFAULT_MAX_TURNS
    notes: str = ""
    snapshots: SnapshotConfig = field(default_factory=SnapshotConfig)
    # Scoring objective for arena games. Standard is the evaluation target
    # (decision D-002); "house" is explicit opt-in for historical comparisons.
    scoring_mode: str = SCORING_MODE_STANDARD

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> RunConfig:
        agents_raw = config.get("agents")
        if agents_raw is None:
            agents_raw = _legacy_agents_to_list(config)
        if not isinstance(agents_raw, list):
            raise ValueError("RunConfig 'agents' must be a list.")

        agents = [AgentConfig.from_dict(item) for item in agents_raw]
        num_games = int(config.get("num_games", 100))
        seed = int(config.get("seed", 0))
        seat_policy = str(config.get("seat_policy", "randomized"))
        output_root = str(config.get("output_root", DEFAULT_OUTPUT_ROOT))
        max_turns = int(config.get("max_turns", DEFAULT_MAX_TURNS))
        notes = str(config.get("notes", ""))
        snapshots_raw = config.get("snapshots") or {}
        if not isinstance(snapshots_raw, Mapping):
            raise ValueError("RunConfig 'snapshots' must be an object when provided.")
        snapshots = SnapshotConfig.from_dict(snapshots_raw)
        scoring_mode = str(config.get("scoring_mode", SCORING_MODE_STANDARD))
        run_config = cls(
            agents=agents,
            num_games=num_games,
            seed=seed,
            seat_policy=seat_policy,
            output_root=output_root,
            max_turns=max_turns,
            notes=notes,
            snapshots=snapshots,
            scoring_mode=scoring_mode,
        )
        run_config.validate()
        return run_config

    def validate(self) -> None:
        if self.num_games <= 0:
            raise ValueError("num_games must be > 0.")
        if len(self.agents) != len(Player):
            raise ValueError(
                f"Arena expects exactly {len(Player)} agents; received {len(self.agents)}."
            )
        if len({agent.name for agent in self.agents}) != len(self.agents):
            raise ValueError("Agent names must be unique.")
        if self.seat_policy not in SUPPORTED_SEAT_POLICIES:
            raise ValueError(
                f"Unsupported seat_policy '{self.seat_policy}'. Expected one of {sorted(SUPPORTED_SEAT_POLICIES)}."
            )
        if self.max_turns <= 0:
            raise ValueError("max_turns must be > 0.")
        if self.scoring_mode not in VALID_SCORING_MODES:
            raise ValueError(
                f"Unsupported scoring_mode '{self.scoring_mode}'. "
                f"Expected one of {sorted(VALID_SCORING_MODES)}."
            )
        self.snapshots.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agents": [agent.to_dict() for agent in self.agents],
            "num_games": self.num_games,
            "seed": self.seed,
            "seat_policy": self.seat_policy,
            "output_root": self.output_root,
            "max_turns": self.max_turns,
            "notes": self.notes,
            "snapshots": self.snapshots.to_dict(),
            "scoring_mode": self.scoring_mode,
        }

    @property
    def agent_names(self) -> List[str]:
        return [agent.name for agent in self.agents]


def _legacy_agents_to_list(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Support legacy scripts/arena_config.json shape."""
    legacy_agents: List[Dict[str, Any]] = []
    for name, item in config.items():
        if not isinstance(item, Mapping) or "type" not in item:
            continue
        thinking_time_ms = item.get("thinking_time_ms")
        if thinking_time_ms is None and item.get("time_limit") is not None:
            thinking_time_ms = int(float(item["time_limit"]) * 1000)
        params: Dict[str, Any] = {}
        for key, value in item.items():
            if key in {"type", "thinking_time_ms"}:
                continue
            params[key] = value
        legacy_agents.append(
            {
                "name": str(name),
                "type": str(item["type"]),
                "thinking_time_ms": thinking_time_ms,
                "params": params,
            }
        )
    if not legacy_agents:
        raise ValueError(
            "Invalid run config: missing 'agents' list and no legacy agent entries were found."
        )
    return legacy_agents


def load_run_config(config_path: str) -> RunConfig:
    """Load a RunConfig from JSON."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("Run config must be a JSON object.")
    return RunConfig.from_dict(payload)


def stable_hash_int(*parts: Any, mod: int = 2**31 - 1) -> int:
    """Stable integer hash suitable for seeds."""
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) % mod


def game_seed_from_run_seed(run_seed: int, game_index: int) -> int:
    """Derive deterministic per-game seed from run seed and game index."""
    return stable_hash_int(run_seed, game_index, "game_seed")


def _agent_seed(run_seed: int, game_index: int, agent_name: str) -> int:
    return stable_hash_int(run_seed, game_index, agent_name, "agent_seed")


def _resolve_run_id(config: RunConfig) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hash_input = json.dumps(config.to_dict(), sort_keys=True)
    short_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:8]
    return f"{timestamp}_{short_hash}"


def _prepare_run_directory(config: RunConfig) -> Tuple[str, Path]:
    root = Path(config.output_root)
    root.mkdir(parents=True, exist_ok=True)
    base_run_id = _resolve_run_id(config)
    run_id = base_run_id
    run_dir = root / run_id
    attempt = 1
    while run_dir.exists():
        run_id = f"{base_run_id}_{attempt:02d}"
        run_dir = root / run_id
        attempt += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def _seat_assignment_for_game(
    agent_names: Sequence[str],
    game_index: int,
    game_seed: int,
    seat_policy: str,
) -> Dict[str, str]:
    ordered_agents = list(agent_names)
    if seat_policy == "round_robin":
        shift = game_index % len(ordered_agents)
        ordered_agents = ordered_agents[shift:] + ordered_agents[:shift]
    elif seat_policy == "all_permutations":
        # Cycle through every seating (4! = 24): balances seats AND who-moves-
        # after-whom, which a cyclic shift does not (protocol v3, EXP-014a).
        import itertools

        perms = list(itertools.permutations(ordered_agents))
        ordered_agents = list(perms[game_index % len(perms)])
    else:
        seat_rng = random.Random(stable_hash_int(game_seed, "seat_assignment"))
        seat_rng.shuffle(ordered_agents)
    return {str(player.value): ordered_agents[idx] for idx, player in enumerate(Player)}


class _ArenaAgentAdapter:
    """Minimal adapter to normalize move selection + per-move telemetry."""

    def close(self) -> None:
        """Release external resources (subprocesses); default delegates to agent.close()."""
        agent = getattr(self, "agent", None)
        closer = getattr(agent, "close", None)
        if callable(closer):
            closer()

    def choose_move(
        self,
        board: Any,
        player: Player,
        legal_moves: List[Move],
        thinking_time_ms: Optional[int],
    ) -> Tuple[Optional[Move], Dict[str, Any]]:
        raise NotImplementedError


class _SelectActionAdapter(_ArenaAgentAdapter):
    def __init__(self, agent: Any):
        self.agent = agent

    def choose_move(
        self,
        board: Any,
        player: Player,
        legal_moves: List[Move],
        thinking_time_ms: Optional[int],
    ) -> Tuple[Optional[Move], Dict[str, Any]]:
        start = time.perf_counter()
        move = self.agent.select_action(board, player, legal_moves)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        move_stats: Dict[str, Any] = {"timeSpentMs": elapsed_ms}
        if not isinstance(self.agent, MCTSAgent) and hasattr(self.agent, "get_action_info"):
            # Non-MCTS agents (e.g. external engines) report their own counters.
            try:
                agent_stats = (self.agent.get_action_info() or {}).get("stats")
            except Exception:
                agent_stats = None
            if isinstance(agent_stats, dict):
                move_stats["_agent_stats"] = dict(agent_stats)
        if isinstance(self.agent, MCTSAgent):
            info = self.agent.get_action_info()
            mcts_stats = info.get("stats", {})
            if mcts_stats.get("iterations_run") is not None:
                move_stats["iterations_run"] = mcts_stats["iterations_run"]
            if mcts_stats.get("time_elapsed") is not None:
                move_stats["timeSpentMs"] = float(mcts_stats["time_elapsed"]) * 1000.0
            # Pass through full MCTS stats for game logger
            move_stats["_mcts_stats"] = mcts_stats
        return move, move_stats

    def notify_move(self, board_before: Any, board_after: Any, player: Player) -> None:
        """Forward move notification to underlying agent for opponent modeling."""
        if hasattr(self.agent, 'notify_move'):
            self.agent.notify_move(board_before, board_after, player)

    @property
    def has_opponent_modeling(self) -> bool:
        """Check if the underlying agent has opponent modeling enabled."""
        return getattr(self.agent, 'opponent_modeling_enabled', False)


@dataclass
class _ChooseMoveAdapter(_ArenaAgentAdapter):
    """Adapter for gameplay agents exposing choose_move(board, player, legal_moves, budget_ms)."""

    agent: Any
    default_budget_ms: int = 1000

    def play_turn(
        self,
        board: Any,
        player: Player,
        legal_moves: List[Move],
        thinking_time_ms: Optional[int],
    ) -> Tuple[Optional[Move], Dict[str, Any]]:
        budget_ms = int(thinking_time_ms) if thinking_time_ms is not None else int(self.default_budget_ms)
        move, move_stats = self.agent.choose_move(board, player, legal_moves, budget_ms)
        stats: Dict[str, Any] = dict(move_stats or {})
        if "timeBudgetMs" not in stats:
            stats["timeBudgetMs"] = budget_ms
        return move, stats

    def notify_move(self, board_before: Any, board_after: Any, player: Player) -> None:
        """Forward move notification to underlying agent for opponent modeling."""
        if hasattr(self.agent, 'notify_move'):
            self.agent.notify_move(board_before, board_after, player)

    @property
    def has_opponent_modeling(self) -> bool:
        """Check if the underlying agent has opponent modeling enabled."""
        underlying = getattr(self.agent, "_agent", None)
        return bool(getattr(underlying, "opponent_modeling_enabled", False))


def _load_policy_weights_file(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load a policy-weights artifact JSON (or None). Raises on a bad path —
    a config naming a missing artifact must fail loudly, not silently fall
    back to the default heuristic policy."""
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    # policy_learning artifacts wrap the weights in a "policy" block; MLP
    # artifacts are flat.
    if isinstance(data, dict) and "policy" in data and "model_type" not in data:
        return data["policy"]
    return data


def _apply_policy_temperature(
    weights: Optional[Dict[str, Any]], temperature: Optional[float]
) -> Optional[Dict[str, Any]]:
    """Override a policy artifact's softmax temperature (prior-calibration sweeps).

    Higher temperature flattens the PUCT prior toward uniform without retraining;
    ``None`` leaves the artifact untouched. Returns a shallow copy so the on-disk
    artifact is never mutated."""
    if weights is None or temperature is None:
        return weights
    patched = dict(weights)
    patched["temperature"] = float(temperature)
    return patched


def build_agent(config: AgentConfig, seed: int) -> _ArenaAgentAdapter:
    """Instantiate an agent adapter from configuration."""
    agent_type = config.type.lower()
    params = dict(config.params)

    if agent_type == "random":
        return _SelectActionAdapter(RandomAgent(seed=seed))

    if agent_type == "heuristic":
        agent = HeuristicAgent(seed=seed)
        weights = params.get("weights")
        if isinstance(weights, Mapping):
            agent.set_weights(dict(weights))
        return _SelectActionAdapter(agent)

    if agent_type == "greedy":
        # Deterministic argmax of the fixed heuristic (protocol v3 yardstick).
        from agents.greedy_agent import GreedyAgent

        agent = GreedyAgent(seed=seed)
        weights = params.get("weights")
        if isinstance(weights, Mapping):
            agent.set_weights(dict(weights))
        return _SelectActionAdapter(agent)

    if agent_type == "pentobi":
        # External Pentobi GTP engine (anchor opponent / rules cross-check).
        from agents.pentobi_agent import PentobiAgent

        return _SelectActionAdapter(PentobiAgent(
            level=int(params.get("level", 5)), seed=seed,
            binary=params.get("binary"), use_book=bool(params.get("use_book", False)),
        ))

    if agent_type == "challenge_champion_gameplay":
        profile_name = str(params.get("profile", CHALLENGE_CHAMPION_PROFILE))
        if profile_name != CHALLENGE_CHAMPION_PROFILE:
            raise ValueError(
                f"Unsupported gameplay profile '{profile_name}'. "
                f"Only '{CHALLENGE_CHAMPION_PROFILE}' is currently supported."
            )
        gameplay_cfg = dict(params)
        gameplay_cfg["profile"] = CHALLENGE_CHAMPION_PROFILE
        gameplay_cfg.setdefault("seed", seed)
        gameplay_agent = build_deploy_gameplay_agent(AgentType.MCTS, gameplay_cfg)
        if gameplay_agent is None:
            raise ValueError("Failed to construct challenge champion gameplay adapter.")
        budget_ms = int(config.thinking_time_ms) if config.thinking_time_ms is not None else int(
            gameplay_cfg.get("time_budget_ms", 1000)
        )
        return _ChooseMoveAdapter(gameplay_agent, default_budget_ms=budget_ms)

    if agent_type == "mcts":
        if params.get("profile") == CHALLENGE_CHAMPION_PROFILE:
            profile_kwargs = build_mcts_kwargs(load_challenge_champion_profile())
            profile_kwargs.update({k: v for k, v in params.items() if k != "profile"})
            params = profile_kwargs
        deterministic_time_budget = bool(params.get("deterministic_time_budget", True))
        iterations_per_ms = float(params.get("iterations_per_ms", 10.0))
        iterations = int(params.get("iterations", 1000))
        time_limit = params.get("time_limit")
        if deterministic_time_budget and config.thinking_time_ms is not None:
            iterations = max(1, int(round(iterations_per_ms * config.thinking_time_ms)))
            time_limit = None
        elif time_limit is None and config.thinking_time_ms is not None:
            time_limit = float(config.thinking_time_ms) / 1000.0
        agent = MCTSAgent(
            iterations=iterations,
            time_limit=float(time_limit) if time_limit is not None else None,
            exploration_constant=float(params.get("exploration_constant", 1.414)),
            use_transposition_table=bool(params.get("use_transposition_table", True)),
            seed=seed,
            learned_model_path=params.get("learned_model_path"),
            leaf_evaluation_enabled=bool(params.get("leaf_evaluation_enabled", False)),
            progressive_bias_enabled=bool(params.get("progressive_bias_enabled", False)),
            progressive_bias_weight=float(params.get("progressive_bias_weight", 0.25)),
            potential_shaping_enabled=bool(params.get("potential_shaping_enabled", False)),
            potential_shaping_gamma=float(params.get("potential_shaping_gamma", 1.0)),
            potential_shaping_weight=float(params.get("potential_shaping_weight", 1.0)),
            potential_mode=str(params.get("potential_mode", "prob")),
            max_rollout_moves=int(params.get("max_rollout_moves", 50)),
            # Layer 3: Action Reduction
            progressive_widening_enabled=bool(params.get("progressive_widening_enabled", False)),
            pw_c=float(params.get("pw_c", 2.0)),
            pw_alpha=float(params.get("pw_alpha", 0.5)),
            progressive_history_enabled=bool(params.get("progressive_history_enabled", False)),
            progressive_history_weight=float(params.get("progressive_history_weight", 1.0)),
            heuristic_move_ordering=bool(params.get("heuristic_move_ordering", False)),
            # Learned move policy (PUCT prior + rollout/ordering). Inline
            # ``policy_weights`` wins; otherwise ``policy_weights_path`` loads
            # an artifact JSON from disk (mirrors value_model_path — MLP
            # artifacts are ~1 MB and don't belong inline in agent configs).
            policy_prior_enabled=bool(params.get("policy_prior_enabled", False)),
            policy_prior_c=float(params.get("policy_prior_c", 1.5)),
            policy_weights=_apply_policy_temperature(
                params.get("policy_weights")
                if params.get("policy_weights") is not None
                else _load_policy_weights_file(params.get("policy_weights_path")),
                params.get("policy_temperature"),
            ),
            # Layer 4: Simulation Strategy
            rollout_policy=str(params.get("rollout_policy", "heuristic")),
            two_ply_top_k=int(params["two_ply_top_k"]) if params.get("two_ply_top_k") is not None else None,
            rollout_cutoff_depth=int(params["rollout_cutoff_depth"]) if params.get("rollout_cutoff_depth") is not None else None,
            state_eval_weights=params.get("state_eval_weights"),
            state_eval_phase_weights=params.get("state_eval_phase_weights"),
            minimax_backup_alpha=float(params.get("minimax_backup_alpha", 0.0)),
            # Layer 5: History Heuristics & RAVE
            rave_enabled=bool(params.get("rave_enabled", False)),
            rave_k=float(params.get("rave_k", 1000.0)),
            nst_enabled=bool(params.get("nst_enabled", False)),
            nst_weight=float(params.get("nst_weight", 0.5)),
            # Layer 7: Opponent Modeling
            opponent_rollout_policy=str(params.get("opponent_rollout_policy", "same")),
            opponent_modeling_enabled=bool(params.get("opponent_modeling_enabled", False)),
            alliance_detection_enabled=bool(params.get("alliance_detection_enabled", False)),
            alliance_threshold=float(params.get("alliance_threshold", 2.0)),
            kingmaker_detection_enabled=bool(params.get("kingmaker_detection_enabled", False)),
            kingmaker_score_gap=int(params.get("kingmaker_score_gap", 15)),
            adaptive_opponent_enabled=bool(params.get("adaptive_opponent_enabled", False)),
            defensive_weight_shift=float(params.get("defensive_weight_shift", 0.15)),
            # Layer 8: Parallelization
            num_workers=int(params.get("num_workers", 1)),
            virtual_loss=float(params.get("virtual_loss", 1.0)),
            parallel_strategy=str(params.get("parallel_strategy", "root")),
            # Layer 9: Meta-Optimization
            adaptive_exploration_enabled=bool(params.get("adaptive_exploration_enabled", False)),
            adaptive_exploration_base=float(params.get("adaptive_exploration_base", 1.414)),
            adaptive_exploration_avg_bf=float(params.get("adaptive_exploration_avg_bf", 80.0)),
            adaptive_rollout_depth_enabled=bool(params.get("adaptive_rollout_depth_enabled", False)),
            adaptive_rollout_depth_base=int(params.get("adaptive_rollout_depth_base", 20)),
            adaptive_rollout_depth_avg_bf=float(params.get("adaptive_rollout_depth_avg_bf", 80.0)),
            sufficiency_threshold_enabled=bool(params.get("sufficiency_threshold_enabled", False)),
            loss_avoidance_enabled=bool(params.get("loss_avoidance_enabled", False)),
            loss_avoidance_threshold=float(params.get("loss_avoidance_threshold", -50.0)),
            # Rich leaf evaluator (45-feature TD value at MCTS leaves)
            rich_leaf_eval_enabled=bool(params.get("rich_leaf_eval_enabled", False)),
            rich_leaf_weights_path=params.get("rich_leaf_weights_path"),
            # None -> serve the subset the artifact was trained with
            rich_leaf_feature_subset=params.get("rich_leaf_feature_subset"),
            # Model-artifact leaf evaluator (D-015 gate 3); takes precedence
            # over rich_leaf weights when set
            value_model_path=params.get("value_model_path"),
        )
        return _SelectActionAdapter(agent)

    if agent_type in {"fast_mcts", "gameplay_fast_mcts", "gameplay_mcts"}:
        raise ValueError(
            f"Agent type '{config.type}' (FastMCTS) is not a valid MCTS implementation "
            "and has been removed from arena use. FastMCTS does not build a true game "
            "tree and produces misleading results. Use type 'mcts' instead. "
            "See docs/audits/mcts_audit_remediation_plan.md for details."
        )

    raise ValueError(f"Unsupported agent type: {config.type}")


def _any_agent_has_opponent_modeling(
    agent_instances: Mapping[str, _ArenaAgentAdapter],
) -> bool:
    """Check if any agent in the game has opponent modeling enabled."""
    for inst in agent_instances.values():
        if getattr(inst, 'has_opponent_modeling', False):
            return True
    return False


def _extract_move_telemetry(
    raw_stats: Mapping[str, Any],
    fallback_elapsed_ms: float,
) -> Tuple[float, Optional[float]]:
    time_spent_ms = raw_stats.get("timeSpentMs")
    if time_spent_ms is None and raw_stats.get("time_elapsed") is not None:
        time_spent_ms = float(raw_stats["time_elapsed"]) * 1000.0
    if time_spent_ms is None:
        time_spent_ms = fallback_elapsed_ms
    simulations = None
    for key in ("nodesEvaluated", "iterations_run", "simulations", "rollouts"):
        value = raw_stats.get(key)
        if value is not None:
            try:
                simulations = float(value)
                break
            except (TypeError, ValueError):
                continue
    return float(time_spent_ms), simulations


def _maybe_save_heatmap_data(
    agent: _ArenaAgentAdapter,
    turn: int,
    player: "Player",
    board: Any,
    move: Move,
    output_dir: Path,
) -> None:
    """Save heatmap visit-count data for a turn."""
    underlying = getattr(agent, "agent", None)
    
    # Try to grab MCTS children data if available, else empty
    children_data = getattr(underlying, "_last_root_children_data", [])

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store used pieces for offline analysis reconstruction
    used_pieces = {
        str(p.value): list(board.player_pieces_used[p]) 
        for p in board.player_pieces_used
    }

    turn_data = {
        "turn": turn,
        "player": int(player.value),
        "board_grid": board.grid.tolist(),
        "used_pieces": used_pieces,
        "moves": children_data,
        "chosen_move": {"x": move.anchor_col, "y": move.anchor_row},
    }
    path = output_dir / f"turn_{turn:03d}.json"
    with open(path, "w") as f:
        json.dump(turn_data, f)


def _compute_ranks(scores_by_player: Mapping[str, int]) -> Dict[str, int]:
    unique_scores = sorted(set(scores_by_player.values()), reverse=True)
    score_to_rank = {score: rank + 1 for rank, score in enumerate(unique_scores)}
    return {player_id: score_to_rank[score] for player_id, score in scores_by_player.items()}


def _build_snapshot_rows_for_checkpoint(
    *,
    run_id: str,
    game_id: str,
    game_index: int,
    game_seed: int,
    seat_assignment: Mapping[str, str],
    checkpoint_index: int,
    checkpoint_ply: int,
    game: BlokusGame,
    turn_count: int,
    max_turns: int,
    move_generator: LegalMoveGenerator,
) -> List[Dict[str, Any]]:
    context = build_snapshot_runtime_context(
        game.board,
        turn_index=turn_count,
        max_turns=max_turns,
    )
    current_player_id = int(game.get_current_player().value)
    rows: List[Dict[str, Any]] = []
    for player in Player:
        player_id = int(player.value)
        features = extract_player_snapshot_features(
            game.board,
            player=player,
            context=context,
            move_generator=move_generator,
        )
        row: Dict[str, Any] = {
            "run_id": run_id,
            "game_id": game_id,
            "game_index": int(game_index),
            "game_seed": int(game_seed),
            "checkpoint_index": int(checkpoint_index),
            "checkpoint_ply": int(checkpoint_ply),
            "ply": int(context.ply),
            "turn_index": int(turn_count),
            "current_player_id": current_player_id,
            "player_id": player_id,
            "player_name": player.name,
            "agent_name": seat_assignment[str(player_id)],
            "seat_index": player_id - 1,
            "winner_id": None,
            "winner_ids_json": None,
            "label_is_winner": None,
            "final_score": None,
            "final_rank": None,
            "final_scores_json": None,
            "is_tie": None,
        }
        row.update(coerce_feature_dict(features))
        # Also extract state-evaluator features (se_ prefix) for evaluator weight re-fitting
        try:
            se_features = _SE_EVALUATOR.extract_features(game.board, player)
            for fname, fval in se_features.items():
                row[f"se_{fname}"] = float(fval)
        except Exception:
            pass
        rows.append(row)
    return rows


def run_single_game(
    *,
    run_id: str,
    game_index: int,
    game_seed: int,
    run_config: RunConfig,
    seat_assignment: Mapping[str, str],
    agent_configs: Mapping[str, AgentConfig],
    game_logger: Optional[StrategyLogger] = None,
    verbose: bool = False,
    heatmap_output_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run one game and return game record + snapshot rows."""
    random.seed(game_seed)
    np.random.seed(game_seed)
    game_id = f"{run_id}_g{game_index:04d}"
    start = time.perf_counter()

    agent_instances: Dict[str, _ArenaAgentAdapter] = {}
    try:
        for agent_name in set(seat_assignment.values()):
            config = agent_configs[agent_name]
            seed = _agent_seed(run_config.seed, game_index, agent_name)
            agent_instances[agent_name] = build_agent(config, seed=seed)
    except Exception:
        # Do not leak external engine processes if a later agent fails to build.
        for adapter in agent_instances.values():
            adapter.close()
        raise

    per_agent_stats: Dict[str, Dict[str, Any]] = {
        agent_name: {
            "moves": 0.0,
            "total_time_ms": 0.0,
            "total_simulations": 0.0,
            "moves_with_simulations": 0.0,
            "move_times_ms": [],
        }
        for agent_name in set(seat_assignment.values())
    }

    game = BlokusGame(scoring_mode=run_config.scoring_mode)
    move_generator = LegalMoveGenerator()
    passes = 0
    invalid_actions = 0
    turn_count = 0
    truncated = False
    error: Optional[str] = None

    # --- Play-quality accumulators (per-game diagnostics) --------------------
    # Cheap to collect (the values are already on hand each turn): the legal-move
    # count is computed for selection, the piece size is a cached dict lookup, and
    # board occupancy is one array scan at the end.
    legal_move_counts: List[int] = []  # len(legal_moves) at each turn (0 on a pass)
    piece_size_usage: Dict[int, int] = {}                 # size -> count (all seats)
    piece_size_usage_by_agent: Dict[str, Dict[int, int]] = {}  # agent -> {size: count}

    snapshot_rows: List[Dict[str, Any]] = []
    checkpoint_hits: Set[int] = set()
    checkpoint_to_index = {
        checkpoint_ply: idx
        for idx, checkpoint_ply in enumerate(sorted(run_config.snapshots.checkpoints))
    }

    def maybe_capture_snapshot() -> None:
        if not run_config.snapshots.enabled:
            return
        current_ply = int(game.board.move_count)
        if current_ply not in checkpoint_to_index:
            return
        if current_ply in checkpoint_hits:
            return
        snapshot_rows.extend(
            _build_snapshot_rows_for_checkpoint(
                run_id=run_id,
                game_id=game_id,
                game_index=game_index,
                game_seed=game_seed,
                seat_assignment=seat_assignment,
                checkpoint_index=checkpoint_to_index[current_ply],
                checkpoint_ply=current_ply,
                game=game,
                turn_count=turn_count,
                max_turns=run_config.max_turns,
                move_generator=move_generator,
            )
        )
        checkpoint_hits.add(current_ply)

    maybe_capture_snapshot()  # Supports checkpoint at ply=0.

    try:
        while not game.is_game_over() and turn_count < run_config.max_turns:
            current_player = game.get_current_player()
            player_id = str(current_player.value)
            agent_name = seat_assignment[player_id]
            agent_cfg = agent_configs[agent_name]
            agent = agent_instances[agent_name]
            legal_moves = game.get_legal_moves(current_player)

            turn_count += 1
            legal_move_counts.append(len(legal_moves))
            if not legal_moves:
                passes += 1
                game.board._update_current_player()
                game._check_game_over()
                continue

            move_start = time.perf_counter()
            move, raw_stats = agent.choose_move(
                game.board,
                current_player,
                legal_moves,
                agent_cfg.thinking_time_ms,
            )
            move_elapsed_ms = (time.perf_counter() - move_start) * 1000.0
            elapsed_ms, simulations = _extract_move_telemetry(raw_stats, move_elapsed_ms)
            stats_entry = per_agent_stats[agent_name]
            stats_entry["moves"] += 1
            stats_entry["total_time_ms"] += elapsed_ms
            stats_entry["move_times_ms"].append(elapsed_ms)
            if simulations is not None:
                stats_entry["total_simulations"] += simulations
                stats_entry["moves_with_simulations"] += 1

            # Per-move progress reporting for long-running games
            if verbose and (turn_count <= 3 or turn_count % 4 == 0):
                game_elapsed = time.perf_counter() - start
                print(
                    f"  [game {game_index + 1}/{run_config.num_games}] "
                    f"turn {turn_count}, {game_elapsed:.1f}s elapsed, "
                    f"agent={agent_name}, move_ms={elapsed_ms:.0f}",
                    flush=True,
                )

            # Capture heatmap data from MCTS agents
            if heatmap_output_dir is not None and move is not None:
                _maybe_save_heatmap_data(
                    agent, turn_count, current_player, game.board, move, heatmap_output_dir,
                )

            if move is None:
                passes += 1
                game.board._update_current_player()
                game._check_game_over()
                continue

            # Capture board state before move for logging and opponent modeling
            need_board_before = game_logger or _any_agent_has_opponent_modeling(agent_instances)
            board_before = game.board.copy() if need_board_before else None

            if not game.make_move(move, current_player):
                invalid_actions += 1
                passes += 1
                game.board._update_current_player()
                game._check_game_over()
                continue

            # Play-quality: record the size of the piece just successfully placed.
            size = _piece_size(getattr(move, "piece_id", 0))
            piece_size_usage[size] = piece_size_usage.get(size, 0) + 1
            agent_usage = piece_size_usage_by_agent.setdefault(agent_name, {})
            agent_usage[size] = agent_usage.get(size, 0) + 1

            # Layer 7: Notify all agents of the move for opponent tracking
            if board_before is not None:
                for a_name, a_inst in agent_instances.items():
                    if hasattr(a_inst, 'notify_move'):
                        a_inst.notify_move(board_before, game.board, current_player)

            # Log move with MCTS diagnostics if logger attached
            if game_logger and board_before is not None:
                mcts_stats = raw_stats.get("_mcts_stats") if isinstance(raw_stats, dict) else None
                game_logger.on_step(
                    game_id=game_id,
                    turn_index=turn_count,
                    player_id=int(current_player.value),
                    state=board_before,
                    move=move,
                    next_state=game.board,
                    mcts_stats=mcts_stats,
                    decision_time_ms=elapsed_ms,
                )

            maybe_capture_snapshot()
    except Exception:
        error = traceback.format_exc()

    if turn_count >= run_config.max_turns and not game.is_game_over():
        truncated = True
        game.board.game_over = True

    game_result = game.get_game_result()
    scores_by_player = {
        str(player_id): int(score) for player_id, score in game_result.scores.items()
    }
    final_ranks = _compute_ranks(scores_by_player)
    winner_ids = [int(player_id) for player_id in game_result.winner_ids]
    winner_agents = [seat_assignment[str(player_id)] for player_id in winner_ids]
    winner_id = winner_ids[0] if len(winner_ids) == 1 else None
    agent_scores = {
        seat_assignment[player_id]: score for player_id, score in scores_by_player.items()
    }
    agent_ranks = {
        seat_assignment[player_id]: rank
        for player_id, rank in final_ranks.items()
    }

    for row in snapshot_rows:
        player_id = int(row["player_id"])
        player_key = str(player_id)
        row["winner_id"] = winner_id
        row["winner_ids_json"] = json.dumps(winner_ids, sort_keys=True)
        row["label_is_winner"] = int(player_id in winner_ids)
        row["final_score"] = int(scores_by_player[player_key])
        row["final_rank"] = int(final_ranks[player_key])
        row["final_scores_json"] = json.dumps(scores_by_player, sort_keys=True)
        row["is_tie"] = bool(game_result.is_tie)

    duration_sec = time.perf_counter() - start
    for stats_entry in per_agent_stats.values():
        moves = stats_entry["moves"]
        stats_entry["avg_time_ms"] = (
            stats_entry["total_time_ms"] / moves if moves > 0 else 0.0
        )
        if stats_entry["moves_with_simulations"] > 0:
            stats_entry["avg_simulations_per_move"] = (
                stats_entry["total_simulations"] / stats_entry["moves_with_simulations"]
            )
            total_time_s = stats_entry["total_time_ms"] / 1000.0
            stats_entry["simulations_per_second"] = (
                stats_entry["total_simulations"] / total_time_s if total_time_s > 0 else None
            )
        else:
            stats_entry["avg_simulations_per_move"] = None
            stats_entry["simulations_per_second"] = None
            stats_entry["total_simulations"] = None

    # --- Play-quality summary (per-game) ------------------------------------
    grid = game.board.grid
    total_cells = int(grid.size)
    occupied_cells = int(np.count_nonzero(grid))
    score_values = [int(v) for v in scores_by_player.values()]
    play_quality = {
        "avg_legal_moves_per_turn": (
            sum(legal_move_counts) / len(legal_move_counts) if legal_move_counts else 0.0
        ),
        "min_legal_moves": min(legal_move_counts) if legal_move_counts else 0,
        "max_legal_moves": max(legal_move_counts) if legal_move_counts else 0,
        "pass_rate": (passes / turn_count) if turn_count else 0.0,
        "invalid_move_count": int(invalid_actions),
        "game_length_turns": int(turn_count),
        "board_occupancy": (occupied_cells / total_cells) if total_cells else 0.0,
        # Piece usage by size: keys are the number of squares (1..5), values are
        # how many pieces of that size were placed across all seats / per agent.
        "piece_size_usage": {str(k): v for k, v in sorted(piece_size_usage.items())},
        "piece_size_usage_by_agent": {
            agent: {str(k): v for k, v in sorted(usage.items())}
            for agent, usage in piece_size_usage_by_agent.items()
        },
        # Final-score distribution across seats (spread = competitiveness signal).
        "final_score_min": min(score_values) if score_values else 0,
        "final_score_max": max(score_values) if score_values else 0,
        "final_score_spread": (max(score_values) - min(score_values)) if score_values else 0,
    }

    record = {
        "run_id": run_id,
        "game_id": game_id,
        "game_index": game_index,
        "game_seed": game_seed,
        "seat_assignment": dict(seat_assignment),
        "seat_policy": run_config.seat_policy,
        "winner_ids": winner_ids,
        "winner_agents": winner_agents,
        "is_tie": bool(game_result.is_tie),
        "final_scores": scores_by_player,
        "final_ranks": final_ranks,
        "agent_scores": agent_scores,
        "agent_ranks": agent_ranks,
        "winner_id": winner_id,
        "moves_made": int(game.board.move_count),
        "turn_count": int(turn_count),
        "passes": int(passes),
        "invalid_actions": int(invalid_actions),
        "duration_sec": float(duration_sec),
        "truncated": bool(truncated),
        "play_quality": play_quality,
        "agent_move_stats": per_agent_stats,
        "snapshot_checkpoints_hit": sorted(checkpoint_hits),
        "error": error,
        "is_valid_result": True,
        "invalid_reason": "",
        "audit_version": AUDIT_VERSION,
        # Provenance stamps (agent-strength rescue Phase 2): which state/action
        # encodings and scoring objective produced this game.
        "state_schema_version": STATE_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "scoring_mode": run_config.scoring_mode,
    }
    # Release external resources (e.g. pentobi-gtp subprocesses) deterministically.
    for adapter in agent_instances.values():
        try:
            adapter.close()
        except Exception:  # pragma: no cover - best effort teardown
            pass
    return record, snapshot_rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _append_index_row(
    *,
    output_root: Path,
    run_id: str,
    timestamp: str,
    run_config: RunConfig,
    completed_games: int,
) -> None:
    index_path = output_root / "index.csv"
    row = {
        "run_id": run_id,
        "timestamp": timestamp,
        "num_games": run_config.num_games,
        "completed_games": completed_games,
        "agents": "|".join(
            f"{agent.name}:{agent.type}:{agent.thinking_time_ms}" for agent in run_config.agents
        ),
        "seed": run_config.seed,
        "seat_policy": run_config.seat_policy,
        "notes": run_config.notes,
        "audit_version": AUDIT_VERSION,
    }
    write_header = not index_path.exists()
    with index_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_snapshots_dataset(
    run_dir: Path,
    snapshot_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not snapshot_rows:
        return {
            "enabled": False,
            "rows": 0,
            "path_parquet": None,
            "path_csv": None,
            "parquet_written": False,
            "parquet_error": None,
        }
    if pd is None:
        raise RuntimeError("pandas is required for snapshot dataset writing.")

    dataframe = pd.DataFrame(snapshot_rows)
    csv_path = run_dir / "snapshots.csv"
    dataframe.to_csv(csv_path, index=False)

    parquet_path = run_dir / "snapshots.parquet"
    parquet_written = False
    parquet_error: Optional[str] = None
    try:
        dataframe.to_parquet(parquet_path, index=False)
        parquet_written = True
    except Exception as exc:  # pragma: no cover - depends on parquet engine availability
        parquet_error = str(exc)
        (run_dir / "snapshots_parquet_error.txt").write_text(
            "Failed to write snapshots.parquet.\n"
            f"Reason: {parquet_error}\n"
            "Install pyarrow or fastparquet to enable parquet output.\n",
            encoding="utf-8",
        )

    return {
        "enabled": True,
        "rows": int(len(dataframe)),
        "path_parquet": str(parquet_path) if parquet_written else None,
        "path_csv": str(csv_path),
        "parquet_written": parquet_written,
        "parquet_error": parquet_error,
    }


def _build_snapshot_diagnostics(snapshot_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not snapshot_rows or pd is None:
        return {}

    dataframe = pd.DataFrame(snapshot_rows)
    distributions: Dict[str, Dict[str, float]] = {}
    for column in SNAPSHOT_FEATURE_COLUMNS:
        if column not in dataframe.columns:
            continue
        series = dataframe[column].astype(float)
        distributions[column] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "p25": float(series.quantile(0.25)),
            "p75": float(series.quantile(0.75)),
            "min": float(series.min()),
            "max": float(series.max()),
        }

    corr_pairs: List[Dict[str, Any]] = []
    numeric_df = dataframe[SNAPSHOT_FEATURE_COLUMNS].astype(float)
    corr = numeric_df.corr().fillna(0.0)
    columns = list(corr.columns)
    for i, col_a in enumerate(columns):
        for j in range(i + 1, len(columns)):
            col_b = columns[j]
            value = float(corr.iloc[i, j])
            if abs(value) >= 0.95:
                corr_pairs.append(
                    {"feature_a": col_a, "feature_b": col_b, "correlation": value}
                )
    corr_pairs.sort(key=lambda item: abs(item["correlation"]), reverse=True)

    winner_lead: Dict[str, Dict[str, float]] = {}
    lead_metric = "utility_frontier_plus_mobility"
    if lead_metric in dataframe.columns and "label_is_winner" in dataframe.columns:
        for checkpoint, sub_df in dataframe.groupby("checkpoint_index"):
            winners = sub_df[sub_df["label_is_winner"] == 1]
            losers = sub_df[sub_df["label_is_winner"] == 0]
            if winners.empty or losers.empty:
                continue
            winner_lead[str(int(checkpoint))] = {
                "winner_mean": float(winners[lead_metric].mean()),
                "non_winner_mean": float(losers[lead_metric].mean()),
                "gap": float(winners[lead_metric].mean() - losers[lead_metric].mean()),
            }

    return {
        "feature_distributions": distributions,
        "high_correlation_pairs": corr_pairs[:100],
        "winner_lead_by_checkpoint": winner_lead,
    }


def run_experiment(
    run_config: RunConfig,
    *,
    verbose: bool = False,
    enable_game_logging: bool = False,
    resume_run_dir: Optional[str | Path] = None,
    deadline: Optional[float] = None,
) -> Dict[str, Any]:
    """Run a full arena experiment and write all required artifacts.

    If resume_run_dir is provided, appends games to an existing run until
    num_games is reached. Uses the config from the existing run.

    ``deadline`` is an optional ``time.monotonic()`` cutoff checked *before each
    game*. When reached the loop stops early and the summary is computed over the
    games that actually completed. This makes a single arena interruptible at game
    granularity so it cannot overrun a wall-clock budget (e.g. a 100-game battery
    can no longer blow a 45-minute evaluation budget by 4x).
    """
    if resume_run_dir is not None:
        run_dir = Path(resume_run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {run_dir}")
        config_path = run_dir / "run_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Cannot resume: {config_path} not found. "
                "Resume requires an existing run with run_config.json."
            )
        run_config_payload = json.loads(config_path.read_text())
        run_config = RunConfig.from_dict(run_config_payload)
        run_id = run_config_payload.get("run_id", run_dir.name)

        # Load existing games (skip malformed lines from concurrent writes)
        games_path = run_dir / "games.jsonl"
        all_games = []
        if games_path.exists():
            with games_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        if isinstance(record, dict):
                            all_games.append(record)
                    except json.JSONDecodeError:
                        continue

        start_index = len(all_games)
        if start_index >= run_config.num_games:
            return {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "summary": json.loads((run_dir / "summary.json").read_text())
                if (run_dir / "summary.json").exists()
                else {},
                "snapshot_rows": 0,
            }

        if verbose:
            print(
                f"      Resuming: {start_index} games exist, "
                f"running {run_config.num_games - start_index} more"
            )
    else:
        run_id, run_dir = _prepare_run_directory(run_config)
        start_index = 0
        all_games = []
        games_path = run_dir / "games.jsonl"
        timestamp_iso = datetime.now().isoformat(timespec="seconds")
        run_config_payload = run_config.to_dict()
        run_config_payload["run_id"] = run_id
        run_config_payload["created_at"] = timestamp_iso
        _write_json(run_dir / "run_config.json", run_config_payload)

    timestamp_iso = datetime.now().isoformat(timespec="seconds")

    # Create game logger if requested
    game_logger: Optional[StrategyLogger] = None
    if enable_game_logging:
        log_dir = str(run_dir / "game_logs")
        game_logger = StrategyLogger(log_dir=log_dir)

    all_snapshot_rows: List[Dict[str, Any]] = []
    agent_configs = {agent.name: agent for agent in run_config.agents}
    mode = "a" if resume_run_dir else "w"
    with games_path.open(mode, encoding="utf-8") as handle:
        for game_index in range(start_index, run_config.num_games):
            if deadline is not None and time.monotonic() >= deadline:
                if verbose:
                    print(
                        f"      Deadline reached after {game_index - start_index} "
                        f"game(s); stopping arena early.",
                        flush=True,
                    )
                break
            game_seed = game_seed_from_run_seed(run_config.seed, game_index)
            seat_assignment = _seat_assignment_for_game(
                run_config.agent_names,
                game_index,
                game_seed,
                run_config.seat_policy,
            )

            # Initialize logger for this game
            if game_logger:
                game_id = f"{run_id}_g{game_index:04d}"
                agent_ids = {
                    int(pid): seat_assignment[pid]
                    for pid in seat_assignment
                }
                game_logger.on_reset(game_id, game_seed, agent_ids, run_config_payload)

            if verbose:
                agents_str = ", ".join(
                    sorted(set(seat_assignment.values()))
                )
                print(
                    f"[{game_index + 1}/{run_config.num_games}] "
                    f"Starting game (seed={game_seed}, agents={agents_str}) ...",
                    flush=True,
                )

            record, snapshot_rows = run_single_game(
                run_id=run_id,
                game_index=game_index,
                game_seed=game_seed,
                run_config=run_config,
                seat_assignment=seat_assignment,
                agent_configs=agent_configs,
                game_logger=game_logger,
                verbose=verbose,
            )

            # Log game end
            if game_logger:
                scores = {
                    int(k): int(v) for k, v in record["final_scores"].items()
                }
                game_logger.on_game_end(
                    game_id=record["game_id"],
                    final_scores=scores,
                    winner_id=record.get("winner_id"),
                    num_turns=record["turn_count"],
                )
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            all_games.append(record)
            all_snapshot_rows.extend(snapshot_rows)
            if verbose:
                winners = ",".join(record["winner_agents"])
                print(
                    f"[{game_index + 1}/{run_config.num_games}] "
                    f"seed={game_seed} winners={winners} scores={record['agent_scores']}"
                )

    summary = compute_summary(
        all_games,
        run_id=run_id,
        run_seed=run_config.seed,
        seat_policy=run_config.seat_policy,
        agent_names=run_config.agent_names,
        thinking_time_ms_by_agent={
            agent.name: agent.thinking_time_ms for agent in run_config.agents
        },
        run_config=run_config_payload,
    )

    snapshot_meta = _write_snapshots_dataset(run_dir, all_snapshot_rows)
    if snapshot_meta["enabled"]:
        summary["snapshots"] = snapshot_meta
        summary["snapshot_diagnostics"] = _build_snapshot_diagnostics(all_snapshot_rows)
        expected_rows = (
            run_config.num_games
            * len(run_config.snapshots.checkpoints)
            * len(Player)
        )
        summary["snapshots"]["expected_rows_max"] = int(expected_rows)

    summary["audit_version"] = AUDIT_VERSION
    _write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(
        render_summary_markdown(summary),
        encoding="utf-8",
    )
    _append_index_row(
        output_root=Path(run_config.output_root),
        run_id=run_id,
        timestamp=timestamp_iso,
        run_config=run_config,
        completed_games=int(summary["completed_games"]),
    )

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "summary": summary,
        "snapshot_rows": len(all_snapshot_rows),
    }
