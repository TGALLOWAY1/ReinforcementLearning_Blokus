"""Reproducible arena harness for multi-agent Blokus experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from agents.fast_mcts_agent import FastMCTSAgent
from agents.gameplay_fast_mcts import GameplayFastMCTSAgent
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from analytics.tournament.arena_stats import compute_summary, render_summary_markdown
from analytics.winprob.features import (
    SNAPSHOT_FEATURE_COLUMNS,
    build_snapshot_runtime_context,
    coerce_feature_dict,
    extract_player_snapshot_features,
)
from engine.board import Player
from engine.game import BlokusGame
from engine.move_generator import LegalMoveGenerator, Move
from mcts.mcts_agent import MCTSAgent

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency path
    pd = None


DEFAULT_OUTPUT_ROOT = "arena_runs"
DEFAULT_MAX_TURNS = 2500
SUPPORTED_SEAT_POLICIES = {"randomized", "round_robin"}
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
    def from_dict(cls, item: Mapping[str, Any]) -> "AgentConfig":
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
    def from_dict(cls, item: Mapping[str, Any]) -> "SnapshotConfig":
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

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> "RunConfig":
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
        run_config = cls(
            agents=agents,
            num_games=num_games,
            seed=seed,
            seat_policy=seat_policy,
            output_root=output_root,
            max_turns=max_turns,
            notes=notes,
            snapshots=snapshots,
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
    else:
        seat_rng = random.Random(stable_hash_int(game_seed, "seat_assignment"))
        seat_rng.shuffle(ordered_agents)
    return {str(player.value): ordered_agents[idx] for idx, player in enumerate(Player)}


class _ArenaAgentAdapter:
    """Minimal adapter to normalize move selection + per-move telemetry."""

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
        if isinstance(self.agent, MCTSAgent):
            info = self.agent.get_action_info()
            mcts_stats = info.get("stats", {})
            if mcts_stats.get("iterations_run") is not None:
                move_stats["iterations_run"] = mcts_stats["iterations_run"]
            if mcts_stats.get("time_elapsed") is not None:
                move_stats["timeSpentMs"] = float(mcts_stats["time_elapsed"]) * 1000.0
        return move, move_stats


class _FastMCTSAdapter(_ArenaAgentAdapter):
    def __init__(
        self,
        agent: FastMCTSAgent,
        *,
        deterministic_time_budget: bool,
        iterations_per_ms: float,
    ):
        self.agent = agent
        self.deterministic_time_budget = deterministic_time_budget
        self.iterations_per_ms = iterations_per_ms

    def choose_move(
        self,
        board: Any,
        player: Player,
        legal_moves: List[Move],
        thinking_time_ms: Optional[int],
    ) -> Tuple[Optional[Move], Dict[str, Any]]:
        budget = int(thinking_time_ms or max(int(self.agent.time_limit * 1000), 1))
        if self.deterministic_time_budget:
            iteration_cap = max(1, int(round(self.iterations_per_ms * budget)))
            original_iterations = self.agent.iterations
            self.agent.iterations = iteration_cap
            try:
                result = self.agent.think(
                    board,
                    player,
                    legal_moves,
                    max(10_000_000, budget),
                )
            finally:
                self.agent.iterations = original_iterations
            stats = dict(result.get("stats") or {})
            stats["timeBudgetMs"] = budget
            stats["iterationCap"] = iteration_cap
            return result.get("move"), stats
        result = self.agent.think(board, player, legal_moves, budget)
        return result.get("move"), dict(result.get("stats") or {})


class _GameplayFastMCTSAdapter(_ArenaAgentAdapter):
    def __init__(
        self,
        agent: GameplayFastMCTSAgent,
        *,
        deterministic_time_budget: bool,
        iterations_per_ms: float,
    ):
        self.agent = agent
        self.deterministic_time_budget = deterministic_time_budget
        self.iterations_per_ms = iterations_per_ms

    def choose_move(
        self,
        board: Any,
        player: Player,
        legal_moves: List[Move],
        thinking_time_ms: Optional[int],
    ) -> Tuple[Optional[Move], Dict[str, Any]]:
        budget = int(thinking_time_ms or 1)
        if self.deterministic_time_budget:
            iteration_cap = max(1, int(round(self.iterations_per_ms * budget)))
            original_iterations = self.agent._agent.iterations
            self.agent._agent.iterations = iteration_cap
            try:
                move, stats = self.agent.choose_move(
                    board,
                    player,
                    legal_moves,
                    max(10_000_000, budget),
                )
            finally:
                self.agent._agent.iterations = original_iterations
            output_stats = dict(stats or {})
            output_stats["timeBudgetMs"] = budget
            output_stats["iterationCap"] = iteration_cap
            return move, output_stats
        move, stats = self.agent.choose_move(board, player, legal_moves, budget)
        return move, dict(stats or {})


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

    if agent_type == "mcts":
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
        )
        return _SelectActionAdapter(agent)

    if agent_type == "fast_mcts":
        deterministic_time_budget = bool(params.get("deterministic_time_budget", True))
        iterations_per_ms = float(params.get("iterations_per_ms", 20.0))
        default_time = (
            float(config.thinking_time_ms) / 1000.0
            if config.thinking_time_ms is not None
            else float(params.get("time_limit", 0.1))
        )
        agent = FastMCTSAgent(
            iterations=int(params.get("iterations", 5000)),
            time_limit=default_time,
            exploration_constant=float(params.get("exploration_constant", 1.414)),
            seed=seed,
        )
        return _FastMCTSAdapter(
            agent,
            deterministic_time_budget=deterministic_time_budget,
            iterations_per_ms=iterations_per_ms,
        )

    if agent_type in {"gameplay_fast_mcts", "gameplay_mcts"}:
        deterministic_time_budget = bool(params.get("deterministic_time_budget", True))
        iterations_per_ms = float(params.get("iterations_per_ms", 20.0))
        agent = GameplayFastMCTSAgent(
            iterations=int(params.get("iterations", 5000)),
            exploration_constant=float(params.get("exploration_constant", 1.414)),
            seed=seed,
        )
        return _GameplayFastMCTSAdapter(
            agent,
            deterministic_time_budget=deterministic_time_budget,
            iterations_per_ms=iterations_per_ms,
        )

    raise ValueError(f"Unsupported agent type: {config.type}")


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
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run one game and return game record + snapshot rows."""
    random.seed(game_seed)
    np.random.seed(game_seed)
    game_id = f"{run_id}_g{game_index:04d}"
    start = time.perf_counter()

    agent_instances: Dict[str, _ArenaAgentAdapter] = {}
    for agent_name in set(seat_assignment.values()):
        config = agent_configs[agent_name]
        seed = _agent_seed(run_config.seed, game_index, agent_name)
        agent_instances[agent_name] = build_agent(config, seed=seed)

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

    game = BlokusGame()
    move_generator = LegalMoveGenerator()
    passes = 0
    invalid_actions = 0
    turn_count = 0
    truncated = False
    error: Optional[str] = None

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

            if move is None:
                passes += 1
                game.board._update_current_player()
                game._check_game_over()
                continue

            if not game.make_move(move, current_player):
                invalid_actions += 1
                passes += 1
                game.board._update_current_player()
                game._check_game_over()
                continue

            maybe_capture_snapshot()
    except Exception as exc:
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
        "agent_move_stats": per_agent_stats,
        "snapshot_checkpoints_hit": sorted(checkpoint_hits),
        "error": error,
    }
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


def run_experiment(run_config: RunConfig, *, verbose: bool = False) -> Dict[str, Any]:
    """Run a full arena experiment and write all required artifacts."""
    run_id, run_dir = _prepare_run_directory(run_config)
    timestamp_iso = datetime.now().isoformat(timespec="seconds")
    run_config_payload = run_config.to_dict()
    run_config_payload["run_id"] = run_id
    run_config_payload["created_at"] = timestamp_iso

    _write_json(run_dir / "run_config.json", run_config_payload)

    games_path = run_dir / "games.jsonl"
    all_games: List[Dict[str, Any]] = []
    all_snapshot_rows: List[Dict[str, Any]] = []
    agent_configs = {agent.name: agent for agent in run_config.agents}
    with games_path.open("w", encoding="utf-8") as handle:
        for game_index in range(run_config.num_games):
            game_seed = game_seed_from_run_seed(run_config.seed, game_index)
            seat_assignment = _seat_assignment_for_game(
                run_config.agent_names,
                game_index,
                game_seed,
                run_config.seat_policy,
            )
            record, snapshot_rows = run_single_game(
                run_id=run_id,
                game_index=game_index,
                game_seed=game_seed,
                run_config=run_config,
                seat_assignment=seat_assignment,
                agent_configs=agent_configs,
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
