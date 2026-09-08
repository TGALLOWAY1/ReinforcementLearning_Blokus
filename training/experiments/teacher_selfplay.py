"""Phase 7: teacher self-play data pipeline (agent-strength rescue).

Plays standard-scored self-play games between four TEACHER agents — the
Phase-4-validated D-016 configuration (progressive widening + ridge-model
leaves) at the D-008 teacher budget (500 iterations) — and records a FULL
training example per decision (master plan §14):

  record_schema teacher_record_v1:
    game_id, decision_index, ply, player_id, seat_map,
    state             (Board.to_dict(), board_state_v1 — BEFORE the move)
    legal_actions     (move_v1 dicts, canonical order)
    search            ({action, visits, q} per expanded root child;
                       q = root player's mean reward at that child)
    policy_target     (visit distribution over `search` entries, sums to 1)
    selected_action   (move_v1)
    root_value        (root player's visit-weighted mean child q)
    final_scores / final_ranks (per player.value, backfilled at game end)
    search_config, value_model {path, sha256}, game_seed, agent_seed

Output is a manifested, immutable dataset directory: per-game JSONL shards
during generation (so --resume restarts at game granularity), packed into a
single records.jsonl.gz at finalization (shards deleted; decompressed bytes ==
the in-order shard concatenation, so content hashes are stable across the two
layouts). `--validate DIR`
re-checks every record against the engine (state round-trip, legal-move
regeneration, selected-action legality, policy-target alignment,
rank/score consistency, manifest counts) and must pass before any training
consumes the dataset.

    python -m training.experiments.teacher_selfplay \
        --games 18 --seed 20260715 --deadline-minutes 400 --out data/teacher_dataset_v1
    python -m training.experiments.teacher_selfplay --validate data/teacher_dataset_v1
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.board import STATE_SCHEMA_VERSION, Board, Player
from engine.game import SCORING_MODE_STANDARD, BlokusGame
from engine.move_generator import ACTION_SCHEMA_VERSION, Move, get_shared_generator
from mcts.mcts_agent import MCTSAgent

RECORD_SCHEMA_VERSION = "teacher_record_v2"  # v2: + move_selection field
TEACHER_ITERATIONS = 500  # D-008
VALUE_MODEL_PATH = "training/artifacts/value_models/v1/value_v1_ridge_baseline.joblib"

# Self-play diversity: model-leaf teacher agents are fully deterministic (no
# rollouts -> the seeded RNG is never consulted), so four identical teachers
# replay the SAME game regardless of seed (observed: 18/18 identical games).
# Standard remedy: for the first TEMPERATURE_DECISIONS decisions of each game,
# sample the played move from the root visit distribution (tau=1.0, seeded per
# game) instead of taking the argmax; later decisions use the argmax. The
# policy TARGET recorded for training is always the visit distribution.
TEMPERATURE_DECISIONS = 24  # ~first 6 decisions per player

TEACHER_SEARCH_CONFIG: Dict[str, Any] = {
    "iterations": TEACHER_ITERATIONS,
    "rollout_policy": "greedy_sample",
    "greedy_sample_size": 12,
    "rollout_cutoff_depth": 12,
    "heuristic_move_ordering": True,
    "exploration_constant": 1.414,
    "use_transposition_table": True,
    "progressive_widening_enabled": True,
    "pw_c": 2.0,
    "pw_alpha": 0.5,
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Packed-dataset format
#
# During generation each game is written as its own shard (game_NNNN.jsonl) so
# --resume can restart at game granularity. Finalization packs the shards, in
# order, into a single records.jsonl.gz and deletes them: one small file in
# git instead of one per game, with the CONTENT (decompressed bytes) hash
# identical to the shards-concat hash recorded in DATA_LINEAGE.md.
# ---------------------------------------------------------------------------

PACKED_NAME = "records.jsonl.gz"


def iter_dataset_records(dataset_dir: Path):
    """Yield (where, record) for every record, packed or shard layout."""
    packed = dataset_dir / PACKED_NAME
    if packed.exists():
        with gzip.open(packed, "rt", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh):
                if line.strip():
                    yield f"{PACKED_NAME}:{line_no}", json.loads(line)
        return
    for shard in sorted(dataset_dir.glob("game_*.jsonl")):
        for line_no, line in enumerate(shard.read_text().splitlines()):
            yield f"{shard.name}:{line_no}", json.loads(line)


def pack_dataset(out_dir: Path) -> Dict[str, str]:
    """Pack game shards into records.jsonl.gz; verify; delete the shards.

    Returns {"file", "sha256", "content_sha256"} for the manifest. The gzip
    header is written with mtime=0 so packing is byte-reproducible.
    """
    shards = sorted(out_dir.glob("game_*.jsonl"))
    if not shards:
        raise SystemExit(f"{out_dir}: no shards to pack")
    packed = out_dir / PACKED_NAME
    if packed.exists():
        raise SystemExit(f"{packed} already exists — refusing to repack")
    content_hash = hashlib.sha256()
    with open(packed, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            for shard in shards:
                data = shard.read_bytes()
                gz.write(data)
                content_hash.update(data)
    # Round-trip check before deleting anything.
    verify = hashlib.sha256()
    with gzip.open(packed, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            verify.update(chunk)
    if verify.hexdigest() != content_hash.hexdigest():
        packed.unlink()
        raise SystemExit("pack verification failed — shards left untouched")
    for shard in shards:
        shard.unlink()
    return {"file": PACKED_NAME, "sha256": _sha256(str(packed)),
            "content_sha256": content_hash.hexdigest()}


def _build_teacher(seed: int, search_config: Optional[Dict[str, Any]] = None,
                   value_model_path: Optional[str] = VALUE_MODEL_PATH) -> MCTSAgent:
    agent = MCTSAgent(seed=seed, value_model_path=value_model_path,
                      **(search_config or TEACHER_SEARCH_CONFIG))
    agent._capture_root_moves = True
    agent.num_workers = 1
    return agent


def play_teacher_game(game_seed: int, game_id: str,
                      value_model_sha: str,
                      search_config: Optional[Dict[str, Any]] = None,
                      value_model_path: Optional[str] = VALUE_MODEL_PATH,
                      ) -> List[Dict[str, Any]]:
    """Play one 4-teacher game; return the per-decision records (finals filled)."""
    import random as _random

    generator = get_shared_generator()
    game = BlokusGame(scoring_mode=SCORING_MODE_STANDARD, enable_telemetry=False)
    cfg = search_config or TEACHER_SEARCH_CONFIG
    agents = {p: _build_teacher(seed=game_seed * 31 + p.value,
                                search_config=cfg,
                                value_model_path=value_model_path)
              for p in Player}
    seat_map = {str(p.value): f"teacher_{p.value}" for p in Player}
    sample_rng = _random.Random(game_seed)

    records: List[Dict[str, Any]] = []
    decision_index = 0
    consecutive_passes = 0
    while consecutive_passes < len(Player):
        player = game.board.current_player
        legal_moves = generator.get_legal_moves(game.board, player)
        if not legal_moves:
            game.board._update_current_player()
            consecutive_passes += 1
            continue
        consecutive_passes = 0

        state_payload = game.board.to_dict()
        agent = agents[player]
        # select_action short-circuits on a single legal move WITHOUT running
        # a search or refreshing the capture attrs — clear them first so a
        # forced move can never inherit the previous decision's stale root
        # stats (review finding, PR #203).
        agent._last_root_move_stats = None
        move = agent.select_action(game.board, player, legal_moves)
        if move is None:
            game.board._update_current_player()
            continue

        if len(legal_moves) == 1:
            # Forced move: no search ran; the policy target is trivially 1.0.
            stats = [(legal_moves[0], 1, 0.0)]
        else:
            stats = agent._last_root_move_stats or []
        total_visits = sum(v for _, v, _ in stats) or 1
        search_entries = [
            {"action": m.to_dict(), "visits": int(v),
             "q": (r / v) if v else 0.0}
            for m, v, r in stats
        ]
        root_value = (
            sum(e["q"] * e["visits"] for e in search_entries) / total_visits
            if search_entries else 0.0
        )

        # Diversity: opening-phase visit sampling (see TEMPERATURE_DECISIONS).
        move_selection = "argmax"
        if decision_index < TEMPERATURE_DECISIONS and len(stats) > 1:
            move = sample_rng.choices(
                [m for m, _, _ in stats], weights=[v for _, v, _ in stats], k=1
            )[0]
            move_selection = "visit_sample_t1.0"
        records.append({
            "record_schema_version": RECORD_SCHEMA_VERSION,
            "state_schema_version": STATE_SCHEMA_VERSION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "game_id": game_id,
            "decision_index": decision_index,
            "ply": int(game.board.move_count),
            "player_id": int(player.value),
            "seat_map": seat_map,
            "state": state_payload,
            "legal_actions": [m.to_dict() for m in legal_moves],
            "search": search_entries,
            "policy_target": [e["visits"] / total_visits for e in search_entries],
            "selected_action": move.to_dict(),
            "move_selection": move_selection,
            "root_value": float(root_value),
            "search_config": cfg,
            "value_model": ({"path": value_model_path, "sha256": value_model_sha}
                            if value_model_path else None),
            "game_seed": int(game_seed),
            "agent_seed": int(game_seed * 31 + player.value),
            "final_scores": None,   # backfilled below
            "final_ranks": None,
        })
        decision_index += 1
        assert game.make_move(move, player), f"{game_id}: teacher move rejected"
        # Reset agent NST/history continuity is unnecessary between moves;
        # tables are per-game by design.

    result = game.get_game_result()
    scores = {str(pid): int(s) for pid, s in result.scores.items()}
    ordered = sorted(result.scores.items(), key=lambda kv: -kv[1])
    ranks: Dict[str, int] = {}
    rank = 0
    prev_score = None
    for idx, (pid, score) in enumerate(ordered, start=1):
        if score != prev_score:
            rank = idx
            prev_score = score
        ranks[str(pid)] = rank
    for record in records:
        record["final_scores"] = scores
        record["final_ranks"] = ranks
    return records


# ---------------------------------------------------------------------------
# Generation CLI
# ---------------------------------------------------------------------------

def generate(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / PACKED_NAME).exists():
        raise SystemExit(
            f"{out_dir} holds a finalized packed dataset — datasets are "
            "immutable. Use a new --out directory."
        )
    existing = sorted(out_dir.glob("game_*.jsonl"))
    if existing and not args.resume:
        raise SystemExit(
            f"{out_dir} already holds game shards — datasets are immutable. "
            "Use a new --out directory, or --resume to continue an interrupted "
            "generation (same seed scheme; completed shards untouched)."
        )
    search_config = dict(TEACHER_SEARCH_CONFIG, iterations=args.iterations)
    value_model_path = args.value_model if args.value_model else None
    value_model_sha = _sha256(value_model_path) if value_model_path else None
    value_model_entry = ({"path": value_model_path, "sha256": value_model_sha}
                         if value_model_path else None)
    if args.resume and existing:
        manifest_on_disk = json.loads((out_dir / "manifest.json").read_text())
        if manifest_on_disk.get("status") == "finalized":
            raise SystemExit(f"{out_dir} is finalized — refusing to resume.")
        if manifest_on_disk.get("seed") != args.seed:
            raise SystemExit(
                f"--resume seed mismatch: manifest has {manifest_on_disk.get('seed')}, "
                f"got {args.seed}."
            )
        # Existing shards were generated under the saved config; resuming with a
        # different one would silently mix configurations behind a manifest that
        # asserts a single provenance. Refuse mismatches.
        if manifest_on_disk.get("teacher_search_config") != search_config:
            raise SystemExit(
                "--resume search-config mismatch: manifest has "
                f"{manifest_on_disk.get('teacher_search_config')}, current args give "
                f"{search_config}. Re-run with the original --iterations."
            )
        if manifest_on_disk.get("value_model") != value_model_entry:
            raise SystemExit(
                "--resume value-model mismatch: manifest has "
                f"{manifest_on_disk.get('value_model')}, current args give "
                f"{value_model_entry}. Re-run with the original --value-model."
            )
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"

    manifest = {
        "dataset_schema_version": "teacher_dataset_v1",
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "purpose": "Phase 7 teacher self-play (gate C training corpus)",
        "generating_commit": commit,
        "scoring_mode": "standard",
        "teacher_search_config": search_config,
        "value_model": value_model_entry,
        "seed": args.seed,
        "num_games_requested": args.games,
        "status": "generating",
        "games_completed": 0,
        "records_written": 0,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    deadline = time.monotonic() + args.deadline_minutes * 60.0
    t0 = time.monotonic()
    # Count already-complete shards (resume) into the totals.
    total_records = sum(
        len(p.read_text().splitlines()) for p in out_dir.glob("game_*.jsonl")
    )
    completed = len(list(out_dir.glob("game_*.jsonl")))
    fresh_games = 0
    for g in range(args.games):
        if (out_dir / f"game_{g:04d}.jsonl").exists():
            continue  # completed before an interruption (resume)
        if fresh_games >= args.min_games and time.monotonic() >= deadline:
            print(f"deadline reached after {completed}/{args.games} games", flush=True)
            break
        game_seed = args.seed + g
        game_id = f"tds1_s{args.seed}_g{g:04d}"
        t_game = time.monotonic()
        records = play_teacher_game(
            game_seed, game_id, value_model_sha,
            search_config=search_config, value_model_path=value_model_path)
        shard = out_dir / f"game_{g:04d}.jsonl"
        with shard.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        completed += 1
        fresh_games += 1
        total_records += len(records)
        print(f"game {g + 1}/{args.games}: {len(records)} decisions in "
              f"{(time.monotonic() - t_game) / 60:.1f} min "
              f"(scores={records[0]['final_scores']})", flush=True)
        manifest.update(status="generating", games_completed=completed,
                        records_written=total_records)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    packed_info = pack_dataset(out_dir)
    manifest.update(status="finalized", games_completed=completed,
                    records_written=total_records,
                    packed=packed_info,
                    elapsed_sec=round(time.monotonic() - t0, 1))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{total_records} records / {completed} games "
          f"in {(time.monotonic() - t0) / 60:.1f} min -> {out_dir / PACKED_NAME}")
    return 0


# ---------------------------------------------------------------------------
# Validator (must pass before training consumes the dataset)
# ---------------------------------------------------------------------------

def validate(dataset_dir: Path) -> int:
    generator = get_shared_generator()
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    errors: List[str] = []
    records_seen = 0
    game_ids = set()

    def move_key(d: Dict[str, Any]):
        return (d["piece_id"], d["orientation"], d["anchor_row"], d["anchor_col"])

    for where, record in iter_dataset_records(dataset_dir):
            records_seen += 1
            game_ids.add(record["game_id"])
            # Records carry the engine schema versions they were generated under.
            # A mismatch (e.g. the pre-2026-09 piece catalogue, board_state_v1 /
            # move_v1) means piece ids and legal-move sets are not comparable:
            # report it as a validation error instead of letting from_dict raise.
            stamped_state = record.get("state_schema_version",
                                       record["state"].get("schema_version"))
            stamped_action = record.get("action_schema_version")
            if stamped_state != STATE_SCHEMA_VERSION or stamped_action != ACTION_SCHEMA_VERSION:
                errors.append(
                    f"{where}: schema mismatch — record {stamped_state!r}/{stamped_action!r} "
                    f"vs engine {STATE_SCHEMA_VERSION!r}/{ACTION_SCHEMA_VERSION!r} "
                    f"(dataset predates the current piece catalogue)"
                )
                if len(errors) > 20:
                    break
                continue
            board = Board.from_dict(record["state"])
            regenerated = {move_key(m.to_dict())
                           for m in generator.get_legal_moves(
                               board, Player(record["player_id"]))}
            recorded = [move_key(a) for a in record["legal_actions"]]
            if set(recorded) != regenerated:
                errors.append(f"{where}: legal-action set mismatch")
            if move_key(record["selected_action"]) not in regenerated:
                errors.append(f"{where}: selected action not legal")
            search_keys = [move_key(e["action"]) for e in record["search"]]
            if not set(search_keys) <= set(recorded):
                errors.append(f"{where}: search entry not in legal actions")
            pt = record["policy_target"]
            if len(pt) != len(record["search"]) or abs(sum(pt) - 1.0) > 1e-6:
                errors.append(f"{where}: policy target misaligned/unnormalized")
            scores = record["final_scores"]
            ranks = record["final_ranks"]
            if set(scores) != {"1", "2", "3", "4"} or set(ranks) != {"1", "2", "3", "4"}:
                errors.append(f"{where}: player-vector keys wrong")
            else:
                # Recompute standard-competition ranks from the scores and
                # require exact equality (equal scores <=> equal ranks;
                # review finding, PR #203).
                ordered = sorted(scores, key=lambda k: -scores[k])
                expected: Dict[str, int] = {}
                rank = 0
                prev = None
                for idx, key in enumerate(ordered, start=1):
                    if scores[key] != prev:
                        rank = idx
                        prev = scores[key]
                    expected[key] = rank
                if expected != {k: int(v) for k, v in ranks.items()}:
                    errors.append(f"{where}: ranks {ranks} != expected {expected}")
            if len(errors) > 20:
                break

    games_seen = len(game_ids)
    if manifest.get("games_completed") != games_seen:
        errors.append(f"manifest games_completed={manifest.get('games_completed')} "
                      f"!= games found {games_seen}")
    if manifest.get("records_written") != records_seen:
        errors.append(f"manifest records_written={manifest.get('records_written')} "
                      f"!= records found {records_seen}")

    if errors:
        print(f"VALIDATION FAILED ({len(errors)} errors):")
        for e in errors[:20]:
            print("  " + e)
        return 1
    print(f"VALIDATION PASSED: {records_seen} records / {games_seen} games, "
          f"manifest consistent.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m training.experiments.teacher_selfplay", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--games", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--deadline-minutes", type=float, default=400.0)
    parser.add_argument("--min-games", type=int, default=4)
    parser.add_argument("--out", default="data/teacher_dataset_v1")
    parser.add_argument("--iterations", type=int, default=TEACHER_ITERATIONS,
                        help="search budget per move (default: D-008 teacher 500; "
                             "bulk corpora use 50)")
    parser.add_argument("--value-model", default=VALUE_MODEL_PATH,
                        help="leaf value-model artifact; pass '' for rollout leaves")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted (non-finalized) generation: "
                             "skip existing shards, same seed scheme")
    parser.add_argument("--validate", default=None, metavar="DIR",
                        help="validate an existing dataset directory and exit")
    args = parser.parse_args(argv)
    if args.validate:
        return validate(Path(args.validate))
    return generate(args)


if __name__ == "__main__":
    sys.exit(main())
