"""Learned win-probability evaluator for MCTS integration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import joblib
import numpy as np

from analytics.winprob.features import (
    SNAPSHOT_FEATURE_COLUMNS,
    build_snapshot_runtime_context,
    coerce_feature_dict,
    extract_player_snapshot_features,
)
from engine.board import Board, Player
from engine.move_generator import LegalMoveGenerator


def _phase_bucket_from_occupancy(
    board_occupancy: float,
    *,
    early_max: float = 0.33,
    mid_max: float = 0.66,
) -> str:
    if board_occupancy < early_max:
        return "early"
    if board_occupancy < mid_max:
        return "mid"
    return "late"


class LearnedWinProbabilityEvaluator:
    """Model-backed evaluator that estimates per-player win probability."""

    def __init__(
        self,
        artifact_path: str,
        *,
        max_turns: int = 2500,
        potential_mode: str = "prob",
    ) -> None:
        self.artifact_path = str(artifact_path)
        self.max_turns = int(max_turns)
        self.potential_mode = potential_mode
        self.move_generator = LegalMoveGenerator()
        
        self._is_dummy = self.artifact_path.endswith("dummy_model.json")
        if self._is_dummy:
            self.model_type = "dummy"
            self.feature_columns = []
            return
            
        self.artifact: Dict[str, Any] = joblib.load(Path(self.artifact_path))
        self.model_type = str(self.artifact.get("model_type", ""))
        self.feature_columns = list(
            self.artifact.get("feature_columns", SNAPSHOT_FEATURE_COLUMNS)
        )
        if self.model_type not in {"pairwise_logreg", "pairwise_gbt_phase"}:
            raise ValueError(
                f"Unsupported learned evaluator model_type '{self.model_type}' in {self.artifact_path}."
            )

    def _build_cache_key(self, board: Board) -> Tuple[Any, ...]:
        pieces = tuple(
            tuple(sorted(int(piece_id) for piece_id in board.player_pieces_used[player]))
            for player in Player
        )
        return (
            board.grid.tobytes(),
            int(board.move_count),
            pieces,
        )

    def _extract_features_for_all_players(self, board: Board) -> Dict[int, Dict[str, float]]:
        context = build_snapshot_runtime_context(
            board,
            turn_index=int(board.move_count),
            max_turns=self.max_turns,
        )
        by_player: Dict[int, Dict[str, float]] = {}
        for player in Player:
            features = extract_player_snapshot_features(
                board,
                player=player,
                context=context,
                move_generator=self.move_generator,
            )
            by_player[int(player.value)] = coerce_feature_dict(features)
        return by_player

    def _predict_pairwise(
        self,
        feature_i: Mapping[str, float],
        feature_j: Mapping[str, float],
    ) -> float:
        if getattr(self, "_is_dummy", False):
            return 0.5
            
        x = np.array(
            [[float(feature_i[col]) - float(feature_j[col]) for col in self.feature_columns]],
            dtype=float,
        )
        if self.model_type == "pairwise_logreg":
            pipeline = self.artifact["pipeline"]
            return float(pipeline.predict_proba(x)[:, 1][0])

        if self.model_type == "pairwise_gbt_phase":
            occupancy = float(
                (float(feature_i.get("phase_board_occupancy", 0.0)) + float(feature_j.get("phase_board_occupancy", 0.0)))
                / 2.0
            )
            phase = _phase_bucket_from_occupancy(occupancy)
            phase_models = self.artifact.get("phase_models", {})
            fallback_model = self.artifact.get("fallback_model")
            model = phase_models.get(phase) or fallback_model
            if model is None:
                raise RuntimeError(
                    "GBT evaluator artifact missing both phase model and fallback model."
                )
            return float(model.predict_proba(x)[:, 1][0])

        raise RuntimeError(f"Unsupported model type: {self.model_type}")

    def predict_player_win_probability(self, board: Board, player: Player) -> float:
        """Estimate a player's win chance by aggregating pairwise probabilities."""
        if getattr(self, "_is_dummy", False):
            return 0.5
            
        features_by_player = self._extract_features_for_all_players(board)
        player_id = int(player.value)
        feature_i = features_by_player[player_id]
        pairwise_probs = []
        for opponent in Player:
            if opponent == player:
                continue
            feature_j = features_by_player[int(opponent.value)]
            pairwise_probs.append(self._predict_pairwise(feature_i, feature_j))
        if not pairwise_probs:
            return 0.5
        return float(np.mean(pairwise_probs))

    def potential(self, board: Board, player: Player) -> float:
        """Potential function used for reward shaping."""
        probability = self.predict_player_win_probability(board, player)
        probability = min(max(probability, 1e-6), 1 - 1e-6)
        if self.potential_mode == "logit":
            return float(math.log(probability / (1.0 - probability)))
        return float(probability)
