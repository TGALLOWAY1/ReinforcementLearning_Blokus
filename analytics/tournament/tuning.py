"""Definitions for MCTS parameter tunings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class MctsTuning:
    """A specific parameter tuning for the MCTS agent."""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def resolve_params(self, thinking_time_ms: int) -> Dict[str, Any]:
        """Resolve dynamic parameters based on the time budget."""
        resolved = dict(self.params)

        # Handle adaptive bias
        if resolved.get("is_adaptive_bias"):
            del resolved["is_adaptive_bias"]
            # Apply thresholds based on empirical results
            if thinking_time_ms <= 75:
                resolved["progressive_bias_weight"] = 0.5
            elif thinking_time_ms <= 250:
                resolved["progressive_bias_weight"] = 0.25
            else:
                resolved["progressive_bias_weight"] = 0.0

            resolved["_resolved_budget"] = thinking_time_ms

        return resolved

    def to_dict(self, thinking_time_ms: int = None) -> Dict[str, Any]:
        p = self.resolve_params(thinking_time_ms) if thinking_time_ms is not None else dict(self.params)
        return {
            "name": self.name,
            "params": p,
        }


@dataclass(frozen=True)
class TuningSet:
    """A named collection of tunings intended to be evaluated together."""
    name: str
    tunings: List[MctsTuning]


_TUNING_SETS: Dict[str, TuningSet] = {}


def register_tuning_set(tuning_set: TuningSet) -> None:
    _TUNING_SETS[tuning_set.name] = tuning_set


def get_tuning_set(name: str) -> TuningSet:
    if name not in _TUNING_SETS:
        raise ValueError(f"Unknown tuning set '{name}'. Available: {sorted(_TUNING_SETS.keys())}")
    return _TUNING_SETS[name]


# -----------------------------------------------------------------------------
# Default Parameter Sets
# -----------------------------------------------------------------------------

_BASE_PARAMS = {
    "deterministic_time_budget": False,
    "use_transposition_table": True,
    "exploration_constant": 1.414,
}

_LEAF_EVAL_BASE = {
    **_BASE_PARAMS,
    "leaf_evaluation_enabled": True,
    "progressive_bias_enabled": True,
    "progressive_bias_weight": 0.25,
    "potential_shaping_enabled": False,
    "learned_model_path": "models/dummy_model.json"
}

# 1. Baseline vs Exploration (Hold typical constants fixed, vary c_uct)
register_tuning_set(TuningSet(
    name="baseline_vs_exploration",
    tunings=[
        MctsTuning("expl_1.0", {**_BASE_PARAMS, "exploration_constant": 1.0}),
        MctsTuning("expl_1.414", {**_BASE_PARAMS, "exploration_constant": 1.414}),
        MctsTuning("expl_2.0", {**_BASE_PARAMS, "exploration_constant": 2.0}),
        MctsTuning("expl_3.0", {**_BASE_PARAMS, "exploration_constant": 3.0}),
    ]
))

# 2. Ablation: Leaf Eval & Bias
register_tuning_set(TuningSet(
    name="ablation_leaf_eval",
    tunings=[
        MctsTuning("base_no_eval", {**_BASE_PARAMS}),
        MctsTuning("eval_no_bias", {
            **_LEAF_EVAL_BASE,
            "progressive_bias_enabled": False
        }),
        MctsTuning("eval_with_bias_0.1", {
            **_LEAF_EVAL_BASE,
            "progressive_bias_weight": 0.1
        }),
        MctsTuning("eval_with_bias_0.25", {
            **_LEAF_EVAL_BASE,
            "progressive_bias_weight": 0.25
        }),
        MctsTuning("eval_with_bias_0.5", {
            **_LEAF_EVAL_BASE,
            "progressive_bias_weight": 0.5
        }),
    ]
))

# 3. Full Tournament (Mixed strategies for ranking)
register_tuning_set(TuningSet(
    name="full_tournament",
    tunings=[
        MctsTuning("base_expl_1.414", {**_BASE_PARAMS, "exploration_constant": 1.414}),
        MctsTuning("base_expl_2.0", {**_BASE_PARAMS, "exploration_constant": 2.0}),
        MctsTuning("eval_bias_0.25_expl_1.414", {
            **_LEAF_EVAL_BASE,
            "progressive_bias_weight": 0.25,
            "exploration_constant": 1.414
        }),
        MctsTuning("eval_bias_0.25_expl_2.0", {
            **_LEAF_EVAL_BASE,
            "progressive_bias_weight": 0.25,
            "exploration_constant": 2.0
        }),
        MctsTuning("eval_bias_0.5_expl_1.414", {
            **_LEAF_EVAL_BASE,
            "progressive_bias_weight": 0.5,
            "exploration_constant": 1.414
        }),
        MctsTuning("eval_bias_all_shaping", {
            **_LEAF_EVAL_BASE,
            "progressive_bias_weight": 0.25,
            "potential_shaping_enabled": True,
            "potential_shaping_weight": 1.0
        }),
    ]
))

# 4. Adaptive Bias Benchmark (Benchmarks dynamic policy vs standard fixed winners)
register_tuning_set(TuningSet(
    name="adaptive_vs_best_fixed",
    tunings=[
        MctsTuning("adaptive_bias", {
            **_LEAF_EVAL_BASE,
            "is_adaptive_bias": True,
        }),
        MctsTuning("fixed_50ms_best", {
            **_LEAF_EVAL_BASE,
            "progressive_bias_weight": 0.5,
        }),
        MctsTuning("fixed_200ms_best", {
            **_LEAF_EVAL_BASE,
            "progressive_bias_weight": 0.0,
        }),
        MctsTuning("fixed_400ms_best", {
            **_LEAF_EVAL_BASE,
            "progressive_bias_weight": 0.25,
        }),
    ]
))
