"""Train win-probability model v2 (phase-aware nonlinear gradient boosting)."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.winprob.dataset import (
    build_pairwise_dataset,
    load_snapshots_dataframe,
    split_pairwise_by_game,
)
from analytics.winprob.features import SNAPSHOT_FEATURE_COLUMNS

PHASE_SEED_OFFSET = {"early": 101, "mid": 211, "late": 307}


def _format_float(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.6f}"


def _top_feature_importance(
    feature_columns: List[str],
    importances: np.ndarray,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    rows = []
    for feature, value in zip(feature_columns, importances):
        rows.append({"feature": feature, "importance": float(value)})
    rows.sort(key=lambda item: item["importance"], reverse=True)
    return rows[:top_n]


def _build_report_markdown(
    *,
    snapshots_path: str,
    dataset_meta: Dict[str, Any],
    split_meta: Dict[str, Any],
    phase_meta: Dict[str, Any],
    metrics: Dict[str, Any],
    phase_importances: Dict[str, List[Dict[str, Any]]],
    output_model: str,
) -> str:
    lines: List[str] = []
    lines.append("# Win Probability Model v2 Report")
    lines.append("")
    lines.append("## Dataset")
    lines.append(f"- Source: `{snapshots_path}`")
    lines.append(f"- Pairwise rows: `{dataset_meta['rows']}`")
    lines.append(f"- Games: `{dataset_meta['games']}`")
    lines.append(f"- Ties dropped: `{dataset_meta['ties_dropped']}`")
    lines.append("")
    lines.append("## Split (by game_id)")
    lines.append(f"- Train games: `{split_meta['train_games']}`")
    lines.append(f"- Test games: `{split_meta['test_games']}`")
    lines.append(f"- Train rows: `{split_meta['train_rows']}`")
    lines.append(f"- Test rows: `{split_meta['test_rows']}`")
    lines.append("")
    lines.append("## Phase-aware Training")
    lines.append(f"- Minimum rows per phase model: `{phase_meta['min_phase_rows']}`")
    lines.append(
        "- Phase model rows: "
        + ", ".join(
            f"{phase}={phase_meta['phase_rows'].get(phase, 0)}"
            for phase in ["early", "mid", "late"]
        )
    )
    lines.append(
        "- Trained phase models: "
        + ", ".join(
            phase for phase, enabled in phase_meta["phase_trained"].items() if enabled
        )
    )
    lines.append("")
    lines.append("## Metrics")
    lines.append(f"- Log loss: `{_format_float(metrics.get('log_loss'))}`")
    lines.append(f"- Brier score: `{_format_float(metrics.get('brier_score'))}`")
    lines.append(f"- Pairwise AUC: `{_format_float(metrics.get('pairwise_auc'))}`")
    lines.append(
        f"- Mean predicted p(y=1) for true `y=1`: `{_format_float(metrics.get('mean_pred_for_label_1'))}`"
    )
    lines.append(
        f"- Mean predicted p(y=1) for true `y=0`: `{_format_float(metrics.get('mean_pred_for_label_0'))}`"
    )
    lines.append(f"- Advantage ordering sanity: `{metrics.get('advantage_ordering_ok')}`")
    lines.append("")
    lines.append("## Feature Importance")
    for phase in ["fallback", "early", "mid", "late"]:
        rows = phase_importances.get(phase, [])
        if not rows:
            continue
        lines.append(f"### {phase.capitalize()}")
        lines.append("")
        lines.append("| Feature | Importance |")
        lines.append("| --- | ---: |")
        for row in rows:
            lines.append(f"| {row['feature']} | {row['importance']:.6f} |")
        lines.append("")
    lines.append("## Artifact")
    lines.append(f"- Model path: `{output_model}`")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train phase-aware pairwise gradient-boosted win-probability model."
    )
    parser.add_argument(
        "--snapshots",
        type=str,
        required=True,
        help="Path to snapshots dataset file or run directory.",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="models/winprob_gbt_v2.pkl",
        help="Output model artifact path.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="reports/winprob_gbt_v2.md",
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for split and model training.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of games allocated to test split.",
    )
    parser.add_argument(
        "--tie-policy",
        type=str,
        choices=["drop"],
        default="drop",
        help="Pairwise tie handling policy.",
    )
    parser.add_argument(
        "--min-phase-rows",
        type=int,
        default=200,
        help="Minimum training rows required to train a dedicated phase model.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of boosting stages.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for boosting.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum tree depth per boosting stage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshots_df = load_snapshots_dataframe(args.snapshots)
    pairwise_df, dataset_meta = build_pairwise_dataset(
        snapshots_df,
        feature_columns=SNAPSHOT_FEATURE_COLUMNS,
        tie_policy=args.tie_policy,
    )
    if pairwise_df.empty:
        raise ValueError("Pairwise dataset is empty after preprocessing.")

    train_df, test_df, split_meta = split_pairwise_by_game(
        pairwise_df,
        test_size=float(args.test_size),
        seed=int(args.seed),
    )
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split produced empty partition(s).")

    feature_columns = list(SNAPSHOT_FEATURE_COLUMNS)
    x_train = train_df[feature_columns].astype(float).to_numpy(dtype=float)
    y_train = train_df["label"].astype(int).to_numpy(dtype=int)
    x_test = test_df[feature_columns].astype(float).to_numpy(dtype=float)
    y_test = test_df["label"].astype(int).to_numpy(dtype=int)

    fallback_model = GradientBoostingClassifier(
        random_state=int(args.seed),
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        max_depth=int(args.max_depth),
    )
    fallback_model.fit(x_train, y_train)

    phase_models: Dict[str, GradientBoostingClassifier] = {}
    phase_rows: Dict[str, int] = {}
    phase_trained: Dict[str, bool] = {}
    for phase in ["early", "mid", "late"]:
        phase_train = train_df[train_df["phase_bucket"] == phase]
        phase_rows[phase] = int(len(phase_train))
        if (
            len(phase_train) < int(args.min_phase_rows)
            or phase_train["label"].nunique() < 2
        ):
            phase_trained[phase] = False
            continue
        model = GradientBoostingClassifier(
            random_state=int(args.seed) + PHASE_SEED_OFFSET[phase],
            n_estimators=int(args.n_estimators),
            learning_rate=float(args.learning_rate),
            max_depth=int(args.max_depth),
        )
        model.fit(
            phase_train[feature_columns].astype(float).to_numpy(dtype=float),
            phase_train["label"].astype(int).to_numpy(dtype=int),
        )
        phase_models[phase] = model
        phase_trained[phase] = True

    test_prob = np.zeros(len(test_df), dtype=float)
    test_phases = test_df["phase_bucket"].astype(str).tolist()
    for idx, phase in enumerate(test_phases):
        model = phase_models.get(phase, fallback_model)
        test_prob[idx] = model.predict_proba(x_test[idx : idx + 1])[:, 1][0]
    if np.any(test_prob < 0.0) or np.any(test_prob > 1.0):
        raise ValueError("Predictions out of [0, 1] range.")

    metrics: Dict[str, Any] = {
        "log_loss": float(log_loss(y_test, test_prob)),
        "brier_score": float(brier_score_loss(y_test, test_prob)),
        "pairwise_auc": None,
        "mean_pred_for_label_1": None,
        "mean_pred_for_label_0": None,
        "advantage_ordering_ok": None,
    }
    if len(np.unique(y_test)) > 1:
        metrics["pairwise_auc"] = float(roc_auc_score(y_test, test_prob))
    if np.any(y_test == 1):
        metrics["mean_pred_for_label_1"] = float(np.mean(test_prob[y_test == 1]))
    if np.any(y_test == 0):
        metrics["mean_pred_for_label_0"] = float(np.mean(test_prob[y_test == 0]))
    if (
        metrics["mean_pred_for_label_1"] is not None
        and metrics["mean_pred_for_label_0"] is not None
    ):
        metrics["advantage_ordering_ok"] = bool(
            metrics["mean_pred_for_label_1"] > metrics["mean_pred_for_label_0"]
        )

    phase_importances: Dict[str, List[Dict[str, Any]]] = {
        "fallback": _top_feature_importance(
            feature_columns, fallback_model.feature_importances_
        )
    }
    for phase, model in phase_models.items():
        phase_importances[phase] = _top_feature_importance(
            feature_columns, model.feature_importances_
        )

    phase_meta = {
        "min_phase_rows": int(args.min_phase_rows),
        "phase_rows": phase_rows,
        "phase_trained": phase_trained,
    }

    artifact = {
        "version": "winprob_gbt_v2",
        "model_type": "pairwise_gbt_phase",
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        "feature_columns": feature_columns,
        "tie_policy": args.tie_policy,
        "fallback_model": fallback_model,
        "phase_models": phase_models,
        "metrics": metrics,
        "dataset_meta": dataset_meta,
        "split_meta": split_meta,
        "phase_meta": phase_meta,
    }

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_model)

    report_md = _build_report_markdown(
        snapshots_path=args.snapshots,
        dataset_meta=dataset_meta,
        split_meta=split_meta,
        phase_meta=phase_meta,
        metrics=metrics,
        phase_importances=phase_importances,
        output_model=str(output_model),
    )
    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(report_md, encoding="utf-8")

    print(f"model_path: {output_model}")
    print(f"report_path: {output_report}")
    print(f"log_loss: {metrics['log_loss']:.6f}")
    print(f"brier_score: {metrics['brier_score']:.6f}")
    if metrics["pairwise_auc"] is not None:
        print(f"pairwise_auc: {metrics['pairwise_auc']:.6f}")
    print(f"advantage_ordering_ok: {metrics['advantage_ordering_ok']}")
    print(f"pred_min: {float(test_prob.min()):.6f}")
    print(f"pred_max: {float(test_prob.max()):.6f}")


if __name__ == "__main__":
    main()
