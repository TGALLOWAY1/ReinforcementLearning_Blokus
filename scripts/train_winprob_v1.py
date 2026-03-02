"""Train win-probability model v1 (pairwise logistic regression)."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.winprob.dataset import (
    build_pairwise_dataset,
    load_snapshots_dataframe,
    split_pairwise_by_game,
)
from analytics.winprob.features import SNAPSHOT_FEATURE_COLUMNS


def _format_float(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.6f}"


def _build_report_markdown(
    *,
    snapshots_path: str,
    dataset_meta: Dict[str, Any],
    split_meta: Dict[str, Any],
    metrics: Dict[str, Any],
    calibration_rows: List[Dict[str, float]],
    top_coefficients: List[Dict[str, Any]],
    output_model: str,
) -> str:
    lines: List[str] = []
    lines.append("# Win Probability Model v1 Report")
    lines.append("")
    lines.append("## Dataset")
    lines.append(f"- Source: `{snapshots_path}`")
    lines.append(f"- Pairwise rows: `{dataset_meta['rows']}`")
    lines.append(f"- Games: `{dataset_meta['games']}`")
    lines.append(f"- Checkpoints: `{dataset_meta['checkpoints']}`")
    lines.append(f"- Ties dropped (`{dataset_meta['tie_policy']}`): `{dataset_meta['ties_dropped']}`")
    lines.append("")
    lines.append("## Split (by game_id)")
    lines.append(f"- Train games: `{split_meta['train_games']}`")
    lines.append(f"- Test games: `{split_meta['test_games']}`")
    lines.append(f"- Train rows: `{split_meta['train_rows']}`")
    lines.append(f"- Test rows: `{split_meta['test_rows']}`")
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
    lines.append("## Calibration (uniform bins)")
    lines.append("")
    lines.append("| Bin | Predicted | Empirical |")
    lines.append("| --- | ---: | ---: |")
    for row in calibration_rows:
        lines.append(
            f"| {row['bin']} | {row['predicted']:.6f} | {row['empirical']:.6f} |"
        )
    lines.append("")
    lines.append("## Top Coefficients (absolute)")
    lines.append("")
    lines.append("| Feature | Coefficient | Abs |")
    lines.append("| --- | ---: | ---: |")
    for item in top_coefficients:
        lines.append(
            f"| {item['feature']} | {item['coefficient']:.6f} | {item['abs']:.6f} |"
        )
    lines.append("")
    lines.append("## Artifact")
    lines.append(f"- Model path: `{output_model}`")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train pairwise logistic win-probability model from arena snapshots."
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
        default="models/winprob_logreg_v1.pkl",
        help="Output model artifact path.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="reports/winprob_logreg_v1.md",
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
        "--c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for logistic regression.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Maximum optimizer iterations.",
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
    X_train = train_df[feature_columns].astype(float).to_numpy(dtype=float)
    y_train = train_df["label"].astype(int).to_numpy(dtype=int)
    X_test = test_df[feature_columns].astype(float).to_numpy(dtype=float)
    y_test = test_df["label"].astype(int).to_numpy(dtype=int)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    C=float(args.c),
                    max_iter=int(args.max_iter),
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    test_prob = pipeline.predict_proba(X_test)[:, 1]
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

    prob_true, prob_pred = calibration_curve(y_test, test_prob, n_bins=10, strategy="uniform")
    calibration_rows = []
    for idx, (pred, true) in enumerate(zip(prob_pred, prob_true), start=1):
        calibration_rows.append(
            {"bin": idx, "predicted": float(pred), "empirical": float(true)}
        )

    classifier = pipeline.named_steps["classifier"]
    coef = classifier.coef_[0]
    coeff_rows = []
    for feature, value in zip(feature_columns, coef):
        coeff_rows.append(
            {"feature": feature, "coefficient": float(value), "abs": float(abs(value))}
        )
    coeff_rows.sort(key=lambda item: item["abs"], reverse=True)
    top_coefficients = coeff_rows[:20]

    artifact = {
        "version": "winprob_logreg_v1",
        "model_type": "pairwise_logreg",
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        "feature_columns": feature_columns,
        "tie_policy": args.tie_policy,
        "pipeline": pipeline,
        "metrics": metrics,
        "dataset_meta": dataset_meta,
        "split_meta": split_meta,
    }

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_model)

    report_md = _build_report_markdown(
        snapshots_path=args.snapshots,
        dataset_meta=dataset_meta,
        split_meta=split_meta,
        metrics=metrics,
        calibration_rows=calibration_rows,
        top_coefficients=top_coefficients,
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
