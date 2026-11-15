"""Automated retraining orchestration and champion-challenger promotion pipeline.

This module coordinates:
1. Verification of drift detection signals.
2. Automated invocation of the reusable training pipeline in src/train.py.
3. Champion-versus-challenger evaluation on common labeled evaluation datasets.
4. Conditional promotion of the challenger if and only if it outperforms the active champion.
5. Notification and metrics tracking in the persistent monitoring store.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import (
    MODEL_ALIAS,
    MODEL_NAME,
    PROJECT_ROOT,
    RAW_DATA_DIR,
)
from src.data_loader import (
    DEFAULT_TRAIN_PATH,
    get_features_and_target,
    load_training_data,
)
from src.drift_detection import DriftReport, detect_drift
from src.evaluate import compare_models, evaluate_model
from src.predict import predictor
from src.train import MLFLOW_AVAILABLE, train_model
from monitoring.metrics import metrics_tracker

logger = logging.getLogger(__name__)

# Safe MLflow client import


def orchestrate_retraining(
    drift_report: Optional[Union[DriftReport, Dict[str, Any]]] = None,
    new_training_data_path: Optional[Union[str, Path]] = None,
    evaluation_data_path: Optional[Union[str, Path]] = None,
    reference_data_path: Optional[Union[str, Path]] = None,
    production_data_path: Optional[Union[str, Path]] = None,
    evaluation_metric: str = "f1_score",
    min_improvement: float = 0.0,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """Execute the end-to-end retraining and champion promotion workflow.

    Parameters
    ----------
    drift_report : DriftReport | dict | None
        Existing drift evaluation report. If None, evaluates drift between
        reference_data_path and production_data_path.
    new_training_data_path : str | Path | None
        Path to newly available labeled training data.
    evaluation_data_path : str | Path | None
        Path to labeled holdout dataset used to compare champion vs challenger.
        If None, a stratified sample from the training dataset is used.
    reference_data_path : str | Path | None
        Path to reference baseline (used if drift_report is None).
    production_data_path : str | Path | None
        Path to production inference dataset (used if drift_report is None).
    evaluation_metric : str, default='f1_score'
        Primary metric for model promotion decision (e.g. 'f1_score' or 'accuracy').
    min_improvement : float, default=0.0
        Minimum threshold difference required for candidate to win promotion.
    force_retrain : bool, default=False
        If True, triggers retraining regardless of drift status (for manual maintenance).

    Returns
    -------
    dict[str, Any]
        Structured summary containing decision status, candidate vs champion scores,
        and promotion details.
    """
    logger.info("Starting automated retraining orchestration check...")

    # 1. Evaluate or verify drift signal
    if drift_report is None and reference_data_path and production_data_path:
        drift_report = detect_drift(reference_data_path, production_data_path)

    drift_detected = False
    drift_score = 0.0
    drifted_features = []

    if drift_report is not None:
        if isinstance(drift_report, DriftReport):
            drift_detected = drift_report.is_drift_detected
            drift_score = drift_report.drift_percentage
            drifted_features = drift_report.drifted_features
        elif isinstance(drift_report, dict):
            drift_detected = bool(drift_report.get("is_drift_detected", False))
            drift_score = float(drift_report.get("drift_percentage", 0.0))
            drifted_features = list(drift_report.get("drifted_features", []))

        # Record drift evaluation event in monitoring store
        metrics_tracker.record_drift_event(
            is_drift_detected=drift_detected,
            drift_percentage=drift_score,
            drifted_features=drifted_features,
        )

    # 2. Gatekeeper: Skip if no drift detected and not forced
    if not drift_detected and not force_retrain:
        msg = (
            f"Drift threshold not exceeded (drift score: {drift_score}%). "
            "Automated retraining was not triggered."
        )
        logger.info(msg)
        return {
            "status": "skipped_no_drift",
            "drift_detected": False,
            "drift_score": drift_score,
            "promoted": False,
            "message": msg,
            "comparison": None,
        }

    logger.info(
        "Drift confirmed (score: %s%%). Initiating model retraining pipeline...",
        drift_score,
    )

    # 3. Call reusable training pipeline from src/train.py
    try:
        candidate_model, train_val_metrics = train_model(
            data_path=new_training_data_path,
            experiment_name="HomeLoanRetraining",
            register_model=True,
            model_alias="challenger",
        )
    except Exception as train_exc:
        logger.error("Candidate model training failed: %s", train_exc)
        metrics_tracker.record_retraining_event(
            success=False,
            promoted=False,
            details={"error": str(train_exc)},
        )
        return {
            "status": "training_failed",
            "drift_detected": True,
            "promoted": False,
            "error": str(train_exc),
            "message": f"Training failed with error: {train_exc}",
        }

    # 4. Load labeled evaluation dataset for fair comparison
    eval_df = None
    if evaluation_data_path and Path(evaluation_data_path).exists():
        eval_df = load_training_data(evaluation_data_path)
    elif new_training_data_path and Path(new_training_data_path).exists():
        eval_df = load_training_data(new_training_data_path)
    elif DEFAULT_TRAIN_PATH.exists():
        eval_df = load_training_data(DEFAULT_TRAIN_PATH)

    if eval_df is None or eval_df.empty:
        # Fallback to validation metrics if evaluation data is unavailable
        logger.warning("No evaluation dataset provided; comparing against training validation metrics.")
        cand_score = train_val_metrics.get(evaluation_metric, 0.0)
        champ_score = metrics_tracker.champion_performance or 0.0
        should_promote = cand_score > champ_score
        comparison_details = {
            "candidate_metrics": train_val_metrics,
            "champion_metrics": {"baseline": champ_score},
            "candidate_score": cand_score,
            "champion_score": champ_score,
            "difference": round(cand_score - champ_score, 4),
            "decision": "PROMOTE" if should_promote else "REJECT",
        }
    else:
        # Extract evaluation features and labels
        X_eval, y_eval = get_features_and_target(eval_df)

        # Retrieve current champion model
        champion_model = predictor._model if predictor.is_loaded else None

        # Compare candidate vs champion on exact same evaluation dataset
        should_promote, comparison_details = compare_models(
            candidate_model=candidate_model,
            champion_model=champion_model,
            X_eval=X_eval,
            y_eval=y_eval,
            primary_metric=evaluation_metric,
            min_improvement=min_improvement,
        )

    cand_score = comparison_details["candidate_score"]
    champ_score = comparison_details["champion_score"]
    new_version_tag = f"champion-v{metrics_tracker.successful_promotions + 2}"

    # 5. Model Promotion Decision
    if should_promote:
        logger.info(
            "Promoting candidate model to champion! (%s improved from %s to %s)",
            evaluation_metric,
            champ_score,
            cand_score,
        )

        # Assign alias in MLflow if available
        if MLFLOW_AVAILABLE:
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                client.set_registered_model_alias(
                    MODEL_NAME, "champion", "latest"
                )
            except Exception:
                pass

        # Update in-memory serving predictor so API serves the new champion immediately
        predictor.set_model(
            candidate_model,
            model_alias="champion",
            model_version=new_version_tag,
        )

        # Update monitoring store
        metrics_tracker.record_retraining_event(
            success=True,
            promoted=True,
            champion_score=champ_score,
            candidate_score=cand_score,
            new_model_version=new_version_tag,
            evaluation_metric=evaluation_metric,
            details=comparison_details,
        )

        status_result = "promoted"
        msg = (
            f"Candidate model successfully outperformed champion ({evaluation_metric}: "
            f"{cand_score} vs {champ_score}) and was promoted to champion."
        )
    else:
        logger.info(
            "Candidate model did not outperform champion (%s: %s vs %s). Retaining current champion.",
            evaluation_metric,
            cand_score,
            champ_score,
        )

        # Candidate rejected, champion retained
        metrics_tracker.record_retraining_event(
            success=True,
            promoted=False,
            champion_score=champ_score,
            candidate_score=cand_score,
            evaluation_metric=evaluation_metric,
            details=comparison_details,
        )

        status_result = "rejected_no_improvement"
        msg = (
            f"Candidate model did not outperform champion ({evaluation_metric}: "
            f"{cand_score} vs {champ_score}). Existing champion was retained."
        )

    return {
        "status": status_result,
        "drift_detected": drift_detected,
        "drift_score": drift_score,
        "promoted": should_promote,
        "active_model_version": metrics_tracker.current_model_version,
        "candidate_score": cand_score,
        "champion_score": champ_score,
        "metric_evaluated": evaluation_metric,
        "message": msg,
        "comparison": comparison_details,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Retraining orchestration module loaded.")
