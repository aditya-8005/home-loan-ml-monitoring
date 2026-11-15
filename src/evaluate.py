"""Model evaluation metrics and validation routines.

This module provides standardized model evaluation metrics (Accuracy, F1-Score,
Precision, Recall, ROC-AUC) and champion-versus-challenger comparison logic.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def _standardize_labels(y: Union[pd.Series, np.ndarray, list]) -> np.ndarray:
    """Standardize label array to binary 0/1 integers for scikit-learn metrics."""
    s = pd.Series(y).copy()
    # Map typical loan approval labels
    mapping = {
        "Y": 1,
        "y": 1,
        "Yes": 1,
        "Approved": 1,
        1: 1,
        1.0: 1,
        "1": 1,
        True: 1,
        "N": 0,
        "n": 0,
        "No": 0,
        "Rejected": 0,
        0: 0,
        0.0: 0,
        "0": 0,
        False: 0,
    }
    return s.map(mapping).fillna(0).astype(int).to_numpy()


def evaluate_model(
    model: Any,
    X_eval: pd.DataFrame,
    y_eval: Union[pd.Series, np.ndarray],
    positive_label: Union[str, int] = 1,
) -> Dict[str, float]:
    """Evaluate a trained model or pipeline on a labeled dataset.

    Parameters
    ----------
    model : Any
        Trained Scikit-learn estimator or Pipeline.
    X_eval : pd.DataFrame
        Evaluation features DataFrame.
    y_eval : pd.Series | np.ndarray
        Ground-truth labels.
    positive_label : str | int, default=1
        Label to treat as the positive class.

    Returns
    -------
    dict[str, float]
        Dictionary of computed performance metrics rounded to 4 decimals.
    """
    y_true_binary = _standardize_labels(y_eval)

    # Generate predictions
    raw_preds = model.predict(X_eval)
    y_pred_binary = _standardize_labels(raw_preds)

    acc = float(accuracy_score(y_true_binary, y_pred_binary))
    prec = float(
        precision_score(y_true_binary, y_pred_binary, zero_division=0)
    )
    rec = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
    f1 = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))

    metrics: Dict[str, float] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
    }

    # Calculate ROC-AUC if probability estimation is supported
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X_eval)
            if probas.shape[1] > 1:
                # Use class 1 probabilities
                auc = float(roc_auc_score(y_true_binary, probas[:, 1]))
                metrics["roc_auc"] = round(auc, 4)
        except Exception as exc:
            logger.debug("ROC-AUC computation skipped: %s", exc)

    return metrics


def compare_models(
    candidate_model: Any,
    champion_model: Optional[Any],
    X_eval: pd.DataFrame,
    y_eval: Union[pd.Series, np.ndarray],
    primary_metric: str = "f1_score",
    min_improvement: float = 0.0,
) -> Tuple[bool, Dict[str, Any]]:
    """Compare a newly retrained candidate model against the active champion model.

    Evaluates both candidate and champion on the EXACT same labeled evaluation
    dataset to guarantee fair, statistically rigorous promotion decisions.

    Parameters
    ----------
    candidate_model : Any
        Newly trained candidate model / pipeline.
    champion_model : Any | None
        Current active champion model / pipeline (or None if no champion exists).
    X_eval : pd.DataFrame
        Shared evaluation features dataset.
    y_eval : pd.Series | np.ndarray
        Shared ground-truth evaluation labels.
    primary_metric : str, default='f1_score'
        Metric key used as the promotion criterion (e.g. 'f1_score' or 'accuracy').
    min_improvement : float, default=0.0
        Minimum threshold difference required for candidate to win promotion.

    Returns
    -------
    tuple[bool, dict[str, Any]]
        (should_promote, summary_dictionary)
    """
    candidate_metrics = evaluate_model(candidate_model, X_eval, y_eval)

    # If no champion model exists, candidate automatically qualifies
    if champion_model is None:
        return True, {
            "candidate_metrics": candidate_metrics,
            "champion_metrics": None,
            "primary_metric": primary_metric,
            "candidate_score": candidate_metrics.get(primary_metric, 0.0),
            "champion_score": 0.0,
            "difference": candidate_metrics.get(primary_metric, 0.0),
            "decision": "PROMOTE (First model, no champion exists)",
        }

    champion_metrics = evaluate_model(champion_model, X_eval, y_eval)

    cand_score = candidate_metrics.get(primary_metric, 0.0)
    champ_score = champion_metrics.get(primary_metric, 0.0)
    diff = round(cand_score - champ_score, 4)

    # Strict performance superiority rule
    should_promote = bool(diff > min_improvement)
    decision = (
        f"PROMOTE ({primary_metric} improved by {diff:+.4f})"
        if should_promote
        else f"REJECT ({primary_metric} diff {diff:+.4f} did not exceed {min_improvement})"
    )

    summary = {
        "candidate_metrics": candidate_metrics,
        "champion_metrics": champion_metrics,
        "primary_metric": primary_metric,
        "candidate_score": cand_score,
        "champion_score": champ_score,
        "difference": diff,
        "decision": decision,
    }

    return should_promote, summary
