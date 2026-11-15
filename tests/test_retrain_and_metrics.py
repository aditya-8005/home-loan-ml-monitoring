"""Unit tests for automated retraining, model evaluation, and persistent metrics store."""

import os
os.environ["TESTING"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from monitoring.metrics import MetricsStore
from src.evaluate import compare_models, evaluate_model
from src.retrain import orchestrate_retraining
from src.predict import predictor


class MockEstimator:
    """Mock classifier with configurable predictions and probabilities."""

    def __init__(self, predictions, proba_class_1=0.8):
        self.predictions = np.array(predictions)
        self.proba_class_1 = proba_class_1

    def predict(self, X):
        return self.predictions[: len(X)]

    def predict_proba(self, X):
        n = len(X)
        p1 = self.proba_class_1
        return np.column_stack([np.full(n, 1.0 - p1), np.full(n, p1)])


@pytest.fixture
def sample_eval_data():
    """Create a small valid evaluation dataset."""
    n = 20
    df = pd.DataFrame(
        {
            "Loan_ID": [f"LP001{i:03d}" for i in range(n)],
            "ApplicantIncome": [5000.0] * n,
            "CoapplicantIncome": [0.0] * n,
            "LoanAmount": [120.0] * n,
            "Loan_Amount_Term": [360.0] * n,
            "Credit_History": [1.0] * n,
            "Gender": ["Male"] * n,
            "Married": ["Yes"] * n,
            "Dependents": ["0"] * n,
            "Education": ["Graduate"] * n,
            "Self_Employed": ["No"] * n,
            "Property_Area": ["Urban"] * n,
            "Loan_Status": ["Y"] * (n - 4) + ["N"] * 4,
        }
    )
    return df


def test_metrics_store_json_persistence():
    """Test MetricsStore accurately persists and reloads state from a JSON file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "test_metrics.json"
        store = MetricsStore(filepath=json_path, auto_save=True)

        # Record predictions
        store.record_prediction(loan_status="Y", latency_ms=15.0, success=True)
        store.record_prediction(loan_status="N", latency_ms=25.0, success=True)
        store.record_prediction(success=False)

        # Record drift
        store.record_drift_event(
            is_drift_detected=True,
            drift_percentage=36.4,
            drifted_features=["ApplicantIncome", "Credit_History"],
        )

        # Record retraining
        store.record_retraining_event(
            success=True,
            promoted=True,
            champion_score=0.80,
            candidate_score=0.86,
            new_model_version="champion-v2",
            evaluation_metric="f1_score",
        )

        assert json_path.exists()

        # Create a new store instance pointing to same JSON file
        restored_store = MetricsStore(filepath=json_path, auto_save=False)
        m = restored_store.get_metrics()

        assert m["total_predictions"] == 3
        assert m["successful_predictions"] == 2
        assert m["failed_predictions"] == 1
        assert m["approved_predictions"] == 1
        assert m["rejected_predictions"] == 1
        assert m["last_drift_score"] == 36.4
        assert m["last_drift_status"] == "DRIFT DETECTED"
        assert "ApplicantIncome" in m["last_drifted_features"]
        assert m["retraining_attempts"] == 1
        assert m["successful_promotions"] == 1
        assert m["current_model_version"] == "champion-v2"
        assert m["champion_performance"] == 0.80
        assert m["candidate_performance"] == 0.86


def test_model_evaluation_metrics(sample_eval_data):
    """Test evaluate_model calculates Accuracy, Precision, Recall, F1, and ROC-AUC."""
    X = sample_eval_data.drop(columns=["Loan_ID", "Loan_Status"])
    y = sample_eval_data["Loan_Status"]

    # Model predicting perfect match
    perfect_model = MockEstimator(predictions=["Y"] * 16 + ["N"] * 4, proba_class_1=0.95)
    metrics = evaluate_model(perfect_model, X, y)

    assert metrics["accuracy"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert "roc_auc" in metrics


def test_compare_models_candidate_wins(sample_eval_data):
    """Test candidate is promoted when its score strictly exceeds champion's score."""
    X = sample_eval_data.drop(columns=["Loan_ID", "Loan_Status"])
    y = sample_eval_data["Loan_Status"]

    candidate = MockEstimator(predictions=["Y"] * 16 + ["N"] * 4)  # 100% accurate
    champion = MockEstimator(predictions=["Y"] * 10 + ["N"] * 10)  # less accurate

    should_promote, summary = compare_models(
        candidate_model=candidate,
        champion_model=champion,
        X_eval=X,
        y_eval=y,
        primary_metric="f1_score",
    )

    assert should_promote is True
    assert "PROMOTE" in summary["decision"]
    assert summary["candidate_score"] > summary["champion_score"]


def test_compare_models_candidate_loses(sample_eval_data):
    """Test candidate is rejected when it performs worse than the champion."""
    X = sample_eval_data.drop(columns=["Loan_ID", "Loan_Status"])
    y = sample_eval_data["Loan_Status"]

    candidate = MockEstimator(predictions=["N"] * 20)  # 0% F1
    champion = MockEstimator(predictions=["Y"] * 16 + ["N"] * 4)  # 100% F1

    should_promote, summary = compare_models(
        candidate_model=candidate,
        champion_model=champion,
        X_eval=X,
        y_eval=y,
        primary_metric="f1_score",
    )

    assert should_promote is False
    assert "REJECT" in summary["decision"]


def test_orchestrate_retraining_skipped_when_no_drift():
    """Test retraining workflow does nothing when drift is not detected."""
    no_drift_report = {
        "is_drift_detected": False,
        "drift_percentage": 9.1,
        "drifted_features": ["Gender"],
    }

    result = orchestrate_retraining(drift_report=no_drift_report, force_retrain=False)

    assert result["status"] == "skipped_no_drift"
    assert result["promoted"] is False
    assert result["drift_detected"] is False
    assert "not exceeded" in result["message"]


def test_orchestrate_retraining_when_drift_detected(sample_eval_data):
    """Test retraining workflow trains candidate and promotes it when it outperforms champion."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "retrain_data.csv"
        sample_eval_data.to_csv(csv_path, index=False)

        drift_report = {
            "is_drift_detected": True,
            "drift_percentage": 27.3,
            "drifted_features": ["ApplicantIncome", "Credit_History", "Property_Area"],
        }

        # Candidate model that performs very well
        mock_candidate = MockEstimator(predictions=["Y"] * 16 + ["N"] * 4)

        # Baseline champion with lower performance
        mock_champion = MockEstimator(predictions=["Y"] * 8 + ["N"] * 12)
        predictor.set_model(mock_champion, model_alias="champion")

        with patch("src.retrain.train_model", return_value=(mock_candidate, {"f1_score": 0.95})):
            result = orchestrate_retraining(
                drift_report=drift_report,
                new_training_data_path=csv_path,
                evaluation_data_path=csv_path,
                evaluation_metric="f1_score",
            )

            assert result["status"] == "promoted"
            assert result["drift_detected"] is True
            assert result["promoted"] is True
            assert result["candidate_score"] > result["champion_score"]
            assert predictor.is_loaded is True
            assert "promoted" in result["message"].lower()
