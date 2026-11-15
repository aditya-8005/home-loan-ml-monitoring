"""Unit and integration tests for FastAPI serving endpoints and schemas."""

import os
os.environ["TESTING"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from pydantic import ValidationError
from fastapi import HTTPException

from api.main import (
    app,
    get_app_info,
    get_health,
    get_metrics,
    predict,
    predict_batch,
    reload_model,
)
from api.schemas import (
    BatchPredictionRequest,
    HealthResponse,
    LoanApplication,
    MetricsResponse,
    PredictionResponse,
    ReloadModelResponse,
)
from src.predict import ModelNotLoadedError, predictor
from monitoring.metrics import metrics_tracker


class DummyClassifierWithProba:
    """Mock ML model implementing both predict and predict_proba."""

    def __init__(self, pred_class=1, proba=(0.2, 0.8), classes=(0, 1)):
        self.pred_class = pred_class
        self.proba = proba
        self.classes_ = np.array(classes)

    def predict(self, X):
        return np.array([self.pred_class] * len(X))

    def predict_proba(self, X):
        return np.array([self.proba] * len(X))


class DummyClassifierWithoutProba:
    """Mock ML model implementing predict without probability estimation."""

    def __init__(self, pred_class="Y"):
        self.pred_class = pred_class

    def predict(self, X):
        return [self.pred_class] * len(X)


@pytest.fixture(autouse=True)
def setup_teardown():
    """Reset predictor and metrics tracker around each test."""
    metrics_tracker.reset()
    yield
    metrics_tracker.reset()


SAMPLE_APPLICATION_PAYLOAD = {
    "Loan_ID": "LP001002",
    "Gender": "Male",
    "Married": "No",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5849.0,
    "CoapplicantIncome": 0.0,
    "LoanAmount": 128.0,
    "Loan_Amount_Term": 360.0,
    "Credit_History": 1.0,
    "Property_Area": "Urban",
}


def test_root_endpoint():
    """Test GET / endpoint logic returning basic application information."""
    data = get_app_info()
    assert data.status == "online"
    assert data.version == "1.0.0"
    assert "Home Loan" in data.name
    assert "/predict" in data.endpoints.values() or "POST /predict" in data.endpoints.values()


def test_health_endpoint_healthy():
    """Test GET /health returns 'healthy' when champion model is loaded."""
    mock_model = DummyClassifierWithProba()
    predictor.set_model(mock_model, model_name="HomeLoanClassifier", model_alias="champion")

    data = get_health()
    assert data.status == "healthy"
    assert data.model_loaded is True
    assert data.model_name == "HomeLoanClassifier"
    assert data.model_alias == "champion"
    assert data.uptime_seconds >= 0.0


def test_health_endpoint_degraded():
    """Test GET /health returns 'degraded' when champion model is unloaded."""
    predictor.unload_model()

    data = get_health()
    assert data.status == "degraded"
    assert data.model_loaded is False


def test_predict_approved():
    """Test /predict returns approval status 'Y' and positive confidence probability."""
    mock_model = DummyClassifierWithProba(pred_class=1, proba=(0.15, 0.85), classes=(0, 1))
    predictor.set_model(mock_model)

    application = LoanApplication(**SAMPLE_APPLICATION_PAYLOAD)
    result = predict(application)

    assert isinstance(result, PredictionResponse)
    assert result.loan_status == "Y"
    assert result.prediction == 1
    assert result.probability == 0.85
    assert result.loan_id == "LP001002"


def test_predict_rejected():
    """Test /predict returns rejection status 'N' and correct probability."""
    mock_model = DummyClassifierWithProba(pred_class=0, proba=(0.82, 0.18), classes=(0, 1))
    predictor.set_model(mock_model)

    payload = dict(SAMPLE_APPLICATION_PAYLOAD)
    payload["Credit_History"] = 0.0
    application = LoanApplication(**payload)
    result = predict(application)

    assert result.loan_status == "N"
    assert result.prediction == 0
    assert result.probability == 0.18


def test_predict_without_probability_support():
    """Test /predict returns None for probability if unsupported by model."""
    mock_model = DummyClassifierWithoutProba(pred_class="Y")
    predictor.set_model(mock_model)

    application = LoanApplication(**SAMPLE_APPLICATION_PAYLOAD)
    result = predict(application)

    assert result.loan_status == "Y"
    assert result.probability is None


def test_predict_with_snake_case_aliases():
    """Test LoanApplication schema accepts snake_case aliases and scores correctly."""
    mock_model = DummyClassifierWithProba(pred_class=1, proba=(0.1, 0.9))
    predictor.set_model(mock_model)

    alias_payload = {
        "loan_id": "LP009999",
        "gender": "Female",
        "married": "Yes",
        "dependents": "1",
        "education": "Graduate",
        "self_employed": "Yes",
        "applicant_income": 4500.0,
        "coapplicant_income": 1200.0,
        "loan_amount": 115.0,
        "loan_amount_term": 360.0,
        "credit_history": 1.0,
        "property_area": "Semiurban",
    }
    application = LoanApplication(**alias_payload)
    result = predict(application)

    assert result.loan_status == "Y"
    assert result.loan_id == "LP009999"


def test_predict_model_not_loaded_503():
    """Test /predict raises HTTP 503 when model cannot be loaded."""
    predictor.unload_model()

    application = LoanApplication(**SAMPLE_APPLICATION_PAYLOAD)
    with patch.object(predictor, "load_model", side_effect=ModelNotLoadedError("Registry offline")):
        with pytest.raises(HTTPException) as exc_info:
            predict(application)
        assert exc_info.value.status_code == 503
        assert "unavailable" in exc_info.value.detail.lower()


def test_predict_validation_missing_required_income():
    """Test Pydantic validation fails when ApplicantIncome is omitted."""
    invalid_payload = dict(SAMPLE_APPLICATION_PAYLOAD)
    del invalid_payload["ApplicantIncome"]

    with pytest.raises(ValidationError):
        LoanApplication(**invalid_payload)


def test_predict_validation_negative_income():
    """Test Pydantic validation fails when ApplicantIncome is negative."""
    invalid_payload = dict(SAMPLE_APPLICATION_PAYLOAD)
    invalid_payload["ApplicantIncome"] = -100.0

    with pytest.raises(ValidationError):
        LoanApplication(**invalid_payload)


def test_predict_validation_negative_loan_amount():
    """Test Pydantic validation fails when LoanAmount is negative."""
    invalid_payload = dict(SAMPLE_APPLICATION_PAYLOAD)
    invalid_payload["LoanAmount"] = -50.0

    with pytest.raises(ValidationError):
        LoanApplication(**invalid_payload)


def test_predict_validation_invalid_term():
    """Test Pydantic validation fails when Loan_Amount_Term is zero or negative."""
    invalid_payload = dict(SAMPLE_APPLICATION_PAYLOAD)
    invalid_payload["Loan_Amount_Term"] = 0.0

    with pytest.raises(ValidationError):
        LoanApplication(**invalid_payload)


def test_metrics_endpoint():
    """Test /metrics reflects accurate serving metrics and updates after prediction."""
    mock_model = DummyClassifierWithProba(pred_class=1, proba=(0.2, 0.8))
    predictor.set_model(mock_model)

    # Initial metrics
    m1 = get_metrics()
    assert m1.total_predictions == 0
    assert m1.approved_predictions == 0

    # Perform 1 approved prediction
    application = LoanApplication(**SAMPLE_APPLICATION_PAYLOAD)
    predict(application)

    m2 = get_metrics()
    assert m2.total_predictions == 1
    assert m2.successful_predictions == 1
    assert m2.approved_predictions == 1
    assert m2.rejected_predictions == 0
    assert m2.approval_rate == 1.0
    assert m2.average_latency_ms >= 0.0


def test_reload_model_success():
    """Test /reload-model reloads champion model successfully."""
    with patch.object(predictor, "reload", return_value={"model_loaded": True}):
        predictor.set_model(DummyClassifierWithProba(), model_version="v2")
        result = reload_model()
        assert isinstance(result, ReloadModelResponse)
        assert result.status == "success"
        assert "reloaded" in result.message.lower()


def test_reload_model_failure_503():
    """Test /reload-model raises HTTP 503 when MLflow reload fails."""
    with patch.object(predictor, "reload", side_effect=ModelNotLoadedError("Registry unreachable")):
        with pytest.raises(HTTPException) as exc_info:
            reload_model()
        assert exc_info.value.status_code == 503
        assert "failed" in exc_info.value.detail.lower()


def test_batch_prediction_endpoint():
    """Test batch prediction scores multiple applications."""
    mock_model = DummyClassifierWithProba(pred_class=1, proba=(0.1, 0.9))
    predictor.set_model(mock_model)

    app1 = LoanApplication(**SAMPLE_APPLICATION_PAYLOAD)
    app2 = LoanApplication(**{**SAMPLE_APPLICATION_PAYLOAD, "Loan_ID": "LP001003", "ApplicantIncome": 3000.0})

    batch_request = BatchPredictionRequest(applications=[app1, app2])
    results = predict_batch(batch_request)

    assert len(results) == 2
    assert results[0].loan_id == "LP001002"
    assert results[1].loan_id == "LP001003"
    assert results[0].loan_status == "Y"
