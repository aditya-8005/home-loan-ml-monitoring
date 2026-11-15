"""Unit tests for statistical drift detection routines."""

import numpy as np
import pandas as pd
import pytest

from src.drift_detection import (
    DEFAULT_DRIFT_SHARE_THRESHOLD,
    DEFAULT_P_VALUE_THRESHOLD,
    DriftReport,
    FeatureDriftResult,
    compute_categorical_drift,
    compute_numerical_drift,
    detect_drift,
    load_production_data,
    load_reference_data,
)


@pytest.fixture
def synthetic_datasets():
    """Generate paired reference and production datasets for testing."""
    np.random.seed(123)
    n = 250

    ref = pd.DataFrame(
        {
            "ApplicantIncome": np.random.normal(5000, 1000, n),
            "CoapplicantIncome": np.random.normal(1500, 400, n),
            "LoanAmount": np.random.normal(130, 30, n),
            "Loan_Amount_Term": np.random.choice([360.0, 180.0], size=n, p=[0.8, 0.2]),
            "Gender": np.random.choice(["Male", "Female"], size=n, p=[0.8, 0.2]),
            "Married": np.random.choice(["Yes", "No"], size=n, p=[0.65, 0.35]),
            "Dependents": np.random.choice(["0", "1", "2", "3+"], size=n, p=[0.5, 0.2, 0.2, 0.1]),
            "Education": np.random.choice(["Graduate", "Not Graduate"], size=n, p=[0.75, 0.25]),
            "Self_Employed": np.random.choice(["No", "Yes"], size=n, p=[0.85, 0.15]),
            "Credit_History": np.random.choice([1.0, 0.0], size=n, p=[0.85, 0.15]),
            "Property_Area": np.random.choice(["Urban", "Semiurban", "Rural"], size=n, p=[0.4, 0.35, 0.25]),
        }
    )

    # Production dataset without drift (sampled from same distributions)
    prod_no_drift = pd.DataFrame(
        {
            "ApplicantIncome": np.random.normal(5000, 1000, n),
            "CoapplicantIncome": np.random.normal(1500, 400, n),
            "LoanAmount": np.random.normal(130, 30, n),
            "Loan_Amount_Term": np.random.choice([360.0, 180.0], size=n, p=[0.8, 0.2]),
            "Gender": np.random.choice(["Male", "Female"], size=n, p=[0.8, 0.2]),
            "Married": np.random.choice(["Yes", "No"], size=n, p=[0.65, 0.35]),
            "Dependents": np.random.choice(["0", "1", "2", "3+"], size=n, p=[0.5, 0.2, 0.2, 0.1]),
            "Education": np.random.choice(["Graduate", "Not Graduate"], size=n, p=[0.75, 0.25]),
            "Self_Employed": np.random.choice(["No", "Yes"], size=n, p=[0.85, 0.15]),
            "Credit_History": np.random.choice([1.0, 0.0], size=n, p=[0.85, 0.15]),
            "Property_Area": np.random.choice(["Urban", "Semiurban", "Rural"], size=n, p=[0.4, 0.35, 0.25]),
        }
    )

    # Production dataset with heavy drift in 3 features:
    # 1. ApplicantIncome (numerical shift)
    # 2. Credit_History (categorical collapse)
    # 3. Property_Area (categorical distribution change)
    prod_drifted = prod_no_drift.copy()
    prod_drifted["ApplicantIncome"] = np.random.normal(9500, 1200, n)
    prod_drifted["Credit_History"] = np.random.choice([1.0, 0.0], size=n, p=[0.30, 0.70])
    prod_drifted["Property_Area"] = np.random.choice(["Urban", "Semiurban", "Rural"], size=n, p=[0.85, 0.10, 0.05])

    return ref, prod_no_drift, prod_drifted


def test_ks_numerical_no_drift():
    """Test KS test flags no drift for two samples from identical distribution."""
    np.random.seed(42)
    ref = pd.Series(np.random.normal(5000, 800, 300))
    prod = pd.Series(np.random.normal(5000, 800, 300))

    result = compute_numerical_drift(ref, prod, feature_name="ApplicantIncome", p_value_threshold=0.05)
    assert isinstance(result, FeatureDriftResult)
    assert result.feature_name == "ApplicantIncome"
    assert result.test_name == "Kolmogorov-Smirnov"
    assert result.p_value >= 0.05
    assert result.drift_detected is False


def test_ks_numerical_with_drift():
    """Test KS test correctly detects distributional drift for shifted continuous feature."""
    np.random.seed(42)
    ref = pd.Series(np.random.normal(5000, 800, 300))
    prod = pd.Series(np.random.normal(12000, 1500, 300))

    result = compute_numerical_drift(ref, prod, feature_name="ApplicantIncome", p_value_threshold=0.05)
    assert result.p_value < 0.05
    assert result.drift_detected is True


def test_chi2_categorical_no_drift():
    """Test Chi-Square test flags no drift for stable categorical proportions."""
    ref = pd.Series(["Urban"] * 100 + ["Rural"] * 50 + ["Semiurban"] * 100)
    prod = pd.Series(["Urban"] * 98 + ["Rural"] * 52 + ["Semiurban"] * 100)

    result = compute_categorical_drift(ref, prod, feature_name="Property_Area", p_value_threshold=0.05)
    assert isinstance(result, FeatureDriftResult)
    assert result.test_name == "Chi-Square"
    assert result.p_value >= 0.05
    assert result.drift_detected is False


def test_chi2_categorical_with_drift():
    """Test Chi-Square test detects drift when proportions invert."""
    ref = pd.Series(["Urban"] * 150 + ["Rural"] * 50)
    prod = pd.Series(["Urban"] * 20 + ["Rural"] * 180)

    result = compute_categorical_drift(ref, prod, feature_name="Property_Area", p_value_threshold=0.05)
    assert result.p_value < 0.05
    assert result.drift_detected is True


def test_missing_values_handled_gracefully():
    """Test that missing values (NaNs) do not crash numerical or categorical tests."""
    np.random.seed(42)
    ref = pd.Series([100.0, np.nan, 200.0, 300.0, np.nan] * 50)
    prod = pd.Series([105.0, 195.0, np.nan, 310.0, 150.0] * 50)

    num_result = compute_numerical_drift(ref, prod, feature_name="LoanAmount")
    assert not np.isnan(num_result.p_value)

    ref_cat = pd.Series(["Male", None, "Female", "Male", np.nan] * 50)
    prod_cat = pd.Series(["Male", "Female", np.nan, "Male", "Female"] * 50)
    cat_result = compute_categorical_drift(ref_cat, prod_cat, feature_name="Gender")
    assert not np.isnan(cat_result.p_value)


def test_detect_drift_no_drift_dataset(synthetic_datasets):
    """Test detect_drift on non-drifted datasets returns NO DRIFT DETECTED."""
    ref, prod_no_drift, _ = synthetic_datasets
    report = detect_drift(ref, prod_no_drift)

    assert isinstance(report, DriftReport)
    assert report.total_features == 11
    # Very few or zero features should drift by chance
    assert report.drift_percentage < 20.0
    assert report.is_drift_detected is False
    assert report.status == "NO DRIFT DETECTED"


def test_detect_drift_drifted_dataset(synthetic_datasets):
    """Test detect_drift on drifted dataset flags overall drift and identifies drifted features."""
    ref, _, prod_drifted = synthetic_datasets
    report = detect_drift(ref, prod_drifted)

    assert report.total_features == 11
    assert "ApplicantIncome" in report.drifted_features
    assert "Credit_History" in report.drifted_features
    assert "Property_Area" in report.drifted_features
    assert report.drift_count >= 3
    assert report.drift_percentage >= 20.0
    assert report.is_drift_detected is True
    assert report.status == "DRIFT DETECTED"


def test_report_formatting_matches_specification(synthetic_datasets):
    """Test generated text report structure matches academic specification."""
    ref, _, prod_drifted = synthetic_datasets
    report = detect_drift(ref, prod_drifted)
    text_report = report.generate_report()

    # Verify key sections
    assert "Drift Detection Report" in text_report
    assert "ApplicantIncome: Drift Detected" in text_report
    assert "Credit_History: Drift Detected" in text_report
    assert "Property_Area: Drift Detected" in text_report
    assert f"Drifted Features: {report.drift_count}" in text_report
    assert f"Total Features: {report.total_features}" in text_report
    assert "Drift Percentage:" in text_report
    assert "Status: DRIFT DETECTED" in text_report


def test_custom_thresholds(synthetic_datasets):
    """Test user-configurable p-value and drift percentage thresholds."""
    ref, _, prod_drifted = synthetic_datasets

    # If drift_share_threshold is set to 90%, 3/11 (27%) should not trigger overall drift
    strict_report = detect_drift(
        ref, prod_drifted, drift_share_threshold=0.90
    )
    assert strict_report.is_drift_detected is False
    assert strict_report.status == "NO DRIFT DETECTED"


def test_missing_reference_file_raises_error():
    """Test load_reference_data raises FileNotFoundError when path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_reference_data("non_existent_reference_file_xyz.csv")


def test_missing_production_file_raises_error():
    """Test load_production_data raises FileNotFoundError when path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_production_data("non_existent_production_file_xyz.csv")
