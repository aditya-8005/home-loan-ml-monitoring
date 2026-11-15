"""Statistical tests and data drift detection module.

This module monitors distribution drift between a reference baseline dataset
and incoming production inference data using SciPy statistical tests:
- Numerical features: Two-sample Kolmogorov-Smirnov (KS) test
- Categorical features: Pearson's Chi-Square test of independence
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from src.config import PROJECT_ROOT, RAW_DATA_DIR, REFERENCE_DATA_DIR

logger = logging.getLogger(__name__)

# Default reference and production dataset paths
DEFAULT_REFERENCE_PATH: Path = REFERENCE_DATA_DIR / "reference.csv"
DEFAULT_PRODUCTION_PATH: Path = RAW_DATA_DIR / "test.csv"

# Academic default thresholds specified for the project
DEFAULT_P_VALUE_THRESHOLD: float = 0.05
DEFAULT_DRIFT_SHARE_THRESHOLD: float = 0.20  # 20% of monitored features

# Standard feature type groupings for home loan approval dataset
NUMERICAL_FEATURES: List[str] = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
]

CATEGORICAL_FEATURES: List[str] = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Credit_History",
    "Property_Area",
]


@dataclass
class FeatureDriftResult:
    """Stores drift test results for an individual feature."""

    feature_name: str
    feature_type: str  # 'numerical' or 'categorical'
    test_name: str  # 'Kolmogorov-Smirnov' or 'Chi-Square'
    statistic: float
    p_value: float
    drift_detected: bool
    threshold: float = DEFAULT_P_VALUE_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "feature_name": self.feature_name,
            "feature_type": self.feature_type,
            "test_name": self.test_name,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 5),
            "drift_detected": self.drift_detected,
            "threshold": self.threshold,
        }


@dataclass
class DriftReport:
    """Encapsulates the complete dataset drift detection summary and report."""

    feature_results: Dict[str, FeatureDriftResult] = field(default_factory=dict)
    p_value_threshold: float = DEFAULT_P_VALUE_THRESHOLD
    drift_share_threshold: float = DEFAULT_DRIFT_SHARE_THRESHOLD

    @property
    def total_features(self) -> int:
        """Total number of monitored features evaluated."""
        return len(self.feature_results)

    @property
    def drifted_features(self) -> List[str]:
        """List of feature names where drift was detected."""
        return [
            name
            for name, res in self.feature_results.items()
            if res.drift_detected
        ]

    @property
    def drift_count(self) -> int:
        """Number of features exhibiting statistically significant drift."""
        return len(self.drifted_features)

    @property
    def drift_percentage(self) -> float:
        """Percentage of monitored features exhibiting drift (0.0 to 100.0)."""
        if self.total_features == 0:
            return 0.0
        return round((self.drift_count / self.total_features) * 100.0, 2)

    @property
    def is_drift_detected(self) -> bool:
        """Whether overall dataset drift exceeds the configured share threshold."""
        if self.total_features == 0:
            return False
        return (self.drift_count / self.total_features) >= self.drift_share_threshold

    @property
    def status(self) -> str:
        """Human-readable overall status string."""
        return "DRIFT DETECTED" if self.is_drift_detected else "NO DRIFT DETECTED"

    def generate_report(self) -> str:
        """Generate human-readable text report formatted according to project specification.

        Format:
        Drift Detection Report

        ApplicantIncome: Drift Detected
        LoanAmount: No Drift
        ...

        Drifted Features: 3
        Total Features: 10
        Drift Percentage: 30%

        Status: DRIFT DETECTED
        """
        lines = ["Drift Detection Report", ""]

        for feature_name, res in self.feature_results.items():
            status_text = "Drift Detected" if res.drift_detected else "No Drift"
            lines.append(f"{feature_name}: {status_text}")

        lines.append("")
        lines.append(f"Drifted Features: {self.drift_count}")
        lines.append(f"Total Features: {self.total_features}")
        # Format percentage nicely (e.g. 30% or 33.33%)
        pct_str = f"{int(self.drift_percentage)}%" if self.drift_percentage.is_integer() else f"{self.drift_percentage:.1f}%"
        lines.append(f"Drift Percentage: {pct_str}")
        lines.append("")
        lines.append(f"Status: {self.status}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert overall report into a structured dictionary."""
        return {
            "status": self.status,
            "is_drift_detected": self.is_drift_detected,
            "drift_count": self.drift_count,
            "total_features": self.total_features,
            "drift_percentage": self.drift_percentage,
            "drift_share_threshold": self.drift_share_threshold,
            "p_value_threshold": self.p_value_threshold,
            "drifted_features": self.drifted_features,
            "feature_metrics": {
                k: v.to_dict() for k, v in self.feature_results.items()
            },
        }

    def __str__(self) -> str:
        return self.generate_report()


def compute_numerical_drift(
    reference: pd.Series,
    current: pd.Series,
    feature_name: str = "numerical_feature",
    p_value_threshold: float = DEFAULT_P_VALUE_THRESHOLD,
) -> FeatureDriftResult:
    """Perform Two-sample Kolmogorov-Smirnov (KS) test on numerical data.

    Parameters
    ----------
    reference : pd.Series
        Baseline reference feature values.
    current : pd.Series
        Incoming production feature values.
    feature_name : str
        Name of the feature being tested.
    p_value_threshold : float, default=0.05
        Significance level below which drift is flagged.

    Returns
    -------
    FeatureDriftResult
        Result containing test statistic, p-value, and drift boolean.
    """
    ref_clean = pd.to_numeric(reference, errors="coerce").dropna()
    curr_clean = pd.to_numeric(current, errors="coerce").dropna()

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        logger.warning(
            "Feature '%s' has insufficient non-null values for KS test.",
            feature_name,
        )
        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="numerical",
            test_name="Kolmogorov-Smirnov",
            statistic=0.0,
            p_value=1.0,
            drift_detected=False,
            threshold=p_value_threshold,
        )

    res = ks_2samp(ref_clean, curr_clean)
    statistic = float(res.statistic)
    p_val = float(res.pvalue)
    drift = bool(p_val < p_value_threshold)

    return FeatureDriftResult(
        feature_name=feature_name,
        feature_type="numerical",
        test_name="Kolmogorov-Smirnov",
        statistic=statistic,
        p_value=p_val,
        drift_detected=drift,
        threshold=p_value_threshold,
    )


def compute_categorical_drift(
    reference: pd.Series,
    current: pd.Series,
    feature_name: str = "categorical_feature",
    p_value_threshold: float = DEFAULT_P_VALUE_THRESHOLD,
) -> FeatureDriftResult:
    """Perform Chi-Square test of independence on categorical distributions.

    Constructs a 2-way contingency table of observed category frequencies
    between reference and production datasets.

    Parameters
    ----------
    reference : pd.Series
        Baseline reference feature values.
    current : pd.Series
        Incoming production feature values.
    feature_name : str
        Name of the feature being tested.
    p_value_threshold : float, default=0.05
        Significance level below which drift is flagged.

    Returns
    -------
    FeatureDriftResult
        Result containing test statistic, p-value, and drift boolean.
    """
    ref_clean = reference.dropna().astype(str)
    curr_clean = current.dropna().astype(str)

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        logger.warning(
            "Feature '%s' has insufficient non-null values for Chi-Square test.",
            feature_name,
        )
        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="categorical",
            test_name="Chi-Square",
            statistic=0.0,
            p_value=1.0,
            drift_detected=False,
            threshold=p_value_threshold,
        )

    ref_counts = ref_clean.value_counts()
    curr_counts = curr_clean.value_counts()
    all_categories = sorted(set(ref_counts.index).union(set(curr_counts.index)))

    # At least two categories required for Chi-square contingency test
    if len(all_categories) < 2:
        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="categorical",
            test_name="Chi-Square",
            statistic=0.0,
            p_value=1.0,
            drift_detected=False,
            threshold=p_value_threshold,
        )

    ref_freqs = [ref_counts.get(cat, 0) for cat in all_categories]
    curr_freqs = [curr_counts.get(cat, 0) for cat in all_categories]

    contingency_table = np.array([ref_freqs, curr_freqs])

    try:
        chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
        statistic = float(chi2_stat)
        p_val = float(p_val)
        drift = bool(p_val < p_value_threshold)
    except Exception as exc:
        logger.warning(
            "Chi-Square test failed for feature '%s': %s", feature_name, exc
        )
        statistic = 0.0
        p_val = 1.0
        drift = False

    return FeatureDriftResult(
        feature_name=feature_name,
        feature_type="categorical",
        test_name="Chi-Square",
        statistic=statistic,
        p_value=p_val,
        drift_detected=drift,
        threshold=p_value_threshold,
    )


def load_reference_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load baseline reference data from CSV.

    Parameters
    ----------
    filepath : str | Path | None
        Path to reference dataset. Defaults to data/reference/reference.csv
        or data/raw/train.csv if reference.csv does not exist.
    """
    candidates = [
        Path(filepath) if filepath else None,
        DEFAULT_REFERENCE_PATH,
        RAW_DATA_DIR / "train.csv",
    ]
    for p in candidates:
        if p and p.exists():
            return pd.read_csv(p)

    raise FileNotFoundError(
        f"Reference dataset not found at '{filepath or DEFAULT_REFERENCE_PATH}'. "
        "Please provide a reference DataFrame or place baseline CSV in data/reference/."
    )


def load_production_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load incoming production-like data from CSV.

    Parameters
    ----------
    filepath : str | Path | None
        Path to production dataset. Defaults to data/raw/test.csv.
    """
    target_path = Path(filepath) if filepath else DEFAULT_PRODUCTION_PATH
    if not target_path.exists():
        raise FileNotFoundError(
            f"Production dataset not found at '{target_path}'. "
            "Please provide a production DataFrame or place CSV in data/raw/test.csv."
        )
    return pd.read_csv(target_path)


def detect_drift(
    reference_data: Union[pd.DataFrame, str, Path],
    production_data: Union[pd.DataFrame, str, Path],
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    p_value_threshold: float = DEFAULT_P_VALUE_THRESHOLD,
    drift_share_threshold: float = DEFAULT_DRIFT_SHARE_THRESHOLD,
) -> DriftReport:
    """Execute statistical drift detection between reference and production datasets.

    Parameters
    ----------
    reference_data : pd.DataFrame | str | Path
        Reference baseline dataset (or path to CSV).
    production_data : pd.DataFrame | str | Path
        Incoming production-like dataset (or path to CSV).
    numerical_features : list[str] | None
        Numerical feature columns to test with Kolmogorov-Smirnov test.
    categorical_features : list[str] | None
        Categorical feature columns to test with Chi-Square test.
    p_value_threshold : float, default=0.05
        Statistical significance p-value threshold for feature-level drift.
    drift_share_threshold : float, default=0.20
        Fraction of drifted features required to flag overall dataset drift (20%).

    Returns
    -------
    DriftReport
        Comprehensive drift report containing individual feature test results,
        drift percentage, and overall status.
    """
    ref_df = (
        load_reference_data(reference_data)
        if isinstance(reference_data, (str, Path))
        else reference_data.copy()
    )
    prod_df = (
        load_production_data(production_data)
        if isinstance(production_data, (str, Path))
        else production_data.copy()
    )

    # Determine numerical and categorical feature sets
    if numerical_features is None and categorical_features is None:
        # Default to canonical columns present in both datasets
        common_cols = [c for c in ref_df.columns if c in prod_df.columns]
        num_cols = [c for c in NUMERICAL_FEATURES if c in common_cols]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in common_cols]

        # If canonical columns not found, infer automatically from types
        if not num_cols and not cat_cols:
            for col in common_cols:
                if col in ("Loan_ID", "Loan_Status", "loan_id", "loan_status"):
                    continue
                if pd.api.types.is_numeric_dtype(ref_df[col]) and ref_df[col].nunique() > 5:
                    num_cols.append(col)
                else:
                    cat_cols.append(col)
    else:
        num_cols = [c for c in (numerical_features or []) if c in ref_df.columns and c in prod_df.columns]
        cat_cols = [c for c in (categorical_features or []) if c in ref_df.columns and c in prod_df.columns]

    feature_results: Dict[str, FeatureDriftResult] = {}

    # 1. Run Kolmogorov-Smirnov test for numerical features
    for col in num_cols:
        res = compute_numerical_drift(
            reference=ref_df[col],
            current=prod_df[col],
            feature_name=col,
            p_value_threshold=p_value_threshold,
        )
        feature_results[col] = res

    # 2. Run Chi-Square test for categorical features
    for col in cat_cols:
        res = compute_categorical_drift(
            reference=ref_df[col],
            current=prod_df[col],
            feature_name=col,
            p_value_threshold=p_value_threshold,
        )
        feature_results[col] = res

    report = DriftReport(
        feature_results=feature_results,
        p_value_threshold=p_value_threshold,
        drift_share_threshold=drift_share_threshold,
    )
    return report


# Convenience aliases
test_numerical_drift = compute_numerical_drift
test_categorical_drift = compute_categorical_drift


if __name__ == "__main__":
    # Demonstration CLI execution with synthetic samples if files do not exist
    print("Executing Demonstration Drift Detection...\n")
    np.random.seed(42)
    n = 200

    ref_demo = pd.DataFrame(
        {
            "ApplicantIncome": np.random.normal(5000, 1500, n),
            "CoapplicantIncome": np.random.normal(1500, 500, n),
            "LoanAmount": np.random.normal(140, 40, n),
            "Loan_Amount_Term": np.random.choice([360.0, 180.0], size=n, p=[0.85, 0.15]),
            "Gender": np.random.choice(["Male", "Female"], size=n, p=[0.8, 0.2]),
            "Married": np.random.choice(["Yes", "No"], size=n, p=[0.65, 0.35]),
            "Dependents": np.random.choice(["0", "1", "2", "3+"], size=n, p=[0.5, 0.2, 0.2, 0.1]),
            "Education": np.random.choice(["Graduate", "Not Graduate"], size=n, p=[0.75, 0.25]),
            "Self_Employed": np.random.choice(["No", "Yes"], size=n, p=[0.85, 0.15]),
            "Credit_History": np.random.choice([1.0, 0.0], size=n, p=[0.85, 0.15]),
            "Property_Area": np.random.choice(["Urban", "Semiurban", "Rural"], size=n, p=[0.35, 0.40, 0.25]),
        }
    )

    # Induce drift in 3 features: ApplicantIncome, Credit_History, Property_Area
    prod_demo = ref_demo.copy()
    prod_demo["ApplicantIncome"] = np.random.normal(9000, 2000, n)  # Substantial income inflation
    prod_demo["Credit_History"] = np.random.choice([1.0, 0.0], size=n, p=[0.40, 0.60])  # Credit drop
    prod_demo["Property_Area"] = np.random.choice(["Urban", "Semiurban", "Rural"], size=n, p=[0.75, 0.15, 0.10])

    rep = detect_drift(ref_demo, prod_demo)
    print(rep.generate_report())
