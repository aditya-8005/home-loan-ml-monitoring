"""Model training and MLflow experiment tracking pipeline.

This module provides a reusable model training function that combines
the Scikit-learn preprocessing ColumnTransformer with a classification estimator,
evaluates validation performance, and logs runs to MLflow.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    MLFLOW_TRACKING_URI,
    MODEL_NAME as DEFAULT_MODEL_NAME,
    PROJECT_ROOT,
    RAW_DATA_DIR,
)
from src.data_loader import (
    DEFAULT_TRAIN_PATH,
    get_features_and_target,
    load_training_data,
)
from src.evaluate import evaluate_model
from src.preprocessing import create_preprocessor

logger = logging.getLogger(__name__)

# Safe MLflow import
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MlflowClient = None
    MLFLOW_AVAILABLE = False


def train_model(
    data_path: Optional[Union[str, Path]] = None,
    experiment_name: str = "HomeLoanApproval",
    register_model: bool = True,
    model_alias: Optional[str] = "challenger",
    model_name: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = 6,
) -> Tuple[Any, Dict[str, Any]]:
    """Train a classification pipeline on labeled loan application records.

    Integrates feature preprocessing, model fitting, validation evaluation,
    and optional MLflow tracking and model registry registration.

    Parameters
    ----------
    data_path : str | Path | None
        Path to labeled training CSV dataset. If None, uses data/raw/train.csv.
    experiment_name : str, default='HomeLoanApproval'
        MLflow experiment name for tracking.
    register_model : bool, default=True
        Whether to register the trained model in the MLflow Model Registry.
    model_alias : str | None, default='challenger'
        Alias to assign to the registered model (e.g. 'challenger' or 'champion').
    model_name : str | None
        Registry model name. Defaults to MODEL_NAME from config ('HomeLoanClassifier').
    test_size : float, default=0.2
        Validation split fraction.
    random_state : int, default=42
        Reproducibility seed.
    n_estimators : int, default=100
        Number of trees in Random Forest.
    max_depth : int, default=6
        Maximum depth of trees.

    Returns
    -------
    tuple[Any, dict[str, Any]]
        (trained_pipeline, validation_metrics_dictionary)
    """
    resolved_model_name = model_name or DEFAULT_MODEL_NAME
    logger.info("Starting model training pipeline using dataset '%s'...", data_path)

    # 1. Load labeled dataset
    df = (
        load_training_data(data_path)
        if (data_path or DEFAULT_TRAIN_PATH.exists())
        else None
    )

    if df is None:
        raise FileNotFoundError(
            f"Training dataset not found at '{data_path or DEFAULT_TRAIN_PATH}'. "
            "Please ensure 'train.csv' exists in 'data/raw/' or pass a valid CSV path."
        )

    # 2. Extract predictor features and target
    features, target = get_features_and_target(df)

    # 3. Train-validation split with target stratification
    X_train, X_val, y_train, y_val = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    # 4. Construct complete preprocessing + estimator Pipeline
    preprocessor = create_preprocessor()
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    # 5. Fit pipeline
    pipeline.fit(X_train, y_train)

    # 6. Evaluate validation performance
    metrics = evaluate_model(pipeline, X_val, y_val)
    logger.info("Validation performance computed: %s", metrics)

    # 7. MLflow Experiment Logging & Model Registration (if MLflow is available)
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=f"train_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}") as run:
                # Log hyperparameters
                mlflow.log_params(
                    {
                        "model_type": "RandomForestClassifier",
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "random_state": random_state,
                        "test_size": test_size,
                    }
                )

                # Log evaluation metrics
                mlflow.log_metrics(metrics)

                # Log Scikit-learn model artifact
                reg_name = resolved_model_name if register_model else None
                model_info = mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    artifact_path="model",
                    registered_model_name=reg_name,
                )

                # Assign alias in registry if requested
                if register_model and model_alias and reg_name:
                    try:
                        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
                        # Resolve registered version
                        latest_versions = client.get_latest_versions(reg_name)
                        if latest_versions:
                            top_ver = latest_versions[0].version
                            client.set_registered_model_alias(
                                reg_name, model_alias, top_ver
                            )
                            logger.info(
                                "Model version %s assigned alias '%s' in registry '%s'.",
                                top_ver,
                                model_alias,
                                reg_name,
                            )
                    except Exception as alias_err:
                        logger.warning("Could not set MLflow model alias: %s", alias_err)

        except Exception as mlflow_err:
            logger.warning("MLflow tracking encountered an issue (training still succeeded): %s", mlflow_err)

    return pipeline, metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Training module loaded successfully.")
