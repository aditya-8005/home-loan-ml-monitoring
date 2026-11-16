# Home Loan Prediction with Automated Model Retraining and Monitoring

## 1. Project Overview

This project is an academic, collaborative machine learning engineering project designed to demonstrate an end-to-end ML lifecycle. The system aims to automate the prediction of home loan approvals, manage model experiments and registration, serve real-time predictions via a lightweight REST API, monitor production data for statistical drift, and trigger automated retraining with a champion-challenger promotion strategy.

The architecture is deliberately kept lightweight, modular, and beginner-friendly, tailored to run smoothly on standard laptops with 8–16 GB of RAM without requiring complex distributed clusters or heavyweight infrastructure.

---

## 2. Problem Statement

Financial institutions process thousands of home loan applications. Manual evaluation of these applications is time-consuming, prone to subjective bias, and operationally expensive. While machine learning classifiers can automate credit risk assessment and approval decisions, real-world data distributions change over time due to macroeconomic conditions, shifting customer demographics, and inflation (data drift).

Without continuous monitoring and automated retraining mechanisms, deployed models suffer from performance degradation in production. This project addresses the challenge by coupling automated model tracking, drift detection, and controlled model promotion into a streamlined, maintainable MLOps pipeline.

---

## 3. Project Objectives

- **Model Training & Experiment Tracking:** Train classification models on historical loan application records while logging metrics, parameters, and artifacts using MLflow.
- **Model Registry & Governance:** Track model versions and alias the top-performing model as the `champion`.
- **Inference Service:** Expose a clean, validated prediction interface using FastAPI and Pydantic schemas.
- **Statistical Drift Detection:** Implement statistical tests (e.g., Kolmogorov-Smirnov test for numerical features, Chi-Square test for categorical features) to detect distributional drift between reference baseline data and incoming production data.
- **Automated Retraining & Promotion:** Automatically trigger retraining when drift is confirmed and promote a challenger model only if its validation performance strictly surpasses the active champion.
- **Observability Dashboard:** Provide a lightweight visual dashboard using Streamlit to observe application status, drift indicators, and model metrics.

---

## 4. Current Development Status vs. Planned Features

> **Status Notice:** This project is currently in its initial foundation stage. The directory skeleton, dependencies, environment templates, and placeholder modules have been established. Machine learning logic, APIs, and dashboards are planned for upcoming milestones and are not yet functional.

### Implemented (Foundation)
- Repository structure and modular organization.
- Dependency definition via `requirements.txt`.
- Environment configuration template (`.env.example`) and Python version specification (`.python-version`).
- Modular placeholders for data loading, preprocessing, training, evaluation, prediction, drift detection, serving, and testing.

### Planned Features (To Be Implemented in Future Tasks)
- **Data Preprocessing & Feature Engineering:** Handling missing values, encoding categorical variables, and scaling numeric features.
- **Model Training Pipeline:** Training classification models (e.g., Logistic Regression, Random Forest) and tracking runs in MLflow.
- **Model Registry Management:** Automated assignment of champion and challenger aliases in the MLflow Model Registry.
- **FastAPI Prediction Service:** Endpoints for single and batch predictions with Pydantic request/response validation.
- **Drift Detection Engine:** Statistical drift testing comparing incoming inference payloads against reference baseline distributions.
- **Automated Retraining Loop:** Safe retraining workflow triggered by drift detection with conditional model promotion.
- **Streamlit Monitoring Dashboard:** Real-time visualization of drift reports, model metrics, and system status.
- **Docker Containerization:** Dockerfile and Docker Compose configurations for containerized deployment.

---

## 5. Technology Stack

- **Language:** Python 3.11
- **Data Manipulation & Computation:** Pandas, NumPy, SciPy
- **Machine Learning:** Scikit-learn, Joblib
- **Experiment Tracking & Registry:** MLflow
- **API Framework & Server:** FastAPI, Uvicorn (standard), Pydantic
- **Testing & HTTP Client:** Pytest, HTTPX
- **Monitoring & Visualization:** Streamlit
- **Containerization (Planned):** Docker Desktop

---

## 6. Architecture Overview

```
                      +-----------------------------+
                      |   Historical Training Data  |
                      |       (data/raw/train.csv)   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Preprocessing Pipeline   |
                      |   (src/preprocessing.py)    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Model Training & Tuning   |
                      |       (src/train.py)        |
                      +--------------+--------------+
                                     |
                      +--------------v--------------+
                      |       MLflow Tracking       |
                      |   & Model Registry (Champion)|
                      +--------------+--------------+
                                     |
         +---------------------------+---------------------------+
         |                                                       |
         v                                                       v
+-------------------------+                           +-------------------------+
|   FastAPI Service       |                           |  Drift Detection Engine |
|   (api/main.py)         |                           | (src/drift_detection.py)|
+------------+------------+                           +------------+------------+
             |                                                     |
             | Incoming Inference Requests                         | Detects Statistical
             v (data/raw/test.csv / payloads)                      | Distribution Shift
+-------------------------+                                        |
|   Live Predictions      |                                        v
|   (src/predict.py)      |                           +-------------------------+
+-------------------------+                           | Automated Retraining &  |
             |                                        | Challenger Evaluation   |
             | Logged Payloads                        |     (src/retrain.py)    |
             v                                        +------------+------------+
+-------------------------+                                        |
|   Streamlit Dashboard   |<---------------------------------------+
| (monitoring/dashboard.py|  (Pushes new champion if challenger outperforms)
+-------------------------+
```

---

## 7. Project Directory Structure

```text
home-loan-ml-monitoring/
│
├── data/
│   ├── raw/                  # Original, immutable datasets (train.csv, test.csv)
│   ├── processed/            # Cleaned, transformed data ready for model ingestion
│   └── reference/            # Baseline dataset used for statistical drift comparison
│
├── src/                      # Core machine learning package
│   ├── __init__.py           # Package marker
│   ├── config.py             # Global configurations, paths, and environment variables
│   ├── data_loader.py        # Dataset loading and splitting routines
│   ├── preprocessing.py      # Feature transformers and imputation pipelines
│   ├── train.py              # Model training script with MLflow logging
│   ├── evaluate.py           # Evaluation metrics computation and validation
│   ├── predict.py            # Inference engine utilizing the registered champion model
│   ├── drift_detection.py    # Statistical tests for numerical and categorical drift
│   └── retrain.py            # Retraining trigger and champion-vs-challenger evaluation
│
├── api/                      # REST API serving layer
│   ├── __init__.py           # Package marker
│   ├── main.py               # FastAPI application setup and endpoint routes
│   └── schemas.py            # Pydantic data models for request and response validation
│
├── monitoring/               # Monitoring and visualization layer
│   ├── __init__.py           # Package marker
│   ├── metrics.py            # Custom drift and prediction metric aggregation
│   └── dashboard.py          # Streamlit monitoring web application
│
├── tests/                    # Automated unit and integration tests
│   ├── test_preprocessing.py # Tests for data cleaning and feature transformations
│   ├── test_api.py           # Tests for FastAPI prediction endpoints
│   └── test_drift.py         # Tests for statistical drift detection routines
│
├── reports/                  # Generated figures, drift reports, and evaluation summaries
│
├── .gitignore                # Git ignore specifications
├── .env.example              # Environment variables template
├── .python-version           # Target Python runtime version (3.11)
├── requirements.txt          # Minimal project dependencies
└── README.md                 # Project documentation
```

---

## 8. Dataset Description

The project uses the standard Home Loan Approval dataset, comprised of two primary files placed in `data/raw/`:

1. **`train.csv` (Labeled Training Dataset):**
   - Contains historical applicant profiles with demographic and financial attributes (e.g., `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Gender`, `Married`, `Education`, `Property_Area`).
   - Includes the binary target variable **`Loan_Status`** (`Y` for approved, `N` for rejected).
   - Used for training, validation, and establishing reference baseline distributions.

2. **`test.csv` (Unlabeled Evaluation Dataset):**
   - Contains the same feature columns as `train.csv` but lacks the `Loan_Status` target column.
   - Intended for simulating real-world batch or streaming inference requests, testing FastAPI serving endpoints, and driving drift detection scenarios.

---

## 9. Local Setup Instructions (Windows PowerShell)

Follow these steps to set up the local development environment on Windows PowerShell.

### 9.1. Prerequisites
- **Python 3.11** installed and accessible from PowerShell.
- **Git** installed and configured.

### 9.2. Clone and Navigate to the Repository
```powershell
# Navigate into the project folder
cd "e:\cloud project\home-loan-ml-monitoring"
```

### 9.3. Virtual Environment Creation and Activation
```powershell
# Create a virtual environment named .venv using Python 3.11
python -m venv .venv

# If PowerShell execution policy restricts scripts, allow local scripts:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

### 9.4. Dependency Installation
```powershell
# Upgrade pip to the latest release
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

### 9.5. Environment Configuration
Copy the configuration template to initialize your local environment:
```powershell
Copy-Item .env.example .env
```
*(Note: `.env` is ignored by Git to prevent committing sensitive machine-specific configurations.)*

---

## 10. Running Tests

Automated testing is managed through `pytest`. Once test suites are implemented in upcoming milestones, execute tests with:

```powershell
# Run the complete test suite
pytest

# Run tests with detailed output and coverage information
pytest -v -s
```

---

## 11. Planned MLflow, FastAPI, and Docker Usage

### MLflow (Experiment Tracking & Model Registry)
- **Local Server Startup:**
  ```powershell
  mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
  ```
- **Tracking:** Training runs log hyperparameters, ROC-AUC, F1-score, confusion matrices, and serialized model artifacts.
- **Model Registry:** The best-performing candidate model is registered under `HomeLoanClassifier` and tagged with the alias `champion`.

### FastAPI (Model Serving)
- **Application Startup (Planned):**
  ```powershell
  uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
  ```
- **Endpoints:**
  - `GET /health`: Health-check and model metadata endpoint.
  - `POST /predict`: Real-time loan approval prediction endpoint accepting validated JSON payloads.
  - `POST /predict/batch`: Batch inference endpoint for multiple applications.

### Streamlit (Monitoring Dashboard)
- **Dashboard Startup (Planned):**
  ```powershell
  streamlit run monitoring/dashboard.py
  ```
- Provides interactive visual summaries of data drift metrics, historical prediction distributions, and active champion model details.

### Docker Usage (Planned)
The system is containerized using Docker and Docker Compose. For the full containerized deployment guide, service architectures, and commands, see [Section 12: Docker Deployment](#12-docker-deployment).

---

## 12. Docker Deployment

The project provides full containerization using Docker and Docker Compose, orchestrating three services on an isolated bridge network (`home-loan-network`):
- **MLflow Tracking Server & Model Registry** (Port 5000)
- **FastAPI Inference Service** (Port 8000)
- **Streamlit Monitoring Dashboard** (Port 8501)

### 12.1. Prerequisites

- **Docker Desktop** installed and running on Windows, macOS, or Linux.
- **Docker Compose** installed (Docker Compose v2 `docker compose` or standalone Compose v1 `docker-compose`).
- **Required Host Ports:** Ensure ports `5000`, `8000`, and `8501` are not in use by other local applications.
- **Port Customization:** If host port conflicts exist, configure alternative host ports by overriding `MLFLOW_PORT`, `API_PORT`, or `DASHBOARD_PORT` in your local `.env` file without altering container definitions.

### 12.2. Environment Configuration

Initialize the local environment configuration file from the template before launching containers:

```powershell
# Windows PowerShell
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

*(Note: `.env` is excluded by `.gitignore` and `.dockerignore` to prevent local configurations from leaking into version control or Docker build contexts.)*

**Configurable Environment Variables:**
- `MLFLOW_TRACKING_URI`: Defaults to `http://localhost:5000` for running Python scripts on the host against the published MLflow container. Within Docker Compose, container-to-container communication is automatically configured to use Docker internal DNS: `http://mlflow:5000`.
- `MODEL_NAME`: Name of the registered model in MLflow (default: `HomeLoanClassifier`).
- `MODEL_ALIAS`: Target model alias for production inference (default: `champion`).
- `MLFLOW_PORT`: Published host port for the MLflow UI (default: `5000`).
- `API_PORT`: Published host port for the FastAPI service (default: `8000`).
- `DASHBOARD_PORT`: Published host port for the Streamlit dashboard (default: `8501`).

### 12.3. Deployment Commands

The following commands use Docker Compose v2 syntax (`docker compose`). If using standalone Compose v1, use `docker-compose` instead.

```powershell
# 1. Build container images and start all services in detached mode
docker compose up --build -d

# 2. Check service status and healthcheck conditions
docker compose ps

# 3. View streaming aggregated logs from all services
docker compose logs -f

# 4. View streaming logs for an individual service
docker compose logs -f api
docker compose logs -f mlflow
docker compose logs -f dashboard

# 5. Restart services
docker compose restart

# 6. Stop all services (preserves database and artifact volumes)
docker compose down
```

### 12.4. Service URLs and Verification

Once services are running and report healthy status in `docker compose ps`, access and verify each service:

| Service | Host URL | Description | Verification Method |
| :--- | :--- | :--- | :--- |
| **MLflow UI** | [http://localhost:5000](http://localhost:5000) | Tracking server & Model Registry | Open in browser; check `/health` returns HTTP 200 |
| **FastAPI Swagger Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive OpenAPI documentation | Open in browser to explore and execute endpoints |
| **FastAPI Health** | [http://localhost:8000/health](http://localhost:8000/health) | Service & model availability check | `curl http://localhost:8000/health` |
| **Streamlit Dashboard** | [http://localhost:8501](http://localhost:8501) | Live metrics & drift observability | Open in browser |

**Verifying Service Health via CLI:**
```powershell
# Inspect FastAPI health status
curl http://localhost:8000/health

# Inspect MLflow server health
curl http://localhost:5000/health
```

### 12.5. First-Time Model Initialization (Cold-Start)

#### Cold-Start Behavior
When launching the container stack for the first time, the MLflow database is empty. FastAPI starts up gracefully in **degraded mode** (`status: "degraded"`, `model_loaded: false`) because no model version tagged with alias `champion` exists in the registry. The API service endpoints remain operational, but prediction requests will return HTTP 503 until a champion model is trained and registered.

#### Initial Training and Champion Registration
To train the baseline model and register it as `champion` in the MLflow Model Registry, execute the training pipeline inside the running `home-loan-api` container:

```powershell
docker compose exec api python -c "from src.train import run_training; run_training(enable_mlflow=True)"
```
*(Compose v1 alternative: `docker-compose exec api python -c "from src.train import run_training; run_training(enable_mlflow=True)"`)*

> **Implementation Note on `enable_mlflow=True`:**
> In `src/train.py`, the CLI block `if __name__ == "__main__":` invokes `run_training()` with its default parameter `enable_mlflow=False` (which only trains and evaluates validation metrics locally without logging). To log parameters, validation metrics, confusion matrix artifacts, and automatically register the model under `HomeLoanClassifier` with the `champion` alias, `enable_mlflow=True` **must** be explicitly passed.

#### Verifying the Model in MLflow
1. Open the MLflow UI in your browser at [http://localhost:5000](http://localhost:5000).
2. Click on the **Models** tab in the top navigation bar.
3. Verify that `HomeLoanClassifier` is present.
4. Click into `HomeLoanClassifier` and confirm that **Version 1** has the alias **`champion`** assigned.

#### Reloading the Model in FastAPI
Once the champion model is registered in MLflow, trigger FastAPI's dynamic reload endpoint without restarting the container:

```powershell
curl -X POST http://localhost:8000/reload-model
```

Verify that the API status transitions from `degraded` to `healthy`:
```powershell
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "HomeLoanClassifier",
  "model_alias": "champion",
  "model_version": "1",
  "uptime_seconds": 120.5,
  "timestamp": "2026-11-16T12:00:00+00:00"
}
```

### 12.6. Storage and Persistence

Docker Compose defines three dedicated named volumes:

- **`mlflow_db`** (mounted at `/mlflow-db` in `mlflow`):
  Persists the SQLite database (`sqlite:////mlflow-db/mlflow.db`) containing experiment metadata, run parameters, metrics, tags, and model registry records.
- **`mlflow_artifacts`** (mounted at `/mlflow-artifacts` in `mlflow` [read-write] and `api` [read-write]):
  Stores serialized model pipelines and file artifacts logged during training runs. Both containers require read-write access because MLflow uses a direct local filesystem artifact store, and the API training process writes model artifacts directly to this shared volume. Both services share the identical path `/mlflow-artifacts`, ensuring direct local filesystem resolution without cross-container URI mismatch.
- **`shared_reports`** (mounted at `/app/reports` in `api` and `dashboard`):
  Stores runtime monitoring data, specifically `reports/metrics.json`. The FastAPI service appends live inference metrics and prediction counts, while the Streamlit dashboard reads this file to render real-time observability charts.

**Volume Lifecycle Management:**
- Running `docker compose down` stops and removes containers while **preserving** all named volumes and persisted data.
- Running `docker compose down -v` permanently **deletes** the named volumes, erasing MLflow tracking records, registered models, and recorded monitoring metrics.

### 12.7. Operational Limitations

- **Cold-Start Degraded Mode:** On first launch or after volume resets (`docker compose down -v`), FastAPI starts in `degraded` mode until the initial champion model is trained and registered.
- **No Background Task Scheduler:** While data drift detection (`src/drift_detection.py`) and automated retraining logic (`src/retrain.py`) are fully implemented and verified in the codebase, a background scheduling mechanism (such as Celery, Airflow, or Cron) is **not yet implemented**.
- **Manual Triggering:** Starting Docker Compose boots the services (MLflow, API, Dashboard) but does not automatically execute training runs, run drift checks, or trigger model retraining without external calls.

---

## 13. Team Collaboration and Git Branching Workflow

To maintain code quality in a collaborative college environment, adhere to the following workflow:

1. **Branching Strategy:**
   - `main`: Production-ready, stable codebase. Direct commits to `main` are restricted.
   - `feature/<feature-name>`: Dedicated feature branches for individual assignments (e.g., `feature/data-preprocessing`, `feature/mlflow-training`, `feature/fastapi-serving`, `feature/drift-monitoring`).
2. **Workflow Steps:**
   ```powershell
   # 1. Fetch latest changes from remote
   git fetch origin

   # 2. Create and switch to a feature branch
   git checkout -b feature/<feature-name>

   # 3. Work on changes, test locally, and stage files
   git add <modified-files>

   # 4. Commit with descriptive messages
   git commit -m "feat: implement data preprocessing and imputation"

   # 5. Push feature branch to GitHub
   git push origin feature/<feature-name>
   ```
3. **Pull Requests & Code Reviews:**
   - Open a Pull Request (PR) targeting `main`.
   - Ensure all unit tests pass before requesting review.
   - At least one team member must review and approve the PR before merging.

---

## 14. Future Implementation Milestones

- **Milestone 1 (Current):** Project foundation, directory scaffolding, dependencies, and documentation.
- **Milestone 2:** Data loading, exploratory data analysis, and preprocessing pipeline (`src/preprocessing.py`).
- **Milestone 3:** Baseline model training, hyperparameter optimization, and MLflow experiment tracking (`src/train.py`, `src/evaluate.py`).
- **Milestone 4:** FastAPI prediction service and Pydantic validation schemas (`api/`).
- **Milestone 5:** Statistical data drift detection engine (`src/drift_detection.py`).
- **Milestone 6:** Automated retraining workflow and champion-challenger promotion mechanism (`src/retrain.py`).
- **Milestone 7:** Streamlit monitoring dashboard (`monitoring/dashboard.py`).
- **Milestone 8:** Docker containerization, integration testing, and final project demonstration.
