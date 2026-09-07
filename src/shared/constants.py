"""
Configuración central del proyecto.

Los identificadores de infraestructura se leen desde variables de entorno para
mantener el repositorio portable y evitar acoplar el código a una cuenta GCP
concreta. Los valores por defecto son placeholders seguros para documentación.
"""

from __future__ import annotations

import os

# ──────────────────── CONFIG GLOBAL GCP ────────────────────
PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id")
REGION: str = os.getenv("GCP_REGION", "europe-west1")
GCS_BUCKET_NAME: str = os.getenv("GCS_BUCKET_NAME", "your-gcs-bucket")
BASE_GCS_PATH = f"gs://{GCS_BUCKET_NAME}"

# ──────────────────── RUTAS DE ARTEFACTOS ──────────────────
PIPELINE_ROOT = f"{BASE_GCS_PATH}/pipeline_root_v3"
STAGING_PATH = f"{BASE_GCS_PATH}/staging_for_custom_jobs"
TENSORBOARD_LOGS_PATH = f"{BASE_GCS_PATH}/tensorboard_logs_v3"

# Rutas de datos
DATA_PATH = f"{BASE_GCS_PATH}/data"
RAW_DATA_PATH = f"{DATA_PATH}/raw"
DATA_FILTERED_FOR_OPT_PATH = f"{BASE_GCS_PATH}/params/data_filtered_for_opt_v3"

# Rutas de parámetros y optimización
PARAMS_PATH = f"{BASE_GCS_PATH}/params"
ARCHITECTURE_PARAMS_PATH = f"{PARAMS_PATH}/architecture_v3"
LOGIC_PARAMS_PATH = f"{PARAMS_PATH}/LSTM_v3"
RL_DATA_INPUTS_PATH = f"{PARAMS_PATH}/rl_inputs_v3"

# Rutas de modelos
MODELS_PATH = f"{BASE_GCS_PATH}/models"
LSTM_MODELS_PATH = f"{MODELS_PATH}/LSTM_v3"
FILTER_MODELS_PATH = f"{MODELS_PATH}/Filter_v5"
PRODUCTION_MODELS_PATH = f"{MODELS_PATH}/production_v3"
RL_MODELS_PATH = f"{MODELS_PATH}/RL_v3"

# Rutas de resultados
BACKTEST_RESULTS_PATH = f"{BASE_GCS_PATH}/backtest_results_v3"

# ──────────────────── SERVICE ACCOUNTS Y SECRETS ───────────
VERTEX_LSTM_SERVICE_ACCOUNT: str = os.getenv(
    "VERTEX_SERVICE_ACCOUNT",
    "your-service-account@your-gcp-project-id.iam.gserviceaccount.com",
)
POLYGON_API_KEY_SECRET_NAME: str = os.getenv(
    "POLYGON_API_KEY_SECRET_NAME", "polygon-api-key"
)
POLYGON_API_KEY_SECRET_VERSION: str = os.getenv(
    "POLYGON_API_KEY_SECRET_VERSION", "latest"
)

# ──────────────────── VERTEX AI DEFAULTS ───────────────────
DEFAULT_VERTEX_GPU_MACHINE_TYPE = "n1-standard-8"
DEFAULT_VERTEX_GPU_ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
DEFAULT_VERTEX_GPU_ACCELERATOR_COUNT = 1
DEFAULT_VERTEX_CPU_MACHINE_TYPE = "n1-standard-4"

# ──────────────────── PIPELINE & TRADING DEFAULTS ──────────
DEFAULT_TIMEFRAME = "15minute"

SPREADS_PIP = {
    "EURUSD": 0.8,
}

DUMMY_INDICATOR_PARAMS = {
    "sma_len": 50,
    "rsi_len": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "stoch_len": 14,
}

# ──────────────────── PUB/SUB TOPICS ───────────────────────
SUCCESS_TOPIC_ID = os.getenv("SUCCESS_TOPIC_ID", "data-ingestion-success")
FAILURE_TOPIC_ID = os.getenv("FAILURE_TOPIC_ID", "data-ingestion-failures")
