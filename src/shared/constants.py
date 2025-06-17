"""
Fuente única de configuración para todo el proyecto:
– IDs y regiones de GCP
– rutas de GCS
– defaults de Vertex AI
"""

from __future__ import annotations

# ──────────────────── CONFIG GLOBAL GCP ────────────────────
PROJECT_ID: str = "trading-ai-460823"
REGION: str = "europe-west1"
GCS_BUCKET_NAME: str = "trading-ai-models-460823"
BASE_GCS_PATH = f"gs://{GCS_BUCKET_NAME}"

# ──────────────────── RUTAS DE ARTEFACTOS ───────────────────────
# AJUSTE: Rutas reorganizadas por función para mayor claridad y mantenimiento.

# Rutas de ejecución de la pipeline
PIPELINE_ROOT = f"{BASE_GCS_PATH}/pipeline_root_v3"
STAGING_PATH = f"{BASE_GCS_PATH}/staging_for_custom_jobs"
TENSORBOARD_LOGS_PATH = f"{BASE_GCS_PATH}/tensorboard_logs_v3"

# Rutas de datos
DATA_PATH = f"{BASE_GCS_PATH}/data"
DATA_FILTERED_FOR_OPT_PATH = f"{BASE_GCS_PATH}/params/data_filtered_for_opt_v3"

# Rutas de parámetros y optimización
PARAMS_PATH = f"{BASE_GCS_PATH}/params"
# AJUSTE: Se añaden rutas específicas para cada etapa de optimización.
ARCHITECTURE_PARAMS_PATH = f"{PARAMS_PATH}/architecture_v3"
LOGIC_PARAMS_PATH = f"{PARAMS_PATH}/LSTM_v3"  # Contiene los parámetros de lógica de trading
RL_DATA_INPUTS_PATH = f"{PARAMS_PATH}/rl_inputs_v3" # Se mantiene por si se reutiliza

# Rutas de modelos
MODELS_PATH = f"{BASE_GCS_PATH}/models"
LSTM_MODELS_PATH = f"{MODELS_PATH}/LSTM_v3"
FILTER_MODELS_PATH = f"{MODELS_PATH}/Filter_v5"
PRODUCTION_MODELS_PATH = f"{MODELS_PATH}/production_v3"
RL_MODELS_PATH = f"{MODELS_PATH}/RL_v3" # Se mantiene por si se reutiliza

# Rutas de resultados
BACKTEST_RESULTS_PATH = f"{BASE_GCS_PATH}/backtest_results_v3"

# ──────────────────── SERVICE ACCOUNTS Y SECRETS ─────────────────────
VERTEX_LSTM_SERVICE_ACCOUNT: str = (
    "data-ingestion-agent@trading-ai-460823.iam.gserviceaccount.com"
)
POLYGON_API_KEY_SECRET_NAME: str = "polygon-api-key"
POLYGON_API_KEY_SECRET_VERSION = "latest"

# ──────────────────── VERTEX AI DEFAULTS ───────────────────
# AJUSTE: Configuración de hardware centralizada y consistente.
DEFAULT_VERTEX_GPU_MACHINE_TYPE = "n1-standard-8"
DEFAULT_VERTEX_GPU_ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
DEFAULT_VERTEX_GPU_ACCELERATOR_COUNT = 1
DEFAULT_VERTEX_CPU_MACHINE_TYPE = "n1-standard-4"

# ──────────────────── PIPELINE & TRADING DEFAULTS ────────────────────
DEFAULT_TIMEFRAME = "15minute"

SPREADS_PIP = {
    "EURUSD": 0.8,
    #"GBPUSD": 1.0,
    #"USDJPY": 0.9,
    #"AUDUSD": 1.1,
    #"USDCAD": 1.2,
}

# AJUSTE: Se añaden parámetros dummy para la fase de preparación de datos,
# evitando hardcodear valores en los componentes.
DUMMY_INDICATOR_PARAMS = {
    "sma_len": 50,
    "rsi_len": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "stoch_len": 14,
}

# ──────────────────── PUB/SUB TOPICS ───────────────────────
SUCCESS_TOPIC_ID = "data-ingestion-success"
FAILURE_TOPIC_ID = "data-ingestion-failures"