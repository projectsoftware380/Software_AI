#!/bin/bash
# -------------------------------------------------------------------
# run_local_promotion.sh: Ejecuta la promoción de modelos localmente.
# -------------------------------------------------------------------
#
# Este script invoca el componente de promoción de modelos. Compara las
# métricas del backtest del nuevo modelo (recién entrenado) con las
# del modelo actualmente en producción. Si el nuevo modelo es mejor,
# lo copia a la carpeta de "producción" en GCS.
#
# Requisitos:
# - Haber ejecutado `run_local_backtest.sh`.
# - Tener Google Cloud SDK autenticado.

set -e

# --- Configuración ---
export PROJECT_ID="$(gcloud config get-value project)"
export PAIR="EURUSD"
export TIMEFRAME="m1"

# --- Rutas de GCS ---
# Directorio con las métricas del nuevo modelo (salida del backtest).
export NEW_METRICS_DIR="gs://${PROJECT_ID}-vertex-ai-pipeline-assets/backtest_results/${PAIR}/${TIMEFRAME}"

# Directorios con los artefactos del nuevo modelo.
export NEW_LSTM_DIR="gs://${PROJECT_ID}-vertex-ai-pipeline-assets/models/lstm/trained/${PAIR}/${TIMEFRAME}"
export NEW_FILTER_PATH="gs://${PROJECT_ID}-vertex-ai-pipeline-assets/models/filter/trained/${PAIR}/${TIMEFRAME}/filter_model.joblib"

# Directorio base donde se almacenan los modelos en producción.
export PRODUCTION_BASE_DIR="gs://${PROJECT_ID}-vertex-ai-pipeline-assets/models/production"

# --- Ejecución del Componente ---
echo "🚀 Iniciando la promoción de modelos local para ${PAIR}..."
echo "   - Comparando métricas de: ${NEW_METRICS_DIR}"

python -m src.components.model_promotion.task \
    --new-metrics-dir "${NEW_METRICS_DIR}" \
    --new-lstm-artifacts-dir "${NEW_LSTM_DIR}" \
    --new-filter-model-path "${NEW_FILTER_PATH}" \
    --production-base-dir "${PRODUCTION_BASE_DIR}" \
    --pair "${PAIR}" \
    --timeframe "${TIMEFRAME}"

echo "✅ Proceso de promoción completado."
