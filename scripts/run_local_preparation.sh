#!/bin/bash
# ---------------------------------------------------------------------
# run_local_preparation.sh: Ejecuta la preparación de datos localmente.
# ---------------------------------------------------------------------
#
# Este script invoca el componente de preparación de datos para que se
# ejecute en la máquina local. Descargará los datos ingeridos desde GCS,
# los procesará y subirá los datos preparados de nuevo a GCS.
#
# Requisitos:
# - Haber ejecutado previamente `run_local_ingestion.sh`.
# - Tener Google Cloud SDK autenticado.

set -e

# --- Configuración ---
export PROJECT_ID="$(gcloud config get-value project)"
export PAIR="EURUSD"
export TIMEFRAME="m1"
export YEARS_TO_KEEP=10
export HOLDOUT_MONTHS=6

# --- Rutas de GCS ---
# La ruta de entrada debe coincidir con la salida de `run_local_ingestion.sh`
export INPUT_GCS_PATH="gs://${PROJECT_ID}-vertex-ai-pipeline-assets/data/ingested/${PAIR}/${TIMEFRAME}/dukascopy_${PAIR}_${TIMEFRAME}.parquet"

# La ruta base de salida para los datos preparados
export OUTPUT_GCS_BASE_PATH="gs://${PROJECT_ID}-vertex-ai-pipeline-assets/data/prepared"

# --- Ejecución del Componente ---
echo "🚀 Iniciando la preparación de datos local para ${PAIR}..."
echo "   - Leyendo desde: ${INPUT_GCS_PATH}"
echo "   - Datos preparados se subirán a: ${OUTPUT_GCS_BASE_PATH}/${PAIR}"

# Invocamos el task del componente directamente
python -m src.components.data_preparation.task \
    --input-data-path "${INPUT_GCS_PATH}" \
    --output-gcs-path "${OUTPUT_GCS_BASE_PATH}" \
    --years-to-keep "${YEARS_TO_KEEP}" \
    --holdout-months "${HOLDOUT_MONTHS}"

echo "✅ Preparación de datos completada. Los archivos están en GCS."
