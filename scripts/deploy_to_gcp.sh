#!/bin/bash
set -x
# scripts/deploy_to_gcp.sh

set -euo pipefail

log() { printf '%(%F %T)T  %s\n' -1 "$*"; }

PROJECT_ID="trading-ai-460823" # O leer de una variable de entorno
REGION="europe-west1"
REPO_NAME="data-ingestion-repo"
IMAGE_NAME="data-ingestion-agent" # Nombre de la imagen base para los componentes

# --- 1. Construir y Subir la Imagen Docker Común ---
log "Paso 1: Construyendo y subiendo la imagen Docker común..."
VERSION_TAG=$(date +"%Y%m%d-%H%M%S")
COMMON_IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${VERSION_TAG}"

# Eliminar imágenes antiguas (opcional, pero buena práctica)
# gcloud artifacts docker images list "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}" --filter="package~${IMAGE_NAME}" --format="get(image)" | xargs -r -L1 gcloud artifacts docker images delete --quiet

docker build --no-cache -t "${COMMON_IMAGE_URI}" .
docker push "${COMMON_IMAGE_URI}"
log "✅ Imagen Docker común subida: ${COMMON_IMAGE_URI}"

# --- 2. Compilar y Desplegar la Pipeline ---
log "Paso 2: Compilando y desplegando la pipeline..."
# Usar el nuevo CLI de despliegue
python src/deploy/cli.py \
    --common-image-uri "${COMMON_IMAGE_URI}" \
    --output-json "algo_trading_mlops_pipeline_v5_final.json" \
    --timeframe "1h" \
    --n-trials-arch 20 \
    --n-trials-logic 30 \
    --backtest-years-to-keep 5 \
    --holdout-months 3

log "🚀 Proceso de despliegue completado."