#!/usr/bin/env bash
###############################################################################
# run.sh – entrypoint para la imagen runner-lstm
#
# ▸ Descarga el código fuente (*.py) desde un bucket GCS
# ▸ Ejecuta el script principal (por defecto train_lstm.py) con
#   los argumentos que reciba el contenedor.
#
# Variables de entorno opcionales:
#   CODE_GCS_URI   Prefijo GCS donde viven los *.py   (def: gs://trading-ai-models-460823/code)
#   MAIN_PY        Script principal a ejecutar        (def: train_lstm.py)
###############################################################################

set -euo pipefail

log() { printf '%(%F %T)T  %s\n' -1 "$*"; }

CODE_GCS_URI="${CODE_GCS_URI:-gs://trading-ai-models-460823/code}"
MAIN_PY="${MAIN_PY:-train_lstm.py}"

TMP_DIR="$(mktemp -d -t code_exec_XXXXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT

log "Directorio de trabajo: ${TMP_DIR}"
cd "${TMP_DIR}"

log "Sincronizando código desde ${CODE_GCS_URI} ..."
# -m == multithread; rsync replica jerarquía y actualiza solo cambios
gsutil -m rsync -r "${CODE_GCS_URI}" .

# Validación básica
if [[ ! -f "${MAIN_PY}" ]]; then
  log "❌  No se encontró ${MAIN_PY} tras la descarga."
  ls -la
  exit 1
fi

log "Contenido descargado:"
ls -la

log "Ejecutando: python ${MAIN_PY} $*"
exec python "${MAIN_PY}" "$@"
