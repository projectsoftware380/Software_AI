#!/usr/bin/env bash
###############################################################################
# run.sh – entrypoint para una imagen de entrenamiento/ejecución
#
# Descarga código fuente desde GCS y ejecuta un script Python indicado por
# MAIN_PY. La ubicación GCS debe proporcionarse explícitamente para evitar
# acoplar la imagen a una cuenta o bucket concreto.
###############################################################################

set -euo pipefail

log() { printf '%(%F %T)T  %s\n' -1 "$*"; }

: "${CODE_GCS_URI:?Falta CODE_GCS_URI, por ejemplo gs://your-bucket/code}"
MAIN_PY="${MAIN_PY:-train_lstm.py}"

TMP_DIR="$(mktemp -d -t code_exec_XXXXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT

log "Directorio de trabajo: ${TMP_DIR}"
cd "${TMP_DIR}"

log "Sincronizando código desde ${CODE_GCS_URI} ..."
gsutil -m rsync -r "${CODE_GCS_URI}" .

if [[ ! -f "${MAIN_PY}" ]]; then
  log "No se encontró ${MAIN_PY} tras la descarga."
  ls -la
  exit 1
fi

log "Ejecutando: python ${MAIN_PY} $*"
exec python "${MAIN_PY}" "$@"
