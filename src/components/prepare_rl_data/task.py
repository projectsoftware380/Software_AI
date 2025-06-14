# src/components/prepare_rl_data/task.py
"""
Prepara los datos que el agente PPO usará para filtrar las señales del LSTM.

Flujo Corregido:
1. Ignora la ruta del modelo de entrada (que puede ser incorrecta).
2. Construye la ruta correcta del directorio de modelos para el par/timeframe actual.
3. Encuentra el modelo más reciente (último timestamp) en ese directorio.
4. Localiza y descarga (modelo, escalador, params) del directorio correcto.
5. Construye indicadores, escala y crea secuencias.
6. Obtiene predicciones y embeddings LSTM.
7. Genera la señal 'raw_signal'.
8. Guarda `obs`, `raw_signal` y `closes` en un .npz y lo sube a GCS.
9. Limpia las versiones antiguas de los datos de RL para el par/timeframe actual.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque

import gcsfs
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from google.cloud import storage
from tensorflow.keras import models

from src.shared import constants, gcs_utils, indicators

# ───────────────────────────── logging ─────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ───────────────────────── configuración global ─────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    if gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("🚀 GPU(s) habilitadas: %s", [g.name for g in gpus])
    else:
        logger.info("ℹ️ No se detectaron GPUs; se usará CPU.")
except Exception as exc:
    logger.warning("⚠️ No se pudo configurar la GPU: %s", exc)

# ───────────────────────── helpers utilitarios ─────────────────────

def _keep_only_latest_version(base_gcs_prefix: str) -> None:
    """
    Mantiene sólo el sub-directorio con timestamp (YYYYMMDDHHMMSS)
    más reciente y borra el resto.
    """
    try:
        fs = gcsfs.GCSFileSystem(project=constants.PROJECT_ID)
        if not base_gcs_prefix.endswith("/"):
            base_gcs_prefix += "/"

        ts_re = re.compile(r"/(\d{14})/?$")
        dirs = [p for p in fs.ls(base_gcs_prefix) if fs.isdir(p) and ts_re.search(p)]

        if len(dirs) <= 1:
            return

        dirs.sort(key=lambda p: ts_re.search(p).group(1), reverse=True)
        for old in dirs[1:]:
            logger.info("🗑️  Borrando versión de datos RL antigua: gs://%s", old)
            fs.rm(old, recursive=True)
    except Exception as exc:
        logger.warning("No se pudo limpiar versiones antiguas de datos RL: %s", exc)


def find_latest_model_dir(base_path: str) -> str:
    """
    Busca dentro de un directorio base en GCS (ej. .../EURUSD/15minute/)
    y devuelve la subcarpeta con el timestamp más reciente.
    """
    client = storage.Client(project=constants.PROJECT_ID)
    bucket_name, prefix = base_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)

    # Usamos el delimitador para listar solo los directorios de primer nivel
    blobs = bucket.list_blobs(prefix=f"{prefix.rstrip('/')}/", delimiter="/")
    
    # El iterador devuelve los directorios en 'prefixes'
    dirs = [p for p in blobs.prefixes]

    if not dirs:
        raise FileNotFoundError(f"No se encontraron directorios de modelo versionados en {base_path}")

    # Ordenar los directorios (que son strings de rutas) y devolver el último
    latest_dir = sorted(dirs, reverse=True)[0]
    return f"gs://{bucket_name}/{latest_dir.rstrip('/')}"

def make_sequences(arr: np.ndarray, win: int) -> np.ndarray:
    """Convierte array 2-D en una pila 3-D de ventanas deslizantes."""
    if len(arr) <= win:
        return np.empty((0, win, arr.shape[1]), dtype=np.float32)
    return (
        np.stack([arr[i - win : i] for i in range(win, len(arr))])
        .astype(np.float32)
    )

# ───────────────────────── función principal ───────────────────────
def run_rl_data_preparation(
    lstm_model_dir: str,  # Este parámetro se ignora, pero se mantiene por compatibilidad
    pair: str,
    timeframe: str,
    output_gcs_base_dir: str,
) -> str:
    """
    Devuelve la URI GCS donde se cargó el `.npz` con:
        obs, raw_signal, closes
    """
    try:
        # --- CORRECCIÓN: CONSTRUIR LA RUTA CORRECTA E IGNORAR LA DE ENTRADA ---
        actual_model_parent_dir = f"{constants.LSTM_MODELS_PATH}/{pair}/{timeframe}"
        logger.info("Buscando el modelo más reciente en la ruta correcta: %s", actual_model_parent_dir)
        
        # Encontrar el directorio del modelo con el último timestamp
        correct_lstm_model_dir = find_latest_model_dir(actual_model_parent_dir)
        logger.info("✅ Directorio de modelo más reciente encontrado: %s", correct_lstm_model_dir)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Descargar artefactos desde el directorio correcto
            model_path = gcs_utils.download_gcs_file(f"{correct_lstm_model_dir}/model.keras", tmp)
            scaler_path = gcs_utils.download_gcs_file(f"{correct_lstm_model_dir}/scaler.pkl", tmp)
            params_path = gcs_utils.download_gcs_file(f"{correct_lstm_model_dir}/params.json", tmp)

            lstm_model = models.load_model(model_path, compile=False)
            scaler = joblib.load(scaler_path)
            hp: dict = json.loads(params_path.read_text())

        emb_model = models.Model(lstm_model.input, lstm_model.layers[-2].output)
        logger.info("✔ Artefactos LSTM cargados y modelo de embeddings listo.")

        # ── datos crudos OHLC ───────────────────────────────────────
        raw_data_path = f"{constants.DATA_PATH}/{pair}/{timeframe}/{pair}_{timeframe}.parquet"
        local_raw_path = gcs_utils.ensure_gcs_path_and_get_local(raw_data_path)

        df_raw = pd.read_parquet(local_raw_path)
        if df_raw.empty:
            raise ValueError("Parquet vacío; no hay datos para preparar RL.")

        df_ind = indicators.build_indicators(df_raw, hp, atr_len=14)
        if df_ind.isna().any().any():
            raise ValueError("Persisten NaNs tras calcular indicadores")

        logger.info("✔ Indicadores calculados (%s filas).", len(df_ind))

        # ── escala y secuencias ────────────────────────────────────
        feature_cols = list(scaler.feature_names_in_)
        X_scaled = scaler.transform(df_ind[feature_cols])
        X_seq = make_sequences(X_scaled, win=hp["win"])
        if X_seq.shape[0] == 0:
            raise ValueError("Secuencias vacías; revisa parámetro win/hp.")

        logger.info("✔ Secuencias generadas: %s", X_seq.shape)

        # ── predicciones y embeddings ──────────────────────────────
        with tf.device("/GPU:0" if gpus else "/CPU:0"):
            preds = lstm_model.predict(X_seq, verbose=0).astype(np.float32)
            embs = emb_model.predict(X_seq, verbose=0).astype(np.float32)

        OBS = np.hstack([preds, embs]).astype(np.float32)
        logger.info("✔ OBS creado: %s", OBS.shape)

        # ── señal raw_signal ───────────────────────────────────────
        pred_up, pred_dn = preds[:, 0], preds[:, 1]
        raw_signal = np.zeros(len(pred_up), dtype=np.int8)
        dq: Deque[int] = deque(maxlen=hp["smooth_win"])

        for i, (u, d) in enumerate(zip(pred_up, pred_dn)):
            mag, diff = max(u, d), abs(u - d)
            raw_dir = 1 if u > d else -1
            cond = ((raw_dir == 1 and mag >= hp["min_thr_up"]) or (raw_dir == -1 and mag >= hp["min_thr_dn"])) and diff >= hp["delta_min"]
            dq.append(raw_dir if cond else 0)
            buys, sells = dq.count(1), dq.count(-1)
            raw_signal[i] = 1 if buys > hp["smooth_win"] // 2 else -1 if sells > hp["smooth_win"] // 2 else 0

        closes = df_ind.close.values[hp["win"] :].astype(np.float32)

        # ── subida del .npz a GCS ──────────────────────────────────
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        final_uri = f"{output_gcs_base_dir.rstrip('/')}/{pair}/{timeframe}/{ts}/ppo_input_data.npz"

        with tempfile.TemporaryDirectory() as tmpdir:
            local_npz = Path(tmpdir) / "ppo_input_data.npz"
            np.savez(local_npz, obs=OBS, raw=raw_signal, closes=closes)
            gcs_utils.upload_gcs_file(local_npz, final_uri)

        logger.info("🎉 Datos RL preparados y subidos a %s", final_uri)
        
        # --- LLAMADA A LA LIMPIEZA AÑADIDA ---
        base_cleanup_path = f"{output_gcs_base_dir.rstrip('/')}/{pair}/{timeframe}"
        _keep_only_latest_version(base_cleanup_path)
        
        return final_uri

    except Exception as exc:
        logger.critical("❌ Fallo en preparación RL: %s", exc, exc_info=True)
        raise

# ───────────────────────── CLI / entrypoint ─────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Task de Preparación de Datos para RL")
    parser.add_argument("--lstm-model-dir", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--output-gcs-base-dir", default=constants.RL_DATA_INPUTS_PATH)
    parser.add_argument("--rl-data-path-output", type=Path, required=True)
    args = parser.parse_args()

    final_output_path = run_rl_data_preparation(
        lstm_model_dir=args.lstm_model_dir,
        pair=args.pair,
        timeframe=args.timeframe,
        output_gcs_base_dir=args.output_gcs_base_dir,
    )

    args.rl_data_path_output.parent.mkdir(parents=True, exist_ok=True)
    args.rl_data_path_output.write_text(final_output_path)