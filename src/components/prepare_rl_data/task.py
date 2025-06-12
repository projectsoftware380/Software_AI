# src/components/prepare_rl_data/task.py
"""
Prepara los datos que el agente PPO usará para filtrar las señales del LSTM.
1. Localiza y descarga (modelo, escalador, params) entrenados por el LSTM.
2. Construye indicadores, escala y crea secuencias.
3. Obtiene predicciones y embeddings LSTM.
4. Genera la señal 'raw_signal'.
5. Guarda `obs`, `raw_signal` y `closes` en un .npz y lo sube a GCS.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from google.cloud import storage
from tensorflow.keras import models  # pylint: disable=import-error

from src.shared import constants, gcs_utils, indicators

# ───────────────────────────── logging ─────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
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
except Exception as exc:  # pragma: no cover
    logger.warning("⚠️ No se pudo configurar la GPU: %s", exc)

# ───────────────────────── helpers utilitarios ─────────────────────


def resolve_artifact_dir(base_dir: str) -> str:
    """
    Si `base_dir` no contiene directamente `model.h5`, busca
    recursivamente un nivel por debajo y devuelve la primera carpeta
    que sí lo contenga.  Lanza FileNotFoundError si no lo encuentra.
    """
    client = storage.Client()
    bucket_name, prefix = base_dir.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)

    # ¿ya está completo?
    if bucket.blob(f"{prefix.rstrip('/')}/model.h5").exists():
        return base_dir.rstrip("/")

    logger.info("🔎 Buscando model.h5 dentro de %s", base_dir)
    # Listamos blobs para detectar subcarpetas
    subdirs: set[str] = set()
    for blob in bucket.list_blobs(prefix=prefix.rstrip("/") + "/"):
        pp = PurePosixPath(blob.name)
        if pp.name == "model.h5":
            logger.info("✔ model.h5 hallado en %s", pp.parent.as_posix())
            return f"gs://{bucket_name}/{pp.parent.as_posix()}"
        if len(pp.parts) >= len(PurePosixPath(prefix).parts) + 2:
            subdirs.add("/".join(pp.parts[:len(PurePosixPath(prefix).parts) + 1]))

    raise FileNotFoundError(
        f"No se encontró model.h5 en {base_dir} ni en sus subdirectorios: {subdirs}"
    )


def make_sequences(arr: np.ndarray, win: int) -> np.ndarray:
    """Convierte array 2-D en una stack 3-D de ventanas deslizantes."""
    if len(arr) <= win:
        return np.empty((0, win, arr.shape[1]), dtype=np.float32)
    return np.stack([arr[i - win: i] for i in range(win, len(arr))]).astype(np.float32)

# ───────────────────────── función principal ───────────────────────


def run_rl_data_preparation(
    lstm_model_dir: str,
    pair: str,
    timeframe: str,
    output_gcs_base_dir: str,
) -> str:
    """
    Devuelve la URI GCS donde se cargó el `.npz` con:
        obs, raw_signal, closes
    """
    logger.info("➡️  Preparación RL - input dir: %s", lstm_model_dir)
    lstm_model_dir = resolve_artifact_dir(lstm_model_dir)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # ── artefactos LSTM ─────────────────────────────────────
            model_path = gcs_utils.download_gcs_file(f"{lstm_model_dir}/model.h5", tmp)
            scaler_path = gcs_utils.download_gcs_file(f"{lstm_model_dir}/scaler.pkl", tmp)
            params_path = gcs_utils.download_gcs_file(f"{lstm_model_dir}/params.json", tmp)

            lstm_model = models.load_model(model_path, compile=False)
            scaler = joblib.load(scaler_path)
            hp: dict = json.loads(params_path.read_text())

        emb_model = models.Model(lstm_model.input, lstm_model.layers[-2].output)
        logger.info("✔ Artefactos LSTM cargados y modelo de embeddings listo.")

        # ── datos crudos OHLC ───────────────────────────────────────
        raw_data_path = f"{constants.DATA_PATH}/{pair}/{timeframe}/{pair}_{timeframe}.parquet"
        local_raw_path = gcs_utils.ensure_gcs_path_and_get_local(raw_data_path)

        if not Path(local_raw_path).exists():
            raise FileNotFoundError(f"Parquet no encontrado: {raw_data_path}")

        df_raw = pd.read_parquet(local_raw_path)
        if df_raw.empty:
            raise ValueError("Parquet vacío; no hay datos para preparar RL.")

        df_ind = indicators.build_indicators(df_raw, hp, atr_len=14)
        if df_ind.isna().any().any():
            raise ValueError("Persisten NaNs tras calcular indicadores")

        logger.info("✔ Indicadores calculados (%s filas).", len(df_ind))

        # ── escala y secuencias ────────────────────────────────────
        try:
            feature_cols = list(scaler.feature_names_in_)  # scikit-learn ≥ 1.0
        except AttributeError:  # compat con versiones <1.0
            feature_cols = list(
                df_ind.select_dtypes(include="number").columns.difference(["timestamp"])
            )
            logger.warning("Scaler sin feature_names_in_; usando columnas numéricas detectadas.")

        X_scaled = scaler.transform(df_ind[feature_cols])
        X_seq = make_sequences(X_scaled, win=hp["win"])
        if X_seq.shape[0] == 0:
            raise ValueError("Secuencias vacías; revisa parámetro win/hp.")

        logger.info("✔ Secuencias generadas: %s", X_seq.shape)

        # ── predicciones y embeddings ──────────────────────────────
        with tf.device("/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"):
            preds = lstm_model.predict(X_seq, verbose=0).astype(np.float32)
            embs = emb_model.predict(X_seq, verbose=0).astype(np.float32)

        if preds.shape[0] != embs.shape[0]:
            raise ValueError("Predicciones y embeddings con longitudes distintas.")

        # ── construcción de OBS ────────────────────────────────────
        OBS = np.hstack([preds, embs]).astype(np.float32)
        logger.info("✔ OBS creado: %s", OBS.shape)

        # ── señal raw_signal ───────────────────────────────────────
        pred_up, pred_dn = preds[:, 0], preds[:, 1]
        raw_signal = np.zeros(len(pred_up), dtype=np.int8)
        dq: deque[int] = deque(maxlen=hp["smooth_win"])

        for i, (u, d) in enumerate(zip(pred_up, pred_dn)):
            mag, diff = max(u, d), abs(u - d)
            raw_dir = 1 if u > d else -1
            cond = (
                ((raw_dir == 1 and mag >= hp["min_thr_up"]) or (raw_dir == -1 and mag >= hp["min_thr_dn"]))
                and diff >= hp["delta_min"]
            )
            dq.append(raw_dir if cond else 0)
            buys, sells = dq.count(1), dq.count(-1)
            raw_signal[i] = 1 if buys > hp["smooth_win"] // 2 else -1 if sells > hp["smooth_win"] // 2 else 0

        closes = df_ind.close.values[hp["win"]:].astype(np.float32)

        # ── subida del .npz a GCS ──────────────────────────────────
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        final_uri = (
            f"{output_gcs_base_dir.rstrip('/')}/{pair}/{timeframe}/{ts}/ppo_input_data.npz"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            local_npz = Path(tmpdir) / "ppo_input_data.npz"
            np.savez(local_npz, obs=OBS, raw=raw_signal, closes=closes)
            gcs_utils.upload_gcs_file(local_npz, final_uri)

        logger.info("🎉 Datos RL preparados y subidos a %s", final_uri)
        return final_uri

    except Exception as exc:  # pragma: no cover
        logger.critical("❌ Fallo en preparación RL: %s", exc, exc_info=True)
        raise


# ───────────────────────── CLI / entrypoint ─────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Task de Preparación de Datos para RL")
    parser.add_argument("--lstm-model-dir", required=True,
                        help="Ruta GCS al directorio base del modelo LSTM entrenado.")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--output-gcs-base-dir", default=constants.RL_DATA_INPUTS_PATH)
    parser.add_argument("--rl-data-path-output", type=Path, required=True,
                        help="Archivo local para escribir la ruta GCS final del .npz")

    args = parser.parse_args()

    final_output_path = run_rl_data_preparation(
        lstm_model_dir=args.lstm_model_dir,
        pair=args.pair,
        timeframe=args.timeframe,
        output_gcs_base_dir=args.output_gcs_base_dir,
    )

    args.rl_data_path_output.parent.mkdir(parents=True, exist_ok=True)
    args.rl_data_path_output.write_text(final_output_path)
