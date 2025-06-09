# src/components/prepare_rl_data/task.py
"""
Tarea del componente de preparación de datos para el agente de RL.

Responsabilidades:
1.  Cargar los artefactos del modelo LSTM entrenado (modelo, escalador y
    parámetros) desde una ruta versionada en GCS.
2.  Cargar los datos OHLC crudos.
3.  Procesar los datos usando la misma lógica que en el entrenamiento del LSTM:
    - Calcular indicadores con los hiperparámetros del modelo cargado.
    - Escalar los datos con el escalador (`scaler.pkl`) cargado.
    - Crear secuencias.
4.  Usar el modelo LSTM para generar predicciones (`up`, `dn`).
5.  Extraer los 'embeddings' de la penúltima capa del LSTM para cada secuencia.
6.  Crear el array de observaciones (`obs`) para el agente RL, combinando
    predicciones y embeddings.
7.  Generar una señal de trading base (`raw_signal`) a partir de las
    predicciones del LSTM.
8.  Guardar los arrays (`obs`, `raw_signal`, `closes`) en un archivo .npz
    y subirlo a una nueva ruta versionada en GCS.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models

from src.shared import constants, gcs_utils, indicators

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("🚀 GPU(s) detectadas y configuradas.")
    else:
        logger.info("ℹ️ No se detectaron GPUs. Se ejecutará en CPU.")
except Exception as e:
    logger.warning(f"⚠️ No se pudo configurar la GPU: {e}")

def make_sequences(arr: np.ndarray, win: int) -> np.ndarray:
    if len(arr) <= win:
        return np.empty((0, win, arr.shape[1]), dtype=np.float32)
    return np.stack([arr[i - win : i] for i in range(win, len(arr))]).astype(np.float32)

def run_rl_data_preparation(
    lstm_model_dir: str, pair: str, timeframe: str, output_gcs_base_dir: str
) -> str:
    try:
        logger.info(f"Cargando artefactos LSTM desde: {lstm_model_dir}")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_path = gcs_utils.download_gcs_file(f"{lstm_model_dir}/model.h5", tmp_path)
            scaler_path = gcs_utils.download_gcs_file(f"{lstm_model_dir}/scaler.pkl", tmp_path)
            params_path = gcs_utils.download_gcs_file(f"{lstm_model_dir}/params.json", tmp_path)

            lstm_model = models.load_model(model_path, compile=False)
            scaler = joblib.load(scaler_path)
            hp = json.loads(params_path.read_text())

        emb_model = models.Model(lstm_model.input, lstm_model.layers[-2].output)
        logger.info("✔ Artefactos LSTM y modelo de embedding cargados.")

        raw_data_path = f"{constants.DATA_PATH}/{pair}/{timeframe}/{pair}_{timeframe}.parquet"
        logger.info(f"Cargando datos crudos desde: {raw_data_path}")
        local_raw_path = gcs_utils.ensure_gcs_path_and_get_local(raw_data_path)
        df_raw = pd.read_parquet(local_raw_path)
        df_ind = indicators.build_indicators(df_raw, hp, atr_len=14)
        logger.info("✔ Indicadores calculados.")

        feature_cols = scaler.feature_names_in_
        X_scaled = scaler.transform(df_ind[feature_cols])
        X_seq = make_sequences(X_scaled, win=hp["win"])

        if X_seq.shape[0] == 0:
            raise ValueError("No se pudieron generar secuencias, el DataFrame resultante es muy corto.")
        logger.info(f"✔ Secuencias generadas con shape: {X_seq.shape}")
        
        logger.info("Generando predicciones y embeddings con el modelo LSTM...")
        with tf.device("/GPU:0" if gpus else "/CPU:0"):
            preds = lstm_model.predict(X_seq, verbose=0).astype(np.float32)
            embs = emb_model.predict(X_seq, verbose=0).astype(np.float32)

        OBS = np.hstack([preds, embs]).astype(np.float32)
        logger.info(f"✔ Array de observaciones (OBS) creado con shape: {OBS.shape}")

        from collections import deque
        raw_signal = np.zeros(len(X_seq), dtype=np.int8)
        dq = deque(maxlen=hp["smooth_win"])
        pred_up, pred_dn = preds[:, 0], preds[:, 1]
        
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
        
        logger.info("✔ Señal de trading base (raw_signal) generada.")

        closes = df_ind.close.values[hp["win"] :].astype(np.float32)
        timestamp_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        output_gcs_path = (
            f"{output_gcs_base_dir.rstrip('/')}/{pair}/{timeframe}/"
            f"{timestamp_str}/ppo_input_data.npz"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            local_npz_path = Path(tmpdir) / "ppo_input_data.npz"
            np.savez(local_npz_path, obs=OBS, raw=raw_signal.astype(np.int8), closes=closes)
            gcs_utils.upload_gcs_file(local_npz_path, output_gcs_path)
        
        logger.info(f"🎉 Tarea completada. Datos para RL disponibles en: {output_gcs_path}")
        return output_gcs_path

    except Exception as e:
        logger.critical(f"❌ Fallo crítico en la preparación de datos para RL: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task de Preparación de Datos para RL.")
    parser.add_argument("--lstm-model-dir", required=True, help="Ruta GCS al directorio versionado del modelo LSTM.")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--output-gcs-base-dir", default=constants.RL_DATA_INPUTS_PATH)
    parser.add_argument("--rl-data-path-output", type=Path, required=True, help="Archivo local para escribir la ruta de salida GCS del .npz")
    
    args = parser.parse_args()

    final_output_path = run_rl_data_preparation(
        lstm_model_dir=args.lstm_model_dir,
        pair=args.pair,
        timeframe=args.timeframe,
        output_gcs_base_dir=args.output_gcs_base_dir,
    )

    args.rl_data_path_output.parent.mkdir(parents=True, exist_ok=True)
    args.rl_data_path_output.write_text(final_output_path)