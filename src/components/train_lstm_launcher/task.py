# src/components/train_lstm/task.py
"""
Tarea del componente de entrenamiento del modelo LSTM.

Responsabilidades:
1.  Cargar los mejores hiperparámetros desde un archivo `best_params.json`.
2.  Cargar los datos de features correspondientes.
3.  Preparar los datos para el entrenamiento:
    - Calcular indicadores con los parámetros óptimos.
    - Crear las secuencias de entrenamiento.
    - Escalar los datos.
4.  Construir y entrenar el modelo LSTM final con la configuración óptima.
5.  Realizar un sanity-check con un backtest rápido.
6.  Guardar los artefactos de entrenamiento (modelo .h5, scaler .pkl, y una
    copia de los parámetros .json) en una nueva carpeta versionada por
    timestamp en GCS.

Este script está diseñado para ser ejecutado como un Vertex AI Custom Job.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import tempfile
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import callbacks, layers, models, optimizers

# Importar los módulos compartidos
from src.shared import constants, gcs_utils, indicators

# --- Configuración de Logging y Entorno ---
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Reproducibilidad y Hardware ---
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


# --- Funciones de Lógica de Negocio ---

def to_sequences(
    mat: np.ndarray, up: np.ndarray, dn: np.ndarray, closes: np.ndarray, win: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convierte datos tabulares en secuencias para el LSTM."""
    X, y_up, y_dn, cl = [], [], [], []
    for i in range(win, len(mat)):
        X.append(mat[i - win : i])
        y_up.append(up[i])
        y_dn.append(dn[i])
        cl.append(closes[i])
    return (
        np.asarray(X, np.float32), np.asarray(y_up, np.float32),
        np.asarray(y_dn, np.float32), np.asarray(cl, np.float32),
    )

def make_model(
    inp_shape: Tuple[int, int], lr: float, dr: float, filt: int, units: int, heads: int
) -> tf.keras.Model:
    """Crea y compila el modelo Keras con los hiperparámetros finales."""
    inp = layers.Input(shape=inp_shape, dtype=tf.float32)
    x = layers.Conv1D(filt, 3, padding="same", activation="relu")(inp)
    x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
    x = layers.MultiHeadAttention(num_heads=heads, key_dim=units)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dr)(x)
    out = layers.Dense(2, dtype="float32")(x)
    model = models.Model(inp, out)
    model.compile(optimizers.Adam(lr), loss="mae")
    return model

def quick_bt(pred, closes, atr_pips, hp: dict, tick: float) -> float:
    """Back-test mínimo para sanity-check (sin comisiones)."""
    rr, up_thr, dn_thr = hp["rr"], hp["min_thr_up"], hp["min_thr_dn"]
    delta_min, swin = hp["delta_min"], hp["smooth_win"]
    net, pos, dq = 0.0, False, deque(maxlen=swin)
    entry, sl, tp, direction = 0.0, 0.0, 0.0, 0

    for (u, d), price, atr in zip(pred, closes, atr_pips):
        mag, diff_val = max(u, d), abs(u - d)
        raw = 1 if u > d else -1
        thr = up_thr if raw == 1 else dn_thr
        dq.append(raw if (mag >= thr and diff_val >= delta_min) else 0)
        buys, sells = dq.count(1), dq.count(-1)
        sig = 1 if buys > swin // 2 else -1 if sells > swin // 2 else 0
        if not pos and sig:
            pos, entry, sl, tp, direction = True, price, thr * atr, rr * (thr * atr), sig
            continue
        if pos:
            pnl = ((price - entry) if direction == 1 else (entry - price)) / tick
            if pnl >= tp or pnl <= -sl:
                net += (tp if pnl >= tp else -sl)
                pos = False
    return net

# --- Orquestación Principal de la Tarea ---

def run_training(
    params_path: str,
    output_gcs_base_dir: str,
    pair: str,
    timeframe: str,
) -> None:
    """Orquesta el proceso completo de entrenamiento del modelo LSTM."""
    try:
        # 1. Cargar hiperparámetros
        logger.info(f"Cargando hiperparámetros desde: {params_path}")
        local_params_path = gcs_utils.ensure_gcs_path_and_get_local(params_path)
        with open(local_params_path) as f:
            hp = json.load(f)

        # 2. Cargar datos de features
        features_gcs_path = hp["features_path"]
        logger.info(f"Cargando datos de features desde: {features_gcs_path}")
        local_features_path = gcs_utils.ensure_gcs_path_and_get_local(features_gcs_path)
        df_raw = pd.read_parquet(local_features_path)
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], errors="coerce")

        # 3. Preparar datos
        logger.info("Preparando datos para el entrenamiento...")
        tick = 0.01 if pair.endswith("JPY") else 0.0001
        atr_len = 14
        df_ind = indicators.build_indicators(df_raw, hp, atr_len=atr_len)
        
        close = df_ind.close.values
        atr_pips = df_ind[f"atr_{atr_len}"].values / tick
        horizon = int(hp["horizon"])
        future_close = np.roll(close, -horizon)
        future_close[-horizon:] = np.nan
        diff = (future_close - close) / tick

        up = np.maximum(diff, 0) / atr_pips
        dn = np.maximum(-diff, 0) / atr_pips
        mask = (~np.isnan(diff)) & (~np.isnan(atr_pips))

        win = int(hp["win"])
        if mask.sum() <= win:
            raise ValueError(f"No hay suficientes datos ({mask.sum()}) para el tamaño de ventana ({win}).")

        feature_cols: List[str] = [c for c in df_ind.columns if c not in {f"atr_{atr_len}", "timestamp"}]
        X_raw = df_ind.loc[mask, feature_cols]

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_raw)

        X_seq, y_up_seq, y_dn_seq, closes_seq = to_sequences(
            X_scaled, up[mask], dn[mask], close[mask], win
        )
        logger.info(f"✅ Datos listos: X={X_seq.shape}, y=({y_up_seq.shape}, {y_dn_seq.shape})")
        
        # 4. Entrenar modelo
        logger.info("Iniciando entrenamiento del modelo LSTM...")
        model = make_model(X_seq.shape[1:], hp["lr"], hp["dr"], hp["filt"], hp["units"], hp["heads"])
        early_stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1)
        
        model.fit(
            X_seq, np.vstack([y_up_seq, y_dn_seq]).T,
            epochs=60, batch_size=128, callbacks=[early_stop], verbose=1
        )
        logger.info("✅ Entrenamiento finalizado.")
        
        # 5. Sanity-check
        bt_score = quick_bt(model.predict(X_seq, verbose=0), closes_seq, atr_pips[win:], hp, tick)
        logger.info(f"⚡ Quick BT (sanity check): {bt_score:.2f} ATR-pips")

        # 6. Guardar artefactos en GCS
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        final_gcs_dir = f"{output_gcs_base_dir.rstrip('/')}/{pair}/{timeframe}/{timestamp}/"
        logger.info(f"Guardando artefactos en: {final_gcs_dir}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_path = tmp_path / "model.h5"
            scaler_path = tmp_path / "scaler.pkl"
            params_out_path = tmp_path / "params.json" # Guardamos una copia de los params usados

            model.save(model_path)
            joblib.dump(scaler, scaler_path)
            with open(params_out_path, "w") as f:
                json.dump(hp, f, indent=2)

            gcs_utils.upload_gcs_file(model_path, final_gcs_dir + model_path.name)
            gcs_utils.upload_gcs_file(scaler_path, final_gcs_dir + scaler_path.name)
            gcs_utils.upload_gcs_file(params_out_path, final_gcs_dir + params_out_path.name)
        
        logger.info(f"🎉 Tarea completada. Artefactos disponibles en: {final_gcs_dir}")

    except Exception as e:
        logger.critical(f"❌ Fallo crítico en el entrenamiento LSTM: {e}", exc_info=True)
        raise

# --- Punto de Entrada para Ejecución como Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task de Entrenamiento LSTM para KFP Custom Job.")
    
    # Argumentos que el Vertex AI Custom Job le pasará
    parser.add_argument("--params", required=True, help="Ruta GCS al JSON de hiperparámetros.")
    parser.add_argument("--output-gcs-base-dir", default=constants.LSTM_MODELS_PATH)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    
    # Argumentos que Vertex puede inyectar y que podemos ignorar de forma segura
    parser.add_argument("--project-id", help=argparse.SUPPRESS)
    parser.add_argument("--gcs-bucket-name", help=argparse.SUPPRESS)
    
    args = parser.parse_args()

    run_training(
        params_path=args.params,
        output_gcs_base_dir=args.output_gcs_base_dir,
        pair=args.pair,
        timeframe=args.timeframe,
    )