# src/components/hyperparam_tuning/task.py
"""
Tarea del componente de optimización de hiperparámetros para el modelo LSTM.

Responsabilidades:
1.  Utilizar Optuna para buscar la mejor combinación de hiperparámetros.
    - Parámetros de la estrategia de trading (horizonte, take-profit, etc.).
    - Parámetros de la arquitectura del modelo LSTM (capas, dropout, etc.).
    - Parámetros de los indicadores técnicos (longitudes de medias, etc.).
2.  Para cada "trial", entrena un modelo LSTM y lo evalúa con un backtest rápido.
3.  Guarda los mejores parámetros encontrados en un archivo `best_params.json`.
4.  Sube el archivo JSON a una ruta versionada en GCS.
5.  Escribe las métricas de optimización a un artefacto de KFP.

Reemplaza la funcionalidad de `optimize_lstm.py`.
"""

from __future__ import annotations

import argparse
import gc
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

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import callbacks, layers, models, optimizers

# Importar los módulos compartidos
from src.shared import constants, gcs_utils, indicators

# --- Configuración de Logging y Entorno ---
warnings.filterwarnings("ignore", category=FutureWarning)
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
        logger.info("🚀 GPU(s) detectadas y configuradas con memoria dinámica y precisión mixta.")
    else:
        logger.info("ℹ️ No se detectaron GPUs. Se ejecutará en CPU.")
except Exception as e:
    logger.warning(f"⚠️ No se pudo configurar la GPU: {e}")


# --- Funciones de Lógica de Negocio (Modelos y Backtest) ---

def make_model(inp_shape: tuple, lr: float, dr: float, filt: int, units: int, heads: int) -> tf.keras.Model:
    """Crea y compila el modelo Keras para un trial de Optuna."""
    x = inp = layers.Input(shape=inp_shape, dtype=tf.float32)
    x = layers.Conv1D(filt, 3, padding="same", activation="relu")(x)
    x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
    x = layers.MultiHeadAttention(num_heads=heads, key_dim=units)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dr)(x)
    out = layers.Dense(2, dtype="float32")(x)
    model = models.Model(inp, out)
    model.compile(optimizers.Adam(lr), loss="mae")
    return model


def quick_bt(pred, closes, atr, rr, up_thr, dn_thr, delta_min, smooth_win, tick) -> float:
    """Backtest rápido para evaluar el rendimiento de un trial."""
    net, pos, dq = 0.0, False, deque(maxlen=smooth_win)
    entry_price = 0.0
    direction = 0
    sl = 0.0
    tp = 0.0

    for (u, d), price, atr_i in zip(pred, closes, atr):
        mag, diff = max(u, d), abs(u - d)
        raw = 1 if u > d else -1
        cond = ((raw == 1 and mag >= up_thr) or (raw == -1 and mag >= dn_thr)) and diff >= delta_min
        dq.append(raw if cond else 0)
        
        buys, sells = dq.count(1), dq.count(-1)
        
        # ===== CORRECCIÓN DEL ERROR AQUÍ =====
        # Se utiliza la variable correcta 'smooth_win' en lugar de 'swin'.
        signal = 1 if buys > smooth_win // 2 else -1 if sells > smooth_win // 2 else 0

        if not pos and signal != 0:
            pos, entry_price, direction = True, price, signal
            sl = (up_thr if direction == 1 else dn_thr) * atr_i
            tp = rr * sl
            continue
        
        if pos:
            pnl = (price - entry_price) / tick if direction == 1 else (entry_price - price) / tick
            if pnl >= tp or pnl <= -sl:
                net += (tp if pnl >= tp else -sl)
                pos = False
    return net


# --- Orquestación Principal de la Tarea ---

def run_optimization(
    features_path: str,
    pair: str,
    timeframe: str,
    n_trials: int,
    output_gcs_path: str,
    metrics_output_file: Path,
) -> None:
    """
    Orquesta el proceso completo de optimización con Optuna.
    """
    logger.info(f"Iniciando optimización para {pair}/{timeframe} con {n_trials} trials.")
    logger.info(f"Datos de entrada: {features_path}")
    logger.info(f"Archivo de salida de parámetros: {output_gcs_path}")

    try:
        # Cargar los datos una sola vez
        local_features_path = gcs_utils.ensure_gcs_path_and_get_local(features_path)
        df_raw = pd.read_parquet(local_features_path)
        if "timestamp" in df_raw.columns:
            df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="ms", errors="coerce")
        else:
            raise ValueError("La columna 'timestamp' es obligatoria y no se encontró.")

        # Definir la función objetivo para Optuna
        def objective(trial: optuna.trial.Trial) -> float:
            # Limpieza de memoria de la sesión de Keras anterior
            tf.keras.backend.clear_session()
            gc.collect()

            tick = 0.01 if pair.endswith("JPY") else 0.0001
            atr_len = 14
            
            p = {
                "horizon": trial.suggest_int("horizon", 10, 30),
                "rr": trial.suggest_float("rr", 1.5, 3.0),
                "min_thr_up": trial.suggest_float("min_thr_up", 0.5, 2.0),
                "min_thr_dn": trial.suggest_float("min_thr_dn", 0.5, 2.0),
                "delta_min": trial.suggest_float("delta_min", 0.01, 0.5),
                "smooth_win": trial.suggest_int("smooth_win", 1, 5),
                "win": trial.suggest_int("win", 20, 60),
                "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                "dr": trial.suggest_float("dr", 0.1, 0.5),
                "filt": trial.suggest_categorical("filt", [16, 32, 64]),
                "units": trial.suggest_categorical("units", [32, 64, 128]),
                "heads": trial.suggest_categorical("heads", [2, 4, 8]),
                "sma_len": trial.suggest_categorical("sma_len", [20, 40, 60]),
                "rsi_len": trial.suggest_categorical("rsi_len", [7, 14, 21]),
                "macd_fast": trial.suggest_categorical("macd_fast", [8, 12]),
                "macd_slow": trial.suggest_categorical("macd_slow", [21, 26]),
                "stoch_len": trial.suggest_categorical("stoch_len", [14, 21]),
            }

            # Preparación de datos para el trial
            df_ind = indicators.build_indicators(df_raw.copy(), p, atr_len=atr_len)
            
            atr_col = f"atr_{atr_len}"
            if atr_col not in df_ind or df_ind[atr_col].isna().all():
                return -1e9

            # Preparación de etiquetas y máscara
            atr = df_ind[atr_col].values / tick
            close = df_ind.close.values
            future_close = np.roll(close, -p["horizon"])
            future_close[-p["horizon"]:] = np.nan
            diff = (future_close - close) / tick
            
            up = np.maximum(diff, 0) / atr
            dn = np.maximum(-diff, 0) / atr
            
            mask = (~np.isnan(diff)) & (~np.isnan(atr))
            if mask.sum() < 1000: return -1e8

            # Selección de features y escalado
            feature_cols = [c for c in df_ind.columns if c not in {atr_col, "timestamp"}]
            X_raw = df_ind.loc[mask, feature_cols].select_dtypes(include=np.number)
            if X_raw.empty or X_raw.shape[0] <= p["win"]: return -1e8

            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_raw)

            # Creación de secuencias
            X_seq = np.stack([X_scaled[i - p["win"]: i] for i in range(p["win"], len(X_scaled))])
            if X_seq.shape[0] < 500: return -1e8
            
            y_up_seq, y_dn_seq = up[mask][p["win"]:], dn[mask][p["win"]:]
            closes_seq, atr_seq = close[mask][p["win"]:], atr[mask][p["win"]:]
            
            X_tr, X_val, y_up_tr, y_up_val, y_dn_tr, y_dn_val, _, closes_val, _, atr_val = train_test_split(
                X_seq, y_up_seq, y_dn_seq, closes_seq, atr_seq, test_size=0.2, shuffle=False
            )
            
            model = make_model(X_tr.shape[1:], p["lr"], p["dr"], p["filt"], p["units"], p["heads"])
            model.fit(
                X_tr, np.vstack([y_up_tr, y_dn_tr]).T,
                validation_data=(X_val, np.vstack([y_up_val, y_dn_val]).T),
                epochs=15, batch_size=64, verbose=0,
                callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            )
            
            # Evaluación y retorno del score
            score = quick_bt(model.predict(X_val, verbose=0), closes_val, atr_val,
                             p["rr"], p["min_thr_up"], p["min_thr_dn"],
                             p["delta_min"], p["smooth_win"], tick)
            return score

        # Ejecución del estudio de Optuna
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Guardado de los mejores resultados
        best_params = study.best_params
        best_params.update({
            "pair": pair, "timeframe": timeframe, "features_path": features_path,
            "optimization_timestamp_utc": datetime.utcnow().isoformat(),
            "best_trial_score": study.best_value,
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            local_json_path = Path(tmpdir) / "best_params.json"
            with open(local_json_path, "w") as f:
                json.dump(best_params, f, indent=2)
            gcs_utils.upload_gcs_file(local_json_path, output_gcs_path)

        # Escribir métricas para KFP
        kfp_metrics = {
            "metrics": [{
                "name": "optuna-best-trial-score",
                "numberValue": study.best_value if study.best_value is not None else -1e9,
                "format": "RAW"
            }]
        }
        metrics_output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_output_file, 'w') as f:
            json.dump(kfp_metrics, f)

        logger.info(f"✅ Optimización completada. Best score: {study.best_value}")

    except Exception as e:
        logger.critical(f"❌ Fallo crítico en la optimización: {e}", exc_info=True)
        raise

# --- Punto de Entrada para Ejecución como Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task de Optimización de Hiperparámetros para KFP.")
    
    parser.add_argument("--features-path", required=True, help="Ruta GCS al Parquet con features.")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--n-trials", type=int, default=constants.DEFAULT_N_TRIALS)
    parser.add_argument("--best-params-path-output", type=Path, required=True, help="Archivo local para la ruta GCS del JSON de parámetros.")
    parser.add_argument("--optimization-metrics-output", type=Path, required=True, help="Archivo local para las métricas de KFP.")
    
    args = parser.parse_args()

    # Construir la ruta de salida versionada
    timestamp_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    output_gcs_path = (
        f"{constants.LSTM_PARAMS_PATH}/{args.pair}/{args.timeframe}/"
        f"{timestamp_str}/best_params.json"
    )

    run_optimization(
        features_path=args.features_path,
        pair=args.pair,
        timeframe=args.timeframe,
        n_trials=args.n_trials,
        output_gcs_path=output_gcs_path,
        metrics_output_file=args.optimization_metrics_output,
    )

    args.best_params_path_output.parent.mkdir(parents=True, exist_ok=True)
    args.best_params_path_output.write_text(output_gcs_path)