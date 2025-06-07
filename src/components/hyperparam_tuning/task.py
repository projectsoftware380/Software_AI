# src/components/hyperparam_tuning/task.py (CORREGIDO)
"""
Tarea del componente de optimización de hiperparámetros para el modelo LSTM.

Responsabilidades:
1.  Cargar los datos de features preparados para la optimización.
2.  Definir un espacio de búsqueda de hiperparámetros para Optuna.
3.  Definir una función 'objective' que entrene un modelo LSTM y lo evalúe
    con un backtest rápido para cada trial de Optuna.
4.  Ejecutar el estudio de Optuna para encontrar la mejor combinación de
    hiperparámetros.
5.  Guardar los mejores parámetros encontrados en un archivo `best_params.json`
    en una nueva carpeta versionada en GCS.
6.  Imprimir la ruta GCS del archivo JSON para que el siguiente componente
    de la pipeline pueda consumirlo.
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

import joblib
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import layers, models, optimizers

# Módulos compartidos
from src.shared import constants, gcs_utils, indicators

# --- Configuración de Logging y Entorno ---
warnings.filterwarnings("ignore", category=UserWarning, module="optuna")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# --- Reproducibilidad ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

# --- Variables Globales para el Estudio de Optuna ---
# Se cargan una vez para evitar I/O repetitivo en cada trial
DF_FEATURES = None
DF_INDICATORS = None


def quick_backtest(predictions: np.ndarray, closes: np.ndarray, atr_pips: np.ndarray, hp: dict, tick: float) -> float:
    """
    Ejecuta un backtest simplificado para evaluar el rendimiento de un conjunto de hiperparámetros.
    Retorna el Profit Factor, un buen indicador de la rentabilidad general.
    """
    rr = hp["rr"]
    up_thr, dn_thr = hp["min_thr_up"], hp["min_thr_dn"]
    delta_min, swin = hp["delta_min"], hp["smooth_win"]

    total_wins, total_losses = 1e-9, 1e-9  # Evitar división por cero
    in_position = False
    entry_price = 0.0
    direction = 0

    dq = deque(maxlen=swin)

    for i, (pred, price, atr) in enumerate(zip(predictions, closes, atr_pips)):
        u, d = pred[0], pred[1]
        mag, diff = max(u, d), abs(u - d)
        raw_signal = 1 if u > d else -1
        threshold = up_thr if raw_signal == 1 else dn_thr

        cond = mag >= threshold and diff >= delta_min
        dq.append(raw_signal if cond else 0)
        
        buys, sells = dq.count(1), dq.count(-1)
        signal = 1 if buys > swin // 2 else -1 if sells > swin // 2 else 0

        if in_position:
            pnl_pips = (price - entry_price) / tick * direction
            stop_loss_pips = threshold * atr
            take_profit_pips = stop_loss_pips * rr

            if pnl_pips <= -stop_loss_pips or pnl_pips >= take_profit_pips:
                final_pnl = take_profit_pips if pnl_pips > 0 else -stop_loss_pips
                if final_pnl > 0:
                    total_wins += final_pnl
                else:
                    total_losses += abs(final_pnl)
                in_position = False
        
        if not in_position and signal != 0:
            in_position = True
            entry_price = price
            direction = signal
            
    return total_wins / total_losses


def objective(trial: optuna.trial.Trial) -> float:
    """
    Función objetivo que Optuna intentará maximizar.
    """
    global DF_FEATURES, DF_INDICATORS
    
    try:
        # 1. Definir el espacio de búsqueda de hiperparámetros
        hp = {
            # Hiperparámetros de indicadores
            "sma_len": trial.suggest_int("sma_len", 20, 200, step=10),
            "rsi_len": trial.suggest_int("rsi_len", 7, 30),
            "macd_fast": trial.suggest_int("macd_fast", 5, 20),
            "macd_slow": trial.suggest_int("macd_slow", 21, 50),
            "stoch_len": trial.suggest_int("stoch_len", 7, 21),
            # Hiperparámetros de la lógica de trading
            "win": trial.suggest_int("win", 10, 60),
            "horizon": trial.suggest_int("horizon", 2, 10),
            "rr": trial.suggest_float("rr", 1.0, 3.0),
            "delta_min": trial.suggest_float("delta_min", 0.0, 0.2),
            "min_thr_up": trial.suggest_float("min_thr_up", 0.2, 1.5),
            "min_thr_dn": trial.suggest_float("min_thr_dn", 0.2, 1.5),
            "smooth_win": trial.suggest_int("smooth_win", 1, 5),
            # Hiperparámetros del modelo
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "dr": trial.suggest_float("dr", 0.1, 0.5),
            "units": trial.suggest_categorical("units", [32, 64, 128]),
            "filt": trial.suggest_categorical("filt", [16, 32, 64]),
            "heads": trial.suggest_categorical("heads", [2, 4, 8]),
        }

        # 2. Preparar los datos con los HPs del trial actual
        if DF_INDICATORS is None:
             DF_INDICATORS = indicators.build_indicators(DF_FEATURES, hp, drop_na=True)

        close, tick = DF_INDICATORS.close.values, 0.0001
        atr_pips = DF_INDICATORS["atr_14"].values / tick
        future_close = np.roll(close, -hp["horizon"])
        future_close[-hp["horizon"]:] = np.nan
        diff = (future_close - close) / tick

        y_up = np.maximum(diff, 0) / atr_pips
        y_dn = np.maximum(-diff, 0) / atr_pips
        mask = ~np.isnan(diff) & ~np.isnan(atr_pips)

        feature_cols = [c for c in DF_INDICATORS.columns if c not in {"atr_14", "timestamp"}]
        X_scaled = RobustScaler().fit_transform(DF_INDICATORS.loc[mask, feature_cols])

        X, y = np.stack([X_scaled[i - hp["win"]: i] for i in range(hp["win"], len(X_scaled))]), np.vstack([y_up[mask][hp["win"]:], y_dn[mask][hp["win"]:]]).T

        # 3. Construir y entrenar un modelo simple
        inp = layers.Input(shape=X.shape[1:])
        x = layers.LSTM(hp["units"])(inp)
        out = layers.Dense(2, dtype="float32")(x)
        model = models.Model(inp, out)
        model.compile(optimizer=optimizers.Adam(hp["lr"]), loss="mae")
        model.fit(X, y, epochs=1, batch_size=256, verbose=0)
        
        # 4. Evaluar y retornar score
        preds = model.predict(X, verbose=0)
        closes_seq = close[mask][hp["win"]:]
        atr_seq = atr_pips[mask][hp["win"]:]
        
        score = quick_backtest(preds, closes_seq, atr_seq, hp, tick)
        
        # Limpieza de memoria
        tf.keras.backend.clear_session()
        del model
        
        return score

    except Exception as e:
        logger.warning(f"Trial fallido: {e}")
        return -1.0 # Retornar un score muy bajo si el trial falla


def run_optimization(features_path: str, pair: str, timeframe: str, n_trials: int, output_gcs_path: str) -> str:
    """
    Orquesta el proceso de optimización.
    """
    global DF_FEATURES
    logger.info(f"Iniciando optimización para {pair}/{timeframe} con {n_trials} trials.")
    logger.info(f"Datos de entrada: {features_path}")
    logger.info(f"Archivo de salida de parámetros: {output_gcs_path}")

    # Cargar los datos una sola vez
    local_features_path = gcs_utils.ensure_gcs_path_and_get_local(features_path)
    DF_FEATURES = pd.read_parquet(local_features_path)
    
    # Crear y ejecutar el estudio
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1) # Usar todos los cores disponibles

    # Guardar los mejores resultados
    best_params = study.best_trial.params
    best_params["features_path"] = features_path # Guardar referencia a los datos usados
    best_params["score"] = study.best_trial.value
    
    logger.info(f"Optimización completada. Mejor score (Profit Factor): {best_params['score']:.4f}")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_json_path = Path(tmpdir) / "best_params.json"
        with open(local_json_path, "w") as f:
            json.dump(best_params, f, indent=2)
        
        gcs_utils.upload_gcs_file(local_json_path, output_gcs_path)
        
    logger.info(f"🎉 Parámetros óptimos guardados en: {output_gcs_path}")
    return output_gcs_path

# --- Punto de Entrada para KFP ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task de Optimización de Hiperparámetros con Optuna.")
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--n-trials", type=int, default=25)
    
    args = parser.parse_args()

    # Construir la ruta de salida versionada
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    final_output_gcs_path = (
        f"{constants.LSTM_PARAMS_PATH}/{args.pair}/{args.timeframe}/"
        f"{timestamp}/best_params.json"
    )

    result_path = run_optimization(
        features_path=args.features_path,
        pair=args.pair,
        timeframe=args.timeframe,
        n_trials=args.n_trials,
        output_gcs_path=final_output_gcs_path,
    )
    
    # Imprimir la ruta final a stdout para que KFP la capture
    print(result_path)