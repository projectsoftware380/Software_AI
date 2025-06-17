# src/components/optimize_model_architecture/task.py
"""
Tarea de Optimización de Hiperparámetros para la Arquitectura del Modelo.

Responsabilidades:
1.  Listar todos los archivos Parquet de datos preparados de una ruta GCS.
2.  Para cada par de divisas:
    a. Ejecutar un estudio de Optuna para encontrar los mejores hiperparámetros.
    b. La métrica a optimizar es la pérdida de validación (`val_loss`).
    c. Utilizar Pruning para descartar `trials` no prometedoras.
    d. Limpiar versiones antiguas de parámetros.
3.  Guardar el archivo `best_architecture.json` para cada par en una ruta GCS
    única y versionada por timestamp.
4.  Devolver la ruta base versionada donde se guardaron los resultados.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import re
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import callbacks, layers, models, optimizers

from src.shared import constants, gcs_utils, indicators

# --- Configuración del Logging ---
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Reproducibilidad y Configuración de GPU ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("🚀 GPU(s) detectadas y configuradas.")
    else:
        raise RuntimeError("No se encontró ninguna GPU.")
except Exception as exc:
    logger.warning("⚠️ No se utilizará GPU (%s). Continuando con CPU.", exc)


# --- Helpers del Modelo y Optuna ---
def make_model(inp_shape, lr, dr, filt, units, heads):
    """Construye el modelo Keras con la arquitectura especificada."""
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

def optuna_pruning_callback(trial: optuna.Trial, epoch: int, logs: dict):
    """Callback para reportar a Optuna y permitir la poda (pruning)."""
    trial.report(logs["val_loss"], step=epoch)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

def to_sequences(mat, up, dn, win):
    """Convierte una matriz de features en secuencias para el LSTM."""
    X, y_up, y_dn = [], [], []
    for i in range(win, len(mat)):
        X.append(mat[i - win:i])
        y_up.append(up[i])
        y_dn.append(dn[i])
    return np.asarray(X, np.float32), np.asarray(y_up, np.float32), np.asarray(y_dn, np.float32)

# --- Función Principal de la Tarea ---
def run_architecture_optimization(
    *,
    features_path: str,
    n_trials: int,
    output_gcs_dir_base: str,
    best_architecture_dir_output: Path,
) -> None:
    """
    Orquesta el proceso completo de optimización de arquitectura para todos los pares.
    """
    logger.info("🚀 Iniciando HPO de Arquitectura con %d trials por par...", n_trials)

    local_features_path = gcs_utils.ensure_gcs_path_and_get_local(features_path)
    df_full = pd.read_parquet(local_features_path)
    
    all_pairs = list(constants.SPREADS_PIP.keys())
    
    for pair in all_pairs:
        logger.info(f"--- Optimizando arquitectura para el par: {pair} ---")
        df_raw = df_full.copy()
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="ms", errors="coerce")

        def objective(trial: optuna.Trial) -> float:
            tf.keras.backend.clear_session()
            gc.collect()

            p = {
                "win": trial.suggest_int("win", 20, 60),
                "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                "dr": trial.suggest_float("dr", 0.1, 0.5),
                "filt": trial.suggest_categorical("filt", [16, 32, 64]),
                "units": trial.suggest_categorical("units", [32, 64, 128]),
                "heads": trial.suggest_categorical("heads", [2, 4, 8]),
                # Incluir otros hiperparámetros de indicadores
                **constants.DUMMY_INDICATOR_PARAMS
            }
            
            df_ind = indicators.build_indicators(df_raw.copy(), p, atr_len=14)
            
            tick = 0.01 if pair.endswith("JPY") else 0.0001
            horizon = p.get("horizon", 20) # Usar un valor por defecto si no está
            
            closes = df_ind.close.values
            atr_vals = df_ind[f"atr_14"].values / tick
            
            future_prices = np.roll(closes, -horizon)
            future_prices[-horizon:] = np.nan
            
            price_diffs = (future_prices - closes) / tick
            
            up_targets = np.maximum(price_diffs, 0) / atr_vals
            dn_targets = np.maximum(-price_diffs, 0) / atr_vals
            
            valid_mask = ~np.isnan(price_diffs)
            
            feature_cols = [c for c in df_ind.columns if "atr_" not in c and c != "timestamp"]
            X_data = df_ind.loc[valid_mask, feature_cols].select_dtypes(include=np.number).astype(np.float32)
            
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_data)
            
            X_seq, y_up_seq, y_dn_seq = to_sequences(
                X_scaled, up_targets[valid_mask], dn_targets[valid_mask], p["win"]
            )
            
            if len(X_seq) == 0:
                raise ValueError("No se generaron secuencias, revisa los parámetros y los datos.")

            X_train, X_val, y_up_train, y_up_val, y_dn_train, y_dn_val = train_test_split(
                X_seq, y_up_seq, y_dn_seq, test_size=0.2, shuffle=False
            )

            model = make_model(X_train.shape[1:], p["lr"], p["dr"], p["filt"], p["units"], p["heads"])
            
            # **CORRECCIÓN APLICADA AQUÍ**
            # Se asigna el resultado de model.fit a la variable 'history'.
            history = model.fit(
                X_train, np.vstack((y_up_train, y_dn_train)).T,
                validation_data=(X_val, np.vstack((y_up_val, y_dn_val)).T),
                epochs=40,
                batch_size=128,
                verbose=0,
                callbacks=[
                    callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                    callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: optuna_pruning_callback(trial, epoch, logs))
                ]
            )
            
            return min(history.history['val_loss'])

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_architecture_params = study.best_params
        best_architecture_params["best_val_loss"] = study.best_value

        pair_output_gcs_path = f"{output_gcs_dir_base}/{pair}/best_architecture.json"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_json = Path(tmpdir) / "best_architecture.json"
            tmp_json.write_text(json.dumps(best_architecture_params, indent=2))
            gcs_utils.upload_gcs_file(tmp_json, pair_output_gcs_path)
        
        logger.info(f"✅ Arquitectura para {pair} guardada. Mejor val_loss: {study.best_value:.4f}")

    parent_dir = str(Path(output_gcs_dir_base).parent)
    gcs_utils.keep_only_latest_version(parent_dir)

    best_architecture_dir_output.parent.mkdir(parents=True, exist_ok=True)
    best_architecture_dir_output.write_text(output_gcs_dir_base)
    logger.info("✍️  Ruta de salida %s escrita para KFP.", output_gcs_dir_base)


# --- Punto de Entrada para Ejecución como Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Task de Optimización de Arquitectura de Modelo")
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--best-architecture-dir-output", type=Path, required=True)
    
    args = parser.parse_args()

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    output_dir_gcs = f"{constants.ARCHITECTURE_PARAMS_PATH}/{ts}"

    run_architecture_optimization(
        features_path=args.features_path,
        n_trials=args.n_trials,
        output_gcs_dir_base=output_dir_gcs,
        best_architecture_dir_output=args.best_architecture_dir_output,
    )