# src/components/optimize_model_architecture/task.py
"""
Tarea de Optimización de Hiperparámetros para la Arquitectura del Modelo.

Responsabilidades:
1.  Listar todos los archivos Parquet de datos preparados de una ruta GCS.
2.  Para cada par de divisas:
    a. Ejecutar un estudio de Optuna para encontrar los mejores hiperparámetros
       de la arquitectura del modelo (capas, unidades, dropout, etc.).
    b. La métrica a optimizar es la pérdida de validación (`val_loss`).
    c. Utilizar Pruning para descartar `trials` no prometedoras y ahorrar cómputo.
    d. Limpiar versiones antiguas de parámetros para mantener solo la más reciente.
3.  Guardar el archivo `best_architecture.json` para cada par en GCS.
4.  Devolver la ruta base donde se guardaron todos los archivos de arquitectura.
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

import gcsfs
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
        logger.info("🚀 GPU(s) detectadas – memoria dinámica y mixed-precision activadas.")
    else:
        raise RuntimeError("No GPU found")
except Exception as exc:
    logger.warning("⚠️ No se utilizará GPU (%s). Continuando con CPU.", exc)

# --- LÓGICA DE LIMPIEZA AÑADIDA ---
def _keep_only_latest_version(base_gcs_prefix: str) -> None:
    """
    Mantiene sólo el sub-directorio con timestamp (YYYYMMDDHHMMSS)
    más reciente y borra el resto.
    """
    try:
        fs = gcsfs.GCSFileSystem(project=constants.PROJECT_ID)
        base_gcs_prefix = base_gcs_prefix.rstrip("/") + "/"
        timestamp_re = re.compile(r"/(\d{14})/?$")
        
        # Necesitamos buscar un nivel más profundo para la estructura de este componente
        # (ej. .../architecture_v3/{pair}/{timestamp}/)
        parent_dir_of_versions = "/".join(base_gcs_prefix.split('/')[:-2])
        
        dirs = [p for p in fs.ls(parent_dir_of_versions) if fs.isdir(p) and timestamp_re.search(p)]

        if len(dirs) <= 1:
            return

        dirs.sort(key=lambda p: timestamp_re.search(p).group(1), reverse=True)
        
        for old_dir in dirs[1:]:
            logger.info("🗑️  Borrando versión de arquitectura antigua: gs://%s", old_dir)
            fs.rm(old_dir, recursive=True)
            
    except Exception as exc:
        logger.warning("No se pudo limpiar versiones antiguas de arquitectura: %s", exc)


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

# --- Función Principal de la Tarea ---

def run_architecture_optimization(
    *,
    features_path: str,
    n_trials: int,
    output_gcs_dir_base: str,
) -> None:
    """
    Orquesta el proceso completo de optimización de arquitectura para todos los pares.
    """
    logger.info("🚀 Iniciando HPO de Arquitectura con %d trials por par...", n_trials)

    all_feature_files = gcs_utils.list_gcs_files(features_path, suffix=".parquet")
    if not all_feature_files:
        raise FileNotFoundError(f"No se encontraron archivos Parquet en {features_path}")

    logger.info(f"Se procesarán {len(all_feature_files)} pares de divisas.")

    for gcs_file_path in all_feature_files:
        pair_match = re.search(r"([A-Z]{6})", gcs_file_path) # Búsqueda más genérica
        if not pair_match:
            logger.warning(f"No se pudo extraer el par del archivo: {gcs_file_path}. Saltando.")
            continue
        
        pair = pair_match.group(1)
        logger.info(f"--- Optimizando arquitectura para el par: {pair} ---")

        local_features_path = gcs_utils.ensure_gcs_path_and_get_local(gcs_file_path)
        df_raw = pd.read_parquet(local_features_path)
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
            }
            
            dummy_trading_params = {"horizon": 15, "atr_len": 14}
            df_ind = indicators.build_indicators(df_raw.copy(), {}, atr_len=dummy_trading_params["atr_len"])

            tick = 0.01 if pair.endswith("JPY") else 0.0001
            atr = df_ind[f"atr_{dummy_trading_params['atr_len']}"].values / tick
            close = df_ind.close.values
            fut_close = np.roll(close, -dummy_trading_params["horizon"])
            fut_close[-dummy_trading_params["horizon"]:] = np.nan
            diff = (fut_close - close) / tick
            up = np.maximum(diff, 0) / atr
            dn = np.maximum(-diff, 0) / atr

            mask = (~np.isnan(diff)) & (~np.isnan(atr))
            feature_cols = [c for c in df_ind.columns if c != "timestamp" and not c.startswith("atr_")]
            X_raw = df_ind.loc[mask, feature_cols].select_dtypes(include=np.number)

            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_raw)

            X_seq = np.stack([X_scaled[i - p["win"]: i] for i in range(p["win"], len(X_scaled))]).astype(np.float32)
            y_up_seq = up[mask][p["win"]:]
            y_dn_seq = dn[mask][p["win"]:]

            X_tr, X_val, y_up_tr, y_up_val, y_dn_tr, y_dn_val = train_test_split(
                X_seq, y_up_seq, y_dn_seq, test_size=0.2, shuffle=False
            )

            model = make_model(X_tr.shape[1:], p["lr"], p["dr"], p["filt"], p["units"], p["heads"])
            
            history = model.fit(
                X_tr, np.column_stack([y_up_tr, y_dn_tr]),
                validation_data=(X_val, np.column_stack([y_up_val, y_dn_val])),
                epochs=20,
                batch_size=64,
                verbose=0,
                callbacks=[
                    callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                    callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: optuna_pruning_callback(trial, epoch, logs))
                ],
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

    # Llamada a la limpieza después de procesar todos los pares.
    # Esto asume que todas las versiones están en el mismo directorio base.
    base_cleanup_path = "/".join(output_gcs_dir_base.split('/')[:-1])
    _keep_only_latest_version(base_cleanup_path)

    logger.info("✅ Optimización de arquitectura completada para todos los pares.")


# --- Punto de Entrada para Ejecución como Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Task de Optimización de Arquitectura de Modelo")
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--best-architecture-dir-output", type=Path, required=True)
    
    args = parser.parse_args()

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    output_dir_gcs = f"{constants.PARAMS_PATH}/architecture_v3/{ts}"

    run_architecture_optimization(
        features_path=args.features_path,
        n_trials=args.n_trials,
        output_gcs_dir_base=output_dir_gcs,
    )

    # Escribir la ruta del directorio de salida para que KFP la pase al siguiente componente
    args.best_architecture_dir_output.parent.mkdir(parents=True, exist_ok=True)
    args.best_architecture_dir_output.write_text(output_dir_gcs)