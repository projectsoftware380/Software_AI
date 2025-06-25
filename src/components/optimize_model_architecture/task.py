# src/components/optimize_model_architecture/task.py
"""
Tarea de Optimización de Hiperparámetros para la Arquitectura del Modelo. (Corregido y con Logging Robusto)
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
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
        logger.info("🚀 GPU(s) detectadas y configuradas para el HPO de arquitectura.")
    else:
        logger.warning("⚠️ No se encontró ninguna GPU. La optimización se ejecutará en CPU y será muy lenta.")
except Exception as exc:
    logger.warning("⚠️ No se pudo configurar la GPU (%s). Continuando con CPU.", exc)


# --- Helpers del Modelo y Optuna (Lógica Original Intacta) ---
def make_model(inp_shape, lr, dr, filt, units, heads):
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
    trial.report(logs["val_loss"], step=epoch)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

def to_sequences(mat, up, dn, win):
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
    pair: str,
    output_gcs_dir_base: str,
    best_architecture_dir_output: Path,
    cleanup: bool = True,
) -> None:
    """
    Orquesta el proceso completo de optimización de arquitectura para UN SOLO PAR.
    """
    logger.info(f"▶️ Iniciando optimize_model_architecture para el par '{pair}' con {n_trials} trials.")
    logger.info(f"  - Features de entrada: {features_path}")
    logger.info(f"  - Directorio base de salida: {output_gcs_dir_base}")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_features_path = gcs_utils.download_gcs_file(features_path, Path(tmpdir))
            if not local_features_path:
                raise FileNotFoundError(f"No se pudo descargar el archivo de features: {features_path}")
            df_raw = pd.read_parquet(local_features_path)
            
        # AJUSTE: Convertir el índice 'timestamp' en una columna para evitar KeyError.
        df_raw = df_raw.reset_index()
        
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="ms", errors="coerce")
        logger.info(f"DataFrame cargado para {pair}. Shape: {df_raw.shape}")

        def objective(trial: optuna.Trial) -> float:
            tf.keras.backend.clear_session()
            gc.collect()

            # --- CORRECCIÓN: Se unen los parámetros del trial con los de los indicadores ---
            trial_params = {
                "win": trial.suggest_int("win", 20, 60),
                "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                "dr": trial.suggest_float("dr", 0.1, 0.5),
                "filt": trial.suggest_categorical("filt", [16, 32, 64]),
                "units": trial.suggest_categorical("units", [32, 64, 128]),
                "heads": trial.suggest_categorical("heads", [2, 4, 8]),
            }
            # Se crea un diccionario completo con todos los parámetros necesarios
            p = {**trial_params, **constants.DUMMY_INDICATOR_PARAMS}
            
            logger.debug(f"Trial #{trial.number}: Probando parámetros {p}")

            # Esta llamada ahora recibe un diccionario 'p' completo y no fallará.
            df_ind = indicators.build_indicators(df_raw.copy(), p, atr_len=14)
            tick = 0.01 if pair.endswith("JPY") else 0.0001
            horizon = p.get("win", 20) # Se usa 'win' como el horizonte
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
                logger.warning(f"Trial #{trial.number}: No se generaron secuencias, podando trial.")
                raise optuna.exceptions.TrialPruned()

            X_train, X_val, y_up_train, y_up_val, y_dn_train, y_dn_val = train_test_split(
                X_seq, y_up_seq, y_dn_seq, test_size=0.2, shuffle=False
            )

            model = make_model(X_train.shape[1:], p["lr"], p["dr"], p["filt"], p["units"], p["heads"])
            
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
            
            val_loss = min(history.history['val_loss'])
            logger.debug(f"Trial #{trial.number} finalizado con val_loss: {val_loss:.5f}")
            return val_loss

        logger.info(f"Iniciando estudio de Optuna para {pair}...")
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        logger.info(f"Estudio de Optuna para {pair} completado.")

        # --- CORRECCIÓN: Se guardan tanto los mejores parámetros del trial como los de los indicadores ---
        best_params_from_trial = study.best_params
        final_best_params = {**best_params_from_trial, **constants.DUMMY_INDICATOR_PARAMS}
        final_best_params["best_val_loss"] = study.best_value

        logger.info(f"Mejor arquitectura encontrada para {pair}. Val_loss: {study.best_value:.5f}")
        logger.info(f"Mejores parámetros finales: {json.dumps(final_best_params, indent=2)}")

        pair_output_gcs_path = f"{output_gcs_dir_base}/best_architecture.json"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_json = Path(tmpdir) / "best_architecture.json"
            # Se guardan los parámetros combinados
            tmp_json.write_text(json.dumps(final_best_params, indent=2))
            logger.info(f"Guardando mejores parámetros en: {pair_output_gcs_path}")
            gcs_utils.upload_gcs_file(tmp_json, pair_output_gcs_path)

        if cleanup:
            base_cleanup_path = f"{constants.ARCHITECTURE_PARAMS_PATH}/{pair}"
            logger.info(f"Iniciando limpieza de versiones antiguas en: {base_cleanup_path}")
            gcs_utils.keep_only_latest_version(base_cleanup_path)

        best_architecture_dir_output.parent.mkdir(parents=True, exist_ok=True)
        best_architecture_dir_output.write_text(output_gcs_dir_base)
        logger.info(f"Ruta de salida '{output_gcs_dir_base}' escrita para KFP.")

    except Exception as e:
        logger.critical(f"❌ Fallo fatal en optimize_model_architecture para el par '{pair}'. Error: {e}", exc_info=True)
        raise

    logger.info(f"🏁 Componente optimize_model_architecture para '{pair}' completado exitosamente.")


# --- Punto de Entrada para Ejecución como Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Task de Optimización de Arquitectura de Modelo")
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--cleanup", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--best-architecture-dir-output", type=Path, required=True)
    
    args = parser.parse_args()
    
    logger.info("Componente 'optimize_model_architecture' iniciado con los siguientes argumentos:")
    for key, value in vars(args).items():
        logger.info(f"  - {key}: {value}")
        
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    output_dir_gcs = f"{constants.ARCHITECTURE_PARAMS_PATH}/{args.pair}/{ts}"

    run_architecture_optimization(
        features_path=args.features_path,
        n_trials=args.n_trials,
        pair=args.pair,
        output_gcs_dir_base=output_dir_gcs,
        best_architecture_dir_output=args.best_architecture_dir_output,
        cleanup=args.cleanup,
    )