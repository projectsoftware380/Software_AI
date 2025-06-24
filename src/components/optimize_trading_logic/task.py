# src/components/optimize_trading_logic/task.py
"""
Tarea de Optimización de Hiperparámetros para la Lógica de Trading. (Corregido y con Logging Robusto)

Responsabilidades:
1.  Recibir datos y una arquitectura de modelo para UN SOLO PAR.
2.  Ejecutar un estudio de Optuna para encontrar los mejores parámetros de lógica.
3.  La métrica a optimizar es el 'sharpe_ratio'.
4.  Guardar el archivo `best_params.json` (conteniendo la lógica Y la arquitectura) 
    para el par procesado en un directorio versionado.
5.  Limpiar las versiones antiguas de los parámetros.
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

# Módulos internos
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

# --- Helpers de Modelo y Secuencias (Lógica Original Intacta) ---
def make_model(inp_shape, lr, dr, filt, units, heads):
    x = inp = tf.keras.layers.Input(shape=inp_shape, dtype=tf.float32)
    x = tf.keras.layers.Conv1D(filt, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(x)
    x = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=units)(x, x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(dr)(x)
    out = tf.keras.layers.Dense(2, dtype="float32")(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(tf.keras.optimizers.Adam(lr), loss="mae")
    return model

def to_sequences(mat, up, dn, win):
    X, y_up, y_dn = [], [], []
    for i in range(win, len(mat)):
        X.append(mat[i - win:i])
        y_up.append(up[i])
        y_dn.append(dn[i])
    return np.asarray(X, np.float32), np.asarray(y_up, np.float32), np.asarray(y_dn, np.float32)

# --- Función Principal de la Tarea ---
def run_hpo_logic(
    *,
    features_path: str,
    architecture_params_file: str,
    n_trials: int,
    pair: str,
    output_gcs_dir_base: str,
    best_params_dir_output: Path,
    cleanup: bool = True,
):
    """
    Orquesta el proceso de HPO para la lógica de trading de UN SOLO PAR.
    """
    logger.info(f"▶️ Iniciando optimize_trading_logic para el par '{pair}' con {n_trials} trials.")
    logger.info(f"  - Features de entrada: {features_path}")
    logger.info(f"  - Archivo de arquitectura: {architecture_params_file}")
    logger.info(f"  - Directorio base de salida: {output_gcs_dir_base}")

    try:
        logger.info("Cargando artefactos de entrada...")
        local_features_path = gcs_utils.ensure_gcs_path_and_get_local(features_path)
        df_full = pd.read_parquet(local_features_path)
        logger.info(f"DataFrame de features cargado. Shape: {df_full.shape}")
        
        local_arch_path = gcs_utils.ensure_gcs_path_and_get_local(architecture_params_file)
        with open(local_arch_path) as f:
            architecture_params = json.load(f)
        logger.info(f"Parámetros de arquitectura cargados: {architecture_params}")

        def objective(trial: optuna.Trial) -> float:
            tf.keras.backend.clear_session()
            gc.collect()

            # --- CORRECCIÓN: Se combinan los parámetros del trial con los de la arquitectura ---
            # Esto asegura que todas las claves necesarias (sma_len, macd_fast, etc.) estén presentes.
            p = {
                "take_profit": trial.suggest_float("take_profit", 0.5, 3.0),
                "stop_loss": trial.suggest_float("stop_loss", 0.5, 3.0),
                "buy_threshold": trial.suggest_float("buy_threshold", 0.5, 1.0),
                "sell_threshold": trial.suggest_float("sell_threshold", 0.5, 1.0),
                **architecture_params,
            }
            logger.debug(f"Trial #{trial.number}: Probando parámetros {p}")

            df_ind = indicators.build_indicators(df_full.copy(), p, atr_len=14)
            tick = 0.01 if pair.endswith("JPY") else 0.0001
            horizon = p.get("win", 20)
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
            
            if len(X_seq) < 10:
                logger.warning(f"Trial #{trial.number}: No hay suficientes secuencias ({len(X_seq)}), devolviendo -1.0")
                return -1.0

            X_train, X_val, y_up_train, y_up_val, y_dn_train, y_dn_val = train_test_split(
                X_seq, y_up_seq, y_dn_seq, test_size=0.2, shuffle=False
            )

            model = make_model(X_train.shape[1:], p["lr"], p["dr"], p["filt"], p["units"], p["heads"])
            model.fit(
                X_train, np.vstack((y_up_train, y_dn_train)).T,
                validation_data=(X_val, np.vstack((y_up_val, y_dn_val)).T),
                epochs=1, batch_size=128, verbose=0,
            )
            
            preds = model.predict(X_val)
            up_pred, dn_pred = preds[:, 0], preds[:, 1]
            
            is_buy = up_pred > p["buy_threshold"]
            is_sell = dn_pred > p["sell_threshold"]
            
            profit = np.where(is_buy, y_up_val - p["take_profit"], 0)
            profit = np.where(is_sell, np.where(y_dn_val < p["stop_loss"], p["stop_loss"] - y_dn_val, 0), profit)

            returns = profit[is_buy | is_sell]
            
            if len(returns) < 10:
                logger.warning(f"Trial #{trial.number}: No hay suficientes trades ({len(returns)}), devolviendo -1.0")
                return -1.0
            
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else -1.0
            logger.debug(f"Trial #{trial.number} finalizado con Sharpe Ratio: {sharpe:.5f}")
            return sharpe

        logger.info(f"Iniciando estudio de Optuna para {pair}...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        logger.info(f"Estudio de Optuna para {pair} completado.")
        
        # --- CORRECCIÓN: Se combinan los parámetros de lógica Y de arquitectura para guardar un artefacto completo ---
        best_logic_params = study.best_params
        best_final_params = {**architecture_params, **best_logic_params}
        best_final_params["sharpe_ratio"] = study.best_value

        logger.info(f"Mejor lógica encontrada para {pair}. Sharpe Ratio: {study.best_value:.5f}")
        logger.info(f"Mejores parámetros combinados guardados: {json.dumps(best_final_params, indent=2)}")

        pair_output_gcs_path = f"{output_gcs_dir_base}/best_params.json"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_json = Path(tmpdir) / "best_params.json"
            # Se guardan los parámetros combinados para el siguiente paso de la pipeline
            tmp_json.write_text(json.dumps(best_final_params, indent=2))
            logger.info(f"Guardando mejores parámetros en: {pair_output_gcs_path}")
            gcs_utils.upload_gcs_file(tmp_json, pair_output_gcs_path)
        
        if cleanup:
            base_cleanup_path = f"{constants.LOGIC_PARAMS_PATH}/{pair}"
            logger.info(f"Iniciando limpieza de versiones antiguas en: {base_cleanup_path}")
            gcs_utils.keep_only_latest_version(base_cleanup_path)

        best_params_dir_output.parent.mkdir(parents=True, exist_ok=True)
        best_params_dir_output.write_text(output_gcs_dir_base)
        logger.info(f"Ruta de salida '{output_gcs_dir_base}' escrita para KFP.")

    except Exception as e:
        logger.critical(f"❌ Fallo fatal en optimize_trading_logic para el par '{pair}'. Error: {e}", exc_info=True)
        raise
    
    logger.info(f"🏁 Componente optimize_trading_logic para '{pair}' completado exitosamente.")

# --- Punto de Entrada para Ejecución como Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Task de Optimización de Lógica de Trading")
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--architecture-params-file", required=True)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--cleanup", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--best-params-dir-output", type=Path, required=True)
    
    args = parser.parse_args()

    logger.info("Componente 'optimize_trading_logic' iniciado con los siguientes argumentos:")
    for key, value in vars(args).items():
        logger.info(f"  - {key}: {value}")

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    output_dir_gcs = f"{constants.LOGIC_PARAMS_PATH}/{args.pair}/{ts}"

    run_hpo_logic(
        features_path=args.features_path,
        architecture_params_file=args.architecture_params_file,
        n_trials=args.n_trials,
        pair=args.pair,
        output_gcs_dir_base=output_dir_gcs,
        best_params_dir_output=args.best_params_dir_output,
        cleanup=args.cleanup,
    )