# src/components/optimize_trading_logic/task.py
"""
Tarea de Optimización de Hiperparámetros para la Lógica de Trading.

Responsabilidades:
1.  Iterar sobre cada par de divisas definido en `constants.py`.
2.  Para cada par:
    a. Cargar su arquitectura de modelo óptima (previamente calculada).
    b. Construir y entrenar ese modelo UNA SOLA VEZ.
    c. Ejecutar un estudio de Optuna para encontrar los mejores parámetros
       de indicadores y de lógica de trading.
    d. La métrica a optimizar es el Sharpe Ratio, incluyendo costos por spread.
    e. Utilizar Pruning para ser más eficiente.
3.  Guardar el archivo `best_params.json` final y completo para cada par.
4.  Limpiar versiones antiguas de estos parámetros para mantener solo la más reciente.
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
from collections import deque  # <--- CORRECCIÓN AÑADIDA AQUÍ
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

# --- Configuración ---
warnings.filterwarnings("ignore", category=FutureWarning)
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
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("🚀 GPU(s) detectadas y configuradas.")
    else:
        raise RuntimeError("No GPU found")
except Exception as exc:
    logger.warning("⚠️ No se utilizará GPU (%s).", exc)

# --- Helpers ---

def _keep_only_latest_version(base_gcs_prefix: str, pair: str) -> None:
    """Mantiene solo la versión más reciente de parámetros para un par y borra el resto."""
    try:
        fs = gcsfs.GCSFileSystem(project=constants.PROJECT_ID)
        parent_dir = f"{base_gcs_prefix}/{pair}"
        timestamp_re = re.compile(r"/(\d{14})/?$")
        
        if not fs.exists(parent_dir):
            return

        dirs = [d for d in fs.ls(parent_dir) if fs.isdir(d) and timestamp_re.search(d)]
        if len(dirs) <= 1:
            return

        dirs.sort(key=lambda p: timestamp_re.search(p).group(1), reverse=True)
        for old_dir in dirs[1:]:
            logger.info("🗑️ Borrando versión de parámetros antigua para %s: gs://%s", pair, old_dir)
            fs.rm(old_dir, recursive=True)
    except Exception as exc:
        logger.warning("No se pudo limpiar versiones antiguas para %s: %s", pair, exc)

def make_model(arch_params: dict, input_shape: tuple) -> tf.keras.Model:
    """Construye el modelo Keras a partir de parámetros de arquitectura fijos."""
    x = inp = layers.Input(shape=input_shape, dtype=tf.float32)
    x = layers.Conv1D(arch_params["filt"], 3, padding="same", activation="relu")(x)
    x = layers.Bidirectional(layers.LSTM(arch_params["units"], return_sequences=True))(x)
    x = layers.MultiHeadAttention(num_heads=arch_params["heads"], key_dim=arch_params["units"])(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(arch_params["dr"])(x)
    out = layers.Dense(2, dtype="float32")(x)
    model = models.Model(inp, out)
    model.compile(optimizers.Adam(arch_params["lr"]), loss="mae")
    return model

def quick_bt(pred: np.ndarray, closes: np.ndarray, atr: np.ndarray, params: dict, tick: float) -> list[float]:
    """Simulador de backtest que devuelve una lista con el PnL de cada operación."""
    trades_pnl = []
    pos = False
    dq = deque(maxlen=params["smooth_win"])
    entry_price = direction = 0.0
    spread_cost = constants.SPREADS_PIP.get(params["pair"], 0.8)

    for (u, d), price, atr_i in zip(pred, closes, atr):
        mag, diff = max(u, d), abs(u - d)
        raw = 1 if u > d else -1
        cond = ((raw == 1 and mag >= params["min_thr_up"]) or (raw == -1 and mag >= params["min_thr_dn"])) and diff >= params["delta_min"]
        dq.append(raw if cond else 0)
        
        buys, sells = dq.count(1), dq.count(-1)
        signal = 1 if buys > params["smooth_win"] // 2 else -1 if sells > params["smooth_win"] // 2 else 0

        if not pos and signal != 0:
            pos, entry_price, direction = True, price, signal
            sl = (params["min_thr_up"] if direction == 1 else params["min_thr_dn"]) * atr_i
            tp = params["rr"] * sl
            continue

        if pos:
            pnl = ((price - entry_price) if direction == 1 else (entry_price - price)) / tick
            if pnl >= tp or pnl <= -sl:
                final_pnl = (tp if pnl >= tp else -sl) - spread_cost
                trades_pnl.append(final_pnl)
                pos = False
    return trades_pnl

# --- Función Principal ---

def run_logic_optimization(
    *,
    features_path_base: str,
    architecture_params_dir: str,
    n_trials: int,
    output_gcs_dir_base: str
) -> None:
    """Orquesta la optimización de la lógica de trading para todos los pares."""
    
    pairs_to_process = list(constants.SPREADS_PIP.keys())
    logger.info(f"🚀 Iniciando HPO de Lógica de Trading para {len(pairs_to_process)} pares...")

    for pair in pairs_to_process:
        logger.info(f"--- Optimizando lógica para el par: {pair} ---")
        
        try:
            # 1. Cargar datos y arquitectura para el par actual
            features_file = gcs_utils.find_gcs_file_for_pair(features_path_base, pair)
            arch_file = gcs_utils.find_gcs_file_for_pair(architecture_params_dir, pair, "best_architecture.json")
            
            local_features_path = gcs_utils.ensure_gcs_path_and_get_local(features_file)
            local_arch_path = gcs_utils.ensure_gcs_path_and_get_local(arch_file)
            
            df_raw = pd.read_parquet(local_features_path)
            df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="ms", errors="coerce")
            
            with open(local_arch_path, 'r') as f:
                arch_params = json.load(f)

            # 2. Preparar datos y entrenar el modelo UNA SOLA VEZ
            atr_len_fixed = 14
            df_ind_base = indicators.build_indicators(df_raw.copy(), {"sma_len": 50, "rsi_len": 14, "macd_fast": 12, "macd_slow": 26, "stoch_len": 14}, atr_len=atr_len_fixed)
            tick = 0.01 if pair.endswith("JPY") else 0.0001
            atr = df_ind_base[f"atr_{atr_len_fixed}"].values / tick
            close = df_ind_base.close.values

            horizon_fixed = 20
            fut_close = np.roll(close, -horizon_fixed)
            fut_close[-horizon_fixed:] = np.nan
            diff = (fut_close - close) / tick
            up = np.maximum(diff, 0) / atr
            dn = np.maximum(-diff, 0) / atr
            
            mask = (~np.isnan(diff)) & (~np.isnan(atr))
            feature_cols = [c for c in df_ind_base.columns if c != "timestamp" and not c.startswith("atr_")]
            X_raw = df_ind_base.loc[mask, feature_cols].select_dtypes(include=np.number)
            
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_raw)

            X_seq = np.stack([X_scaled[i - arch_params["win"]: i] for i in range(arch_params["win"], len(X_scaled))]).astype(np.float32)
            y_up_seq = up[mask][arch_params["win"]:]
            y_dn_seq = dn[mask][arch_params["win"]:]

            X_tr, X_val, y_up_tr, y_up_val, y_dn_tr, y_dn_val, _, closes_val, _, atr_val = train_test_split(
                X_seq, y_up_seq, y_dn_seq, close[mask][arch_params["win"]:], atr[mask][arch_params["win"]:], test_size=0.2, shuffle=False
            )
            
            model = make_model(arch_params, X_tr.shape[1:])
            model.fit(X_tr, np.column_stack([y_up_tr, y_dn_tr]), epochs=20, batch_size=64, verbose=0)
            
            predictions = model.predict(X_val, verbose=0)
            
            # 3. Definir la función objetivo para Optuna
            def objective(trial: optuna.Trial) -> float:
                trading_params = {
                    "pair": pair,
                    "horizon": trial.suggest_int("horizon", 10, 30),
                    "rr": trial.suggest_float("rr", 1.0, 3.0),
                    "min_thr_up": trial.suggest_float("min_thr_up", 0.4, 2.5),
                    "min_thr_dn": trial.suggest_float("min_thr_dn", 0.4, 2.5),
                    "delta_min": trial.suggest_float("delta_min", 0.01, 0.5),
                    "smooth_win": trial.suggest_int("smooth_win", 1, 7, step=2),
                    "sma_len": trial.suggest_categorical("sma_len", [20, 50, 100]),
                    "rsi_len": trial.suggest_categorical("rsi_len", [7, 14, 21]),
                    "macd_fast": trial.suggest_categorical("macd_fast", [8, 12, 16]),
                    "macd_slow": trial.suggest_categorical("macd_slow", [21, 26, 32]),
                    "stoch_len": trial.suggest_categorical("stoch_len", [14, 21, 28]),
                }
                
                trades_pnl = quick_bt(predictions, closes_val, atr_val, trading_params, tick)

                if len(trades_pnl) < 10:
                    return -1.0
                
                pnl_array = np.array(trades_pnl)
                if pnl_array.std() == 0:
                    return -1.0

                sharpe_ratio = pnl_array.mean() / pnl_array.std() * np.sqrt(252 * (24 * 60 / 15))
                return sharpe_ratio if np.isfinite(sharpe_ratio) else -1.0

            # 4. Ejecutar el estudio de Optuna
            # (El pruning no se puede aplicar aquí porque el entrenamiento ocurre fuera del trial)
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

            # 5. Guardar los resultados combinados
            best_final_params = {**arch_params, **study.best_params}
            best_final_params["sharpe_ratio"] = study.best_value

            ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            output_gcs_path = f"{output_gcs_dir_base}/{pair}/{ts}/best_params.json"
            
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_json = Path(tmpdir) / "best_params.json"
                tmp_json.write_text(json.dumps(best_final_params, indent=2))
                gcs_utils.upload_gcs_file(tmp_json, output_gcs_path)

            logger.info(f"✅ Lógica para {pair} guardada. Mejor Sharpe: {study.best_value:.4f}")

            # 6. Limpiar versiones antiguas para este par
            _keep_only_latest_version(output_gcs_dir_base, pair)

        except Exception as e:
            logger.error(f"❌ Falló la optimización de lógica para el par {pair}: {e}", exc_info=True)
            continue
    
    logger.info("✅ Optimización de lógica de trading completada para todos los pares.")


# --- Punto de Entrada ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Task de Optimización de Lógica de Trading")
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--architecture-params-dir", required=True)
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--best-params-dir-output", type=Path, required=True)
    parser.add_argument("--optimization-metrics-output", type=Path, required=True)
    
    args = parser.parse_args()

    output_dir_gcs = f"{constants.LSTM_PARAMS_PATH}"

    run_logic_optimization(
        features_path_base=args.features_path,
        architecture_params_dir=args.architecture_params_dir,
        n_trials=args.n_trials,
        output_gcs_dir_base=output_dir_gcs,
    )
    
    args.best_params_dir_output.parent.mkdir(parents=True, exist_ok=True)
    args.best_params_dir_output.write_text(output_dir_gcs)
    
    args.optimization_metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.optimization_metrics_output.write_text(json.dumps({"metrics": []}))