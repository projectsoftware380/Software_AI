# src/components/optimize_trading_logic/task.py
"""
Tarea de Optimización de Hiperparámetros para la Lógica de Trading.

Responsabilidades:
1.  Para cada par, cargar su arquitectura de modelo óptima desde una ruta versionada.
2.  Construir y entrenar el modelo una sola vez por par.
3.  Ejecutar un estudio de Optuna para encontrar los mejores parámetros de trading.
4.  Guardar el archivo `best_params.json` en una nueva ruta GCS versionada.
5.  Propagar la ruta de salida versionada al siguiente componente.
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
        raise RuntimeError("No se encontró ninguna GPU.")
except Exception as exc:
    logger.warning("⚠️ No se utilizará GPU (%s).", exc)

# --- Helpers ---
# La lógica de make_model y quick_bt no necesita cambios.
def make_model(arch_params: dict, input_shape: tuple) -> tf.keras.Model:
    # ... (sin cambios)
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
    # ... (sin cambios)
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
    features_path: str,
    architecture_params_dir: str,
    n_trials: int,
    output_gcs_dir_base: str,
    best_params_dir_output: Path,
) -> None:
    """Orquesta la optimización de la lógica de trading para todos los pares."""
    
    # AJUSTE: La tarea ahora crea un único directorio de salida versionado para la ejecución.
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    versioned_output_dir = f"{output_gcs_dir_base}/{ts}"
    logger.info(f"Directorio de salida para esta ejecución: {versioned_output_dir}")
    
    pairs_to_process = list(constants.SPREADS_PIP.keys())
    local_features_path = gcs_utils.ensure_gcs_path_and_get_local(features_path)
    df_full = pd.read_parquet(local_features_path)

    for pair in pairs_to_process:
        logger.info(f"--- Optimizando lógica para el par: {pair} ---")
        
        try:
            # 1. Cargar datos y arquitectura para el par actual
            # AJUSTE: Busca el archivo de arquitectura dentro del directorio versionado recibido.
            arch_file = gcs_utils.find_gcs_file_in_timestamped_dirs(
                base_gcs_path=f"{architecture_params_dir}/{pair}", filename="best_architecture.json", recursive=False
            )
            if not arch_file:
                raise FileNotFoundError(f"No se encontró best_architecture.json para {pair} en {architecture_params_dir}")

            local_arch_path = gcs_utils.ensure_gcs_path_and_get_local(arch_file)
            
            # Asumimos que df_full contiene datos de todos los pares, o se podría filtrar
            df_raw = df_full # df_full[df_full['pair'] == pair] si es necesario
            df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="ms", errors="coerce")
            
            with open(local_arch_path, 'r') as f:
                arch_params = json.load(f)

            # 2. Preparar datos y entrenar el modelo (lógica sin cambios)
            # ...
            
            # 3. Definir la función objetivo para Optuna (lógica sin cambios)
            def objective(trial: optuna.Trial) -> float:
                # ...
                return sharpe_ratio if np.isfinite(sharpe_ratio) else -1.0
            
            # 4. Ejecutar el estudio de Optuna (lógica sin cambios)
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

            # 5. Guardar los resultados combinados
            best_final_params = {**arch_params, **study.best_params}
            best_final_params["sharpe_ratio"] = study.best_value

            # AJUSTE: La ruta de salida final se construye dentro del directorio versionado de la ejecución.
            output_gcs_path = f"{versioned_output_dir}/{pair}/best_params.json"
            
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_json = Path(tmpdir) / "best_params.json"
                tmp_json.write_text(json.dumps(best_final_params, indent=2))
                gcs_utils.upload_gcs_file(tmp_json, output_gcs_path)

            logger.info(f"✅ Lógica para {pair} guardada en {output_gcs_path}. Mejor Sharpe: {study.best_value:.4f}")

        except Exception as e:
            logger.error(f"❌ Falló la optimización de lógica para el par {pair}: {e}", exc_info=True)
            continue
    
    # 6. Limpieza de versiones antiguas
    gcs_utils.keep_only_latest_version(output_gcs_dir_base)

    # AJUSTE CRÍTICO: Propagar la ruta del directorio versionado al siguiente componente.
    best_params_dir_output.parent.mkdir(parents=True, exist_ok=True)
    best_params_dir_output.write_text(versioned_output_dir)
    logger.info("✍️  Ruta de salida %s escrita para KFP.", versioned_output_dir)


# --- Punto de Entrada ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Task de Optimización de Lógica de Trading")
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--architecture-params-dir", required=True)
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--best-params-dir-output", type=Path, required=True)
    parser.add_argument("--optimization-metrics-output", type=Path, required=True)
    
    args = parser.parse_args()

    # AJUSTE: El script ahora recibe la ruta base desde las constantes y construye
    # la ruta versionada internamente.
    run_logic_optimization(
        features_path=args.features_path,
        architecture_params_dir=args.architecture_params_dir,
        n_trials=args.n_trials,
        output_gcs_dir_base=constants.LOGIC_PARAMS_PATH,
        best_params_dir_output=args.best_params_dir_output,
    )
    
    # KFP requiere que se escriba un archivo de métricas, aunque esté vacío.
    args.optimization_metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.optimization_metrics_output.write_text(json.dumps({"metrics": []}))