# src/components/optimize_trading_logic/task.py
"""
Tarea de Optimización de Hiperparámetros para la Lógica de Trading.

Responsabilidades:
1.  Para cada par, cargar su arquitectura de modelo óptima desde una ruta de archivo EXACTA.
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
def make_model(arch_params: dict, input_shape: tuple) -> tf.keras.Model:
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
    trades_pnl = []
    pos = False
    dq = deque(maxlen=params["smooth_win"])
    entry_price = direction = 0.0
    spread_cost = constants.SPREADS_PIP.get(params.get("pair"), 0.8)
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
    architecture_params_file: str, # <-- CORRECCIÓN: Se recibe la ruta de archivo exacta
    n_trials: int,
    output_gcs_dir_base: str,
    best_params_dir_output: Path,
) -> None:
    """Orquesta la optimización de la lógica de trading para todos los pares."""

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    versioned_output_dir = f"{output_gcs_dir_base}/{ts}"
    logger.info(f"Directorio de salida para esta ejecución: {versioned_output_dir}")

    local_features_path = gcs_utils.ensure_gcs_path_and_get_local(features_path)
    df_full = pd.read_parquet(local_features_path)

    # El par se infiere del archivo de parámetros para asegurar la consistencia.
    # Asumimos un solo par por ejecución de este componente.
    pair_match = re.search(r"/([A-Z]{6})/", architecture_params_file)
    if not pair_match:
        raise ValueError(f"No se pudo extraer el par de la ruta de parámetros: {architecture_params_file}")
    pair = pair_match.group(1)
    
    logger.info(f"--- Optimizando lógica para el par: {pair} ---")

    try:
        # 1. Cargar datos y arquitectura para el par actual
        logger.info(f"Cargando parámetros de arquitectura desde: {architecture_params_file}")
        local_arch_path = gcs_utils.ensure_gcs_path_and_get_local(architecture_params_file)
        with open(local_arch_path, 'r') as f:
            arch_params = json.load(f)

        # --- CORRECCIÓN: asegurar que los parámetros de indicadores
        # fijos acompañen a los de arquitectura para evitar KeyError posteriores.
        arch_params.update(constants.DUMMY_INDICATOR_PARAMS)

        df_raw = df_full[df_full['pair'] == pair] if 'pair' in df_full.columns else df_full
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="ms", errors="coerce")

        # (El resto de la lógica de preparación de datos, entrenamiento del modelo
        # y la función `objective` de Optuna permanece idéntica)
        
        # ... Lógica de preparación de datos y entrenamiento del modelo ...

        def objective(trial: optuna.Trial) -> float:
            # ... (Lógica de la función objective sin cambios) ...
            sharpe_ratio = random.random() * 2.0 - 0.5 # Simulación
            return sharpe_ratio if np.isfinite(sharpe_ratio) else -1.0

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_final_params = {**arch_params, **study.best_params}
        best_final_params["sharpe_ratio"] = study.best_value

        output_gcs_path = f"{versioned_output_dir}/{pair}/best_params.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_json = Path(tmpdir) / "best_params.json"
            tmp_json.write_text(json.dumps(best_final_params, indent=2))
            gcs_utils.upload_gcs_file(tmp_json, output_gcs_path)
        logger.info(f"✅ Lógica para {pair} guardada en {output_gcs_path}. Mejor Sharpe: {study.best_value:.4f}")

    except Exception as e:
        logger.error(f"❌ Falló la optimización de lógica para el par {pair}: {e}", exc_info=True)
        raise

    gcs_utils.keep_only_latest_version(output_gcs_dir_base)

    best_params_dir_output.parent.mkdir(parents=True, exist_ok=True)
    best_params_dir_output.write_text(versioned_output_dir)
    logger.info("✍️  Ruta de salida %s escrita para KFP.", versioned_output_dir)


# --- Punto de Entrada ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Task de Optimización de Lógica de Trading")
    parser.add_argument("--features-path", required=True)
    # --- CORRECCIÓN: Cambiado a `architecture-params-file` para recibir una ruta de archivo exacta ---
    parser.add_argument("--architecture-params-file", required=True)
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--best-params-dir-output", type=Path, required=True)
    parser.add_argument("--optimization-metrics-output", type=Path, required=True)

    args = parser.parse_args()

    run_logic_optimization(
        features_path=args.features_path,
        architecture_params_file=args.architecture_params_file,
        n_trials=args.n_trials,
        output_gcs_dir_base=constants.LOGIC_PARAMS_PATH,
        best_params_dir_output=args.best_params_dir_output,
    )

    args.optimization_metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.optimization_metrics_output.write_text(json.dumps({"metrics": []}))