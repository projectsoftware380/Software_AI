# RUTA: src/components/train_lstm/main.py
# 𝐂𝐎́𝐃𝐈𝐆𝐎 𝐂𝐎𝐑𝐑𝐄𝐆𝐈𝐃𝐎 — Se adhiere estrictamente a la ruta de salida proporcionada.

from __future__ import annotations
import argparse, json, logging, os, random, sys, re
from datetime import datetime, timezone
from pathlib import Path

import joblib, numpy as np, pandas as pd, pandas_ta as ta, tensorflow as tf
from google.cloud import storage
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import callbacks, layers, models, mixed_precision

# AJUSTE: Se importa gcs_utils para usar sus funciones estandarizadas.
from src.shared import constants, gcs_utils

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
def setup_environment() -> None:
    """Configura semillas y el entorno de GPU."""
    np.random.seed(42); tf.random.set_seed(42); random.seed(42)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("GPU requerida y no detectada — abortando entrenamiento.")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    mixed_precision.set_global_policy("mixed_float16")
    logger.info("🚀 GPUs: %s | mixed_precision: %s",
                [g.name for g in gpus], mixed_precision.global_policy().name)

# ─────────── Helpers de features y secuencias (sin cambios) ───────────────
def build_indicators(df, sma_len, rsi_len, macf, macs, stoch_len, atr_len):
    out = df.copy()
    out[f"sma_{sma_len}"]   = ta.sma(out.close,  length=sma_len)
    out[f"rsi_{rsi_len}"]   = ta.rsi(out.close,  length=rsi_len)
    macd                   = ta.macd(out.close, fast=macf, slow=macs, signal=9)
    out[[f"macd_{macf}_{macs}", f"macd_signal_{macf}_{macs}", f"macd_hist_{macf}_{macs}"]] = macd
    stoch                  = ta.stoch(out.high, out.low, out.close, k=stoch_len, d=3)
    out[[f"stoch_k_{stoch_len}", f"stoch_d_{stoch_len}"]] = stoch
    out[f"atr_{atr_len}"]   = ta.atr(out.high, out.low, out.close, length=atr_len)
    out.bfill(inplace=True)
    return out

def to_sequences(mat, up, dn, win):
    X, y_up, y_dn = [], [], []
    for i in range(win, len(mat)):
        X.append(mat[i-win:i]); y_up.append(up[i]); y_dn.append(dn[i])
    return np.asarray(X, np.float32), np.asarray(y_up, np.float32), np.asarray(y_dn, np.float32)

def make_model(inp_sh, lr, dr, filt, units, heads):
    inp = layers.Input(shape=inp_sh, dtype=tf.float32)
    x   = layers.Conv1D(filt, 3, padding="same", activation="relu")(inp)
    x   = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
    x   = layers.MultiHeadAttention(num_heads=heads, key_dim=units)(x, x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dropout(dr)(x)
    out = layers.Dense(2, dtype="float32")(x)
    mdl = models.Model(inp, out)
    mdl.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mae")
    return mdl

# ───────────────────── Lógica de entrenamiento principal ────────────
# AJUSTE: La función ahora recibe la ruta de salida final y exacta.
def train_final_model(
    pair: str,
    timeframe: str,
    params_path: str,
    features_gcs_path: str,
    output_gcs_final_dir: str # <- Argumento corregido
):
    """
    Ejecuta el pipeline de entrenamiento completo y guarda los artefactos en la
    ruta GCS final proporcionada.
    """
    setup_environment()

    # Descarga de artefactos necesarios
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        loc_params = gcs_utils.download_gcs_file(params_path, tmp_path)
        hp = json.loads(loc_params.read_text())
        
        loc_feat = gcs_utils.download_gcs_file(features_gcs_path, tmp_path)
        df_raw = pd.read_parquet(loc_feat).reset_index(drop=True)
        logger.info("Features parquet cargado — shape %s", df_raw.shape)

        # Lógica de preparación de datos y entrenamiento (sin cambios)
        tick = 0.01 if pair.endswith("JPY") else 0.0001
        df_b = build_indicators(df_raw, hp["sma_len"], hp["rsi_len"],
                                hp["macd_fast"], hp["macd_slow"], hp["stoch_len"], 14)

        clo, atr = df_b.close.values, df_b["atr_14"].values / tick
        fut = np.roll(clo, -hp["horizon"]); fut[-hp["horizon"]:] = np.nan
        diff = (fut - clo) / tick
        up, dn = np.maximum(diff, 0)/atr, np.maximum(-diff, 0)/atr
        mask = (~np.isnan(diff))

        feat_cols = df_b.columns.difference([c for c in df_b.columns if "atr_" in c] + ["timestamp"])
        X_raw = df_b.loc[mask, feat_cols].select_dtypes(include=np.number).astype(np.float32)
        scaler = RobustScaler(); X_scaled = scaler.fit_transform(X_raw)

        X_seq, y_up, y_dn = to_sequences(X_scaled, up[mask], dn[mask], hp["win"])
        logger.info("Dataset final: X=%s", X_seq.shape)

        model = make_model(X_seq.shape[1:], hp["lr"], hp["dr"], hp["filt"], hp["units"], hp["heads"])
        model.fit(X_seq, np.vstack([y_up, y_dn]).T, epochs=60, batch_size=128,
                  callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                  verbose=1)

        # AJUSTE: Guardar artefactos localmente antes de subirlos.
        loc_artifacts_dir = tmp_path / "artifacts"
        loc_artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        model.save(loc_artifacts_dir / "model.keras")
        joblib.dump(scaler, loc_artifacts_dir / "scaler.pkl")
        (loc_artifacts_dir / "params.json").write_text(json.dumps(hp, indent=4))
        
        # AJUSTE: Subir el directorio de artefactos a la RUTA FINAL EXACTA proporcionada.
        gcs_utils.upload_local_directory_to_gcs(loc_artifacts_dir, output_gcs_final_dir)
        logger.info("✅ Artefactos subidos a %s", output_gcs_final_dir)

# AJUSTE: La lógica de limpieza se elimina de este script.

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="Ruta GCS al archivo de parámetros.")
    ap.add_argument("--pair", required=True)
    ap.add_argument("--timeframe", required=True)
    ap.add_argument("--features-gcs-path", required=True, help="Ruta GCS al parquet de features.")
    
    # AJUSTE: El argumento de salida ahora es la ruta final y exacta.
    ap.add_argument(
        "--output-gcs-final-dir",
        required=True,
        help="Ruta GCS final y exacta donde se guardarán los artefactos del modelo."
    )
    
    a = ap.parse_args()
    
    train_final_model(
        pair=a.pair,
        timeframe=a.timeframe,
        params_path=a.params,
        features_gcs_path=a.features_gcs_path,
        output_gcs_final_dir=a.output_gcs_final_dir
    )