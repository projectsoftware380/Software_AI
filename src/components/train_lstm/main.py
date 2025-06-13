# RUTA: src/components/train_lstm/main.py
# 𝐂𝐎́𝐃𝐈𝐆𝐎 𝐀𝐂𝐓𝐔𝐀𝐋𝐈𝐙𝐀𝐃𝐎 — aborta si no hay GPU

from __future__ import annotations
import argparse, json, logging, os, random, sys
from datetime import datetime, timezone
from pathlib import Path

import joblib, numpy as np, pandas as pd, pandas_ta as ta, tensorflow as tf
from google.cloud import storage
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import callbacks, layers, models, mixed_precision

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
def setup_environment() -> None:
    """Semillas + **fail-fast** si Vertex arrancó sin GPU."""
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

# ─────────── helpers de features y secuencias ───────────────
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

# ────────────── utilidades GCS (sin cambios) ────────────────
def download_gcs_file(uri: str, dest: str | Path):
    logger.info("Descargando %s → %s", uri, dest)
    bucket, blob = uri.replace("gs://", "").split("/", 1)
    client = storage.Client()
    client.bucket(bucket).blob(blob).download_to_filename(dest)

def upload_local_directory_to_gcs(src: str | Path, uri: str):
    logger.info("Subiendo %s → %s", src, uri)
    bucket, prefix = uri.replace("gs://", "").split("/", 1)
    client = storage.Client(); src = Path(src)
    for f in src.rglob("*"):
        if f.is_file():
            client.bucket(bucket).blob(f"{prefix}/{f.relative_to(src)}").upload_from_filename(f)

# ───────────────────── pipeline de entrenamiento ────────────
def train_final_model(pair:str, timeframe:str, params_path:str,
                      features_gcs_path:str, output_gcs_base_dir:str):
    setup_environment()

    tmp = Path("/tmp/data"); tmp.mkdir(exist_ok=True)
    loc_params = tmp/"params.json";   download_gcs_file(params_path, loc_params)
    hp         = json.loads(loc_params.read_text())
    loc_feat   = tmp/Path(features_gcs_path).name; download_gcs_file(features_gcs_path, loc_feat)
    df_raw     = pd.read_parquet(loc_feat).reset_index(drop=True)
    logger.info("Features parquet cargado — shape %s", df_raw.shape)

    tick = 0.01 if pair.endswith("JPY") else 0.0001
    df_b = build_indicators(df_raw, hp["sma_len"], hp["rsi_len"],
                            hp["macd_fast"], hp["macd_slow"], hp["stoch_len"], 14)

    clo, atr = df_b.close.values, df_b["atr_14"].values / tick
    fut      = np.roll(clo, -hp["horizon"]); fut[-hp["horizon"]:] = np.nan
    diff     = (fut - clo) / tick
    up, dn   = np.maximum(diff, 0)/atr, np.maximum(-diff, 0)/atr
    mask     = (~np.isnan(diff))

    feat_cols = df_b.columns.difference([c for c in df_b.columns if "atr_" in c] + ["timestamp"])
    X_raw = df_b.loc[mask, feat_cols].select_dtypes(include=np.number).astype(np.float32)
    scaler = RobustScaler(); X_scaled = scaler.fit_transform(X_raw)

    X_seq, y_up, y_dn = to_sequences(X_scaled, up[mask], dn[mask], hp["win"])
    logger.info("Dataset final: X=%s", X_seq.shape)

    model = make_model(X_seq.shape[1:], hp["lr"], hp["dr"], hp["filt"], hp["units"], hp["heads"])
    model.fit(X_seq, np.vstack([y_up, y_dn]).T, epochs=60, batch_size=128,
              callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
              verbose=1)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    gcs_out = f"{output_gcs_base_dir}/{pair}/{timeframe}/{ts}"
    loc_dir = Path(f"/tmp/artifacts/{ts}"); loc_dir.mkdir(parents=True)
    
    # === AJUSTE CORREGIDO: Se cambia de .h5 a .keras ===
    model.save(loc_dir/"model.keras"); joblib.dump(scaler, loc_dir/"scaler.pkl")
    # ======================================================

    (loc_dir/"params.json").write_text(json.dumps(hp, indent=4))
    upload_local_directory_to_gcs(loc_dir, gcs_out)
    logger.info("✅ Artefactos subidos a %s", gcs_out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--output-gcs-base-dir", required=True)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--timeframe", required=True)
    ap.add_argument("--features-gcs-path", required=True)
    a = ap.parse_args()
    train_final_model(a.pair, a.timeframe, a.params, a.features_gcs_path, a.output_gcs_base_dir)