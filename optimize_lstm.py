#!/usr/bin/env python3
import os, sys, random, warnings, tempfile, json, gc
from pathlib import Path
import argparse
from datetime import datetime

# ── Importaciones de librerías principales ──────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
if not hasattr(np, "NaN"): np.NaN = np.nan
import pandas as pd, optuna, joblib
from google.cloud import storage
from google.oauth2 import service_account
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, mixed_precision
from collections import deque

# Importación corregida para módulos de core
from indicators import build_indicators

# reproducibilidad
random.seed(42); np.random.seed(42); tf.random.set_seed(42)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)
mixed_precision.set_global_policy("mixed_float16")

# ── helpers GCS ────────────────────────────────────────────────
def gcs_client():
    """
    Obtiene un cliente de GCS, usando credenciales de cuenta de servicio
    si GOOGLE_APPLICATION_CREDENTIALS está definida, o por defecto si es en GCP.
    """
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        creds = service_account.Credentials.from_service_account_file(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        )
        return storage.Client(credentials=creds)
    return storage.Client()

def download_gs(uri: str) -> Path:
    """
    Descarga un archivo desde GCS a un directorio temporal.
    """
    bucket, blob = uri[5:].split("/", 1)
    local = Path(tempfile.mkdtemp()) / Path(blob).name
    gcs_client().bucket(bucket).blob(blob).download_to_filename(local)
    return local

def upload_gs(local: Path, uri: str):
    """
    Subir archivo a GCS desde una ruta local.
    """
    bucket, blob = uri[5:].split("/", 1)
    gcs_client().bucket(bucket).blob(blob).upload_from_filename(str(local))

def maybe_local(path: str) -> Path:
    """
    Verifica si la ruta es GCS o local, y descarga desde GCS si es necesario.
    """
    return download_gs(path) if path.startswith("gs://") else Path(path)

# ── CLI ────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--features", required=True, help="Ruta gs:// al parquet con los features OHLC.")
p.add_argument("--pair", required=True, help="Símbolo del par de trading (ej: EURUSD).")
p.add_argument("--timeframe", required=True, help="Timeframe de los datos (ej: 15minute).")
p.add_argument("--output", required=True, help="Ruta gs:// donde se guardará el best_params.json.")
p.add_argument("--n-trials", type=int, default=25, help="Número de trials de Optuna.")
args = p.parse_args()

PAIR, TF = args.pair, args.timeframe
tick     = 0.01 if PAIR.endswith("JPY") else 0.0001
ATR_LEN  = 14
EPOCHS_OPT, BATCH_OPT = 15, 64

# ── datos base ─────────────────────────────────────────────────
# df_raw se carga desde GCS usando maybe_local.
# Los Parquets ya vienen con 'open', 'high', 'low', 'close', 'timestamp' y otros.
df_raw = pd.read_parquet(maybe_local(args.features)).reset_index(drop=True)

# Asegurarse de que la columna 'timestamp' sea de tipo datetime, si existe.
if 'timestamp' in df_raw.columns:
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], unit='ms', errors='coerce')

# ── modelo base ────────────────────────────────────────────────
def make_model(inp_sh, lr, dr, filt, units, heads):
    inp = layers.Input(shape=inp_sh, dtype=tf.float32)
    x   = layers.Conv1D(filt, 3, padding="same", activation="relu")(inp)
    x   = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
    x   = layers.MultiHeadAttention(num_heads=heads, key_dim=units)(x, x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dropout(dr)(x)
    out = layers.Dense(2, dtype="float32")(x)
    model = models.Model(inp, out)
    model.compile(optimizers.Adam(lr), loss="mae")
    return model

# ▶▶▶ Corrección en quick_bt: nombres de parámetros actualizados ◀◀◀
def quick_bt(pred, closes, atr_pips, rr, up_thr, dn_thr, delta_min, smooth_win):
    net, pos = 0.0, False
    dq = deque(maxlen=smooth_win)
    for (u, d), price, atr in zip(pred, closes, atr_pips):
        mag, diff = max(u, d), abs(u - d)
        raw = 1 if u > d else -1
        cond = ((raw == 1 and mag >= up_thr) or (raw == -1 and mag >= dn_thr)) and diff >= delta_min
        dq.append(raw if cond else 0)
        buys, sells = dq.count(1), dq.count(-1)
        signal = 1 if buys > smooth_win // 2 else -1 if sells > smooth_win // 2 else 0
        if not pos and signal:
            pos, entry, ed = True, price, signal
            sl = (up_thr if ed == 1 else dn_thr) * atr
            tp = rr * sl
            continue
        if pos and signal:
            sl = min(sl, (up_thr if signal == 1 else dn_thr) * atr)
            tp = max(tp, rr * sl)
        if pos:
            pnl = (price - entry) / tick if ed == 1 else (entry - price) / tick
            if pnl >= tp or pnl <= -sl:
                net += tp if pnl >= tp else -sl
                pos = False
    return net

# ── Optuna objective ───────────────────────────────────────────
def objective(trial):
    pars = {
        "horizon":    trial.suggest_int("horizon", 10, 30),
        "rr":         trial.suggest_float("rr", 1.5, 3.0),
        "min_thr_up": trial.suggest_float("min_thr_up", 0.5, 2.0),
        "min_thr_dn": trial.suggest_float("min_thr_dn", 0.5, 2.0),
        "delta_min":  trial.suggest_float("delta_min", 0.01, 0.5),
        "smooth_win": trial.suggest_int("smooth_win", 1, 5),
        "win":        trial.suggest_int("win", 20, 60),
        "lr":   trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "dr":   trial.suggest_float("dr", 0.1, 0.5),
        "filt": trial.suggest_categorical("filt", [16, 32, 64]),
        "units":trial.suggest_categorical("units", [32, 64, 128]),
        "heads":trial.suggest_categorical("heads", [2, 4, 8]),
        "sma_len":  trial.suggest_categorical("sma_len",  [20, 40, 60]),
        "rsi_len":  trial.suggest_categorical("rsi_len",  [7, 14, 21]),
        "macd_fast":trial.suggest_categorical("macd_fast",[8, 12]),
        "macd_slow":trial.suggest_categorical("macd_slow",[21, 26]),
        "stoch_len":trial.suggest_categorical("stoch_len",[14, 21]),
    }

    # Construir indicadores sobre una copia de df_raw
    df = build_indicators(df_raw.copy(), pars, ATR_LEN)

    # Validación de la columna ATR antes de usarla
    atr_col = f"atr_{ATR_LEN}"
    if atr_col not in df.columns:
        print(f"Trial {trial.number}: ERROR - columna '{atr_col}' no existe. Saltando trial.")
        return -1e9

    num_nans = df[atr_col].isna().sum()
    total_rows = len(df)
    if num_nans == total_rows:
        print(f"Trial {trial.number}: ERROR - columna '{atr_col}' tiene todos NaNs ({total_rows} filas). Saltando trial.")
        return -1e9

    # Rellenar NaNs puntuales en ATR
    df[atr_col] = df[atr_col].fillna(method="bfill").fillna(method="ffill")
    atr = df[atr_col].values / tick

    # Resto de variables necesarias
    clo = df.close.values
    fut  = np.roll(clo, -pars["horizon"])
    fut[-pars["horizon"]:] = np.nan
    diff = (fut - clo) / tick
    up   = np.maximum(diff, 0) / atr
    dn   = np.maximum(-diff, 0) / atr
    mask = (~np.isnan(diff)) & (np.maximum(up, dn) >= 0)

    # Columnas de features (excluyendo ATR y timestamp)
    features_for_model = [col for col in df.columns if col not in [atr_col, 'timestamp']]
    X_raw_filtered = df.loc[mask, features_for_model]

    # Mantener solo columnas numéricas
    X_raw_filtered = X_raw_filtered.select_dtypes(include=np.number)

    if X_raw_filtered.empty or len(X_raw_filtered) < pars["win"]:
        print(f"Trial {trial.number}: X_raw_filtered vacío o menor que ventana ({len(X_raw_filtered)} < {pars['win']}). Saltando trial.")
        return -1e9

    y_up, y_dn, clo_m, atr_m = up[mask], dn[mask], clo[mask], atr[mask]

    sc  = RobustScaler()
    X_s = sc.fit_transform(X_raw_filtered)

    def seq(arr, w):
        return np.stack([arr[i-w:i] for i in range(w, len(arr))]).astype(np.float32)

    X_seq = seq(X_s, pars["win"])
    if len(X_seq) < 500:
        print(f"Trial {trial.number}: Longitud de secuencia insuficiente ({len(X_seq)}). Saltando trial.")
        return -1e6

    up_s, dn_s = y_up[pars["win"]:], y_dn[pars["win"]:]
    clo_s, atr_s = clo_m[pars["win"]:], atr_m[pars["win"]:]

    X_tr, X_val, up_tr, up_val, dn_tr, dn_val, cl_tr, cl_val, at_tr, at_val = \
        train_test_split(
            X_seq, up_s, dn_s, clo_s, atr_s,
            test_size=0.2, shuffle=False
        )

    m = make_model(
        X_tr.shape[1:], pars["lr"], pars["dr"],
        pars["filt"], pars["units"], pars["heads"]
    )
    m.fit(
        X_tr,
        np.vstack([up_tr, dn_tr]).T,
        validation_data=(X_val, np.vstack([up_val, dn_val]).T),
        epochs=EPOCHS_OPT, batch_size=BATCH_OPT,
        verbose=0,
        callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )

    score = quick_bt(
        m.predict(X_val, verbose=0),
        cl_val, at_val,
        pars["rr"], pars["min_thr_up"], pars["min_thr_dn"],
        pars["delta_min"], pars["smooth_win"]
    )

    tf.keras.backend.clear_session()
    gc.collect()
    return score

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
best = study.best_params

# Guardar best_params.json en GCS
output_gcs_path = f"{args.output}/best_params.json"
best_params_content = json.dumps({
    **best,
    "pair": PAIR,
    "timeframe": TF,
    "features_path": args.features,
    "timestamp": datetime.utcnow().isoformat()
}, indent=2)

with tempfile.TemporaryDirectory() as tmpdir:
    local_tmp_file = Path(tmpdir) / "best_params.json"
    local_tmp_file.write_text(best_params_content)
    upload_gs(local_tmp_file, output_gcs_path)

print(f"✅ best_params.json guardado en {output_gcs_path}")
