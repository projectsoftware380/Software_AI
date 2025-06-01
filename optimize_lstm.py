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
        # Corrección: usar up_thr, dn_thr, delta_min
        cond = ((raw == 1 and mag >= up_thr) or (raw == -1 and mag >= dn_thr)) and diff >= delta_min
        dq.append(raw if cond else 0)
        buys, sells = dq.count(1), dq.count(-1)
        signal = 1 if buys > smooth_win // 2 else -1 if sells > smooth_win // 2 else 0
        if not pos and signal:
            pos, entry, ed = True, price, signal
            # Corrección: usar up_thr, dn_thr
            sl = (up_thr if ed == 1 else dn_thr) * atr
            tp = rr * sl
            continue
        if pos and signal:
            # Corrección: usar up_thr, dn_thr
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

    # ▶▶▶ Inicio de la corrección para TypeError en columna ATR ◀◀◀
    atr_col = f"atr_{ATR_LEN}"
    if atr_col not in df.columns:
        print(f"Trial {trial.number}: ERROR - La columna ATR '{atr_col}' no existe en el DataFrame después de build_indicators. Saltando trial.")
        return -1e9 # Retornar valor muy bajo

    # Verificar si la columna está completamente vacía (None) o llena de NaNs
    if df[atr_col].isnull().all():
        print(f"Trial {trial.number}: ERROR - La columna ATR '{atr_col}' está completamente llena de NaNs ({len(df[atr_col])} valores). Saltando trial.")
        return -1e9 # Retornar valor muy bajo
    
    num_nans_before_fill = df[atr_col].isna().sum()
    if num_nans_before_fill > 0:
        print(f"Trial {trial.number}: INFO - La columna ATR '{atr_col}' contiene {num_nans_before_fill} NaNs antes del rellenado.")

    # Rellenar NaNs puntuales en ATR (bfill y luego ffill para cubrir bordes)
    df[atr_col] = df[atr_col].fillna(method="bfill").fillna(method="ffill")

    # Verificar si después del rellenado aún hay NaNs (esto podría ocurrir si toda la columna era NaN)
    if df[atr_col].isnull().any():
        print(f"Trial {trial.number}: ERROR - La columna ATR '{atr_col}' todavía contiene NaNs después de intentar rellenar. Esto indica que la columna podría haber estado completamente vacía o tener problemas estructurales. Saltando trial.")
        return -1e9

    print(f"Trial {trial.number}: INFO - Columna ATR '{atr_col}' validada y rellenada. NaNs restantes: {df[atr_col].isna().sum()}.")
    atr = df[atr_col].values / tick
    # ▶▶▶ Fin de la corrección para TypeError en columna ATR ◀◀◀

    # Resto de variables necesarias
    clo = df.close.values
    fut  = np.roll(clo, -pars["horizon"])
    fut[-pars["horizon"]:] = np.nan
    diff = (fut - clo) / tick
    up   = np.maximum(diff, 0) / atr
    dn   = np.maximum(-diff, 0) / atr
    mask = (~np.isnan(diff)) & (np.maximum(up, dn) >= 0) & (~np.isnan(atr)) # Añadida condición ~np.isnan(atr)

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
        # Asegurar que haya suficientes datos para la primera secuencia
        if len(arr) < w:
            return np.array([]) # Retornar array vacío si no hay suficientes datos
        return np.stack([arr[i-w:i] for i in range(w, len(arr))]).astype(np.float32)

    X_seq = seq(X_s, pars["win"])
    
    # Verificar si X_seq está vacío después de la creación de secuencias
    if X_seq.shape[0] == 0:
        print(f"Trial {trial.number}: No se pudieron crear secuencias (X_seq está vacío). Probablemente datos insuficientes después del enmascaramiento y antes de la secuenciación. Saltando trial.")
        return -1e9 # Usar -1e9 para consistencia con otros errores de datos

    if len(X_seq) < 500: # Este umbral podría ser ajustado
        print(f"Trial {trial.number}: Longitud de secuencia insuficiente ({len(X_seq)} después de seq). Saltando trial.")
        return -1e6 # Puede ser un valor diferente para distinguir de otros errores

    # Ajustar los slices para que coincidan con la longitud de X_seq
    # El slicing [pars["win"]:] original podría ser incorrecto si seq() ya manejó la ventana.
    # La función seq ya crea secuencias a partir de la ventana 'w', por lo que los arrays 'y' deben alinearse con la salida de seq.
    # Si seq devuelve N secuencias, y_up, y_dn, etc., deben tener N elementos correspondientes a la etiqueta de la *última* observación de cada secuencia.
    # El primer índice válido después de seq(arr, w) es len(arr) - w.
    # Los targets y_up, y_dn deben ser seleccionados desde el índice (pars["win"] -1) hasta (len(mask) -1), y luego tomar los primeros len(X_seq) elementos.
    
    # Correcta alineación de las etiquetas y datos auxiliares con X_seq
    # y_up, y_dn, clo_m, atr_m son los datos después del enmascaramiento inicial.
    # X_s es la versión escalada de X_raw_filtered (que ya está enmascarada).
    # seq(X_s, pars["win"]) crea secuencias. Si X_s tiene M filas, seq crea M - pars["win"] + 1 secuencias.
    # Las etiquetas deben corresponder al final de cada ventana.
    # Entonces, si X_seq[0] usa X_s[0:pars["win"]], la etiqueta es y_up[pars["win"]-1] (si el target es para el último paso de la ventana)
    # O y_up[pars["win"]] si el target es para el paso *siguiente* a la ventana.
    # Dado que 'fut' se calcula con np.roll(clo, -pars["horizon"]), 'diff' y por ende 'up'/'dn' son targets *futuros*.
    # El target para la secuencia X_s[i:i+pars["win"]] es y_up[i+pars["win"]-1] (asumiendo que y_up está alineado con X_s).
    
    # Si X_s tiene N_s filas, X_seq tendrá N_s - pars["win"] + 1 filas.
    # Las etiquetas y_up, y_dn, etc., ya están enmascaradas y alineadas con X_s.
    # Necesitamos tomar las etiquetas desde el final de la primera ventana hasta el final.
    start_index_for_labels = pars["win"] - 1
    end_index_for_labels = len(y_up) # o len(X_s)

    # Esta es la forma más común: la etiqueta corresponde al paso *después* de la ventana
    # up_s = y_up[pars["win"] -1 : len(X_seq) + pars["win"] -1]
    # dn_s = y_dn[pars["win"] -1 : len(X_seq) + pars["win"] -1]
    # clo_s = clo_m[pars["win"] -1 : len(X_seq) + pars["win"] -1]
    # atr_s = atr_m[pars["win"] -1 : len(X_seq) + pars["win"] -1]

    # Sin embargo, el código original usaba `up_s, dn_s = y_up[pars["win"]:], y_dn[pars["win"]:]`
    # Esto implica que si X_seq tiene L elementos, los targets son y_up[pars["win"]:pars["win"]+L]
    # Vamos a mantener la lógica original lo más posible, pero asegurando que los índices sean válidos.
    
    num_sequences = X_seq.shape[0]
    if num_sequences == 0:
        print(f"Trial {trial.number}: X_seq está vacío, no se pueden generar etiquetas. Saltando trial.")
        return -1e9

    # Las etiquetas y datos auxiliares deben tener la misma longitud que X_seq
    # El primer target y_up[pars["win"]] corresponde a la secuencia X_s[0:pars["win"]]
    # El último target y_up[pars["win"] + num_sequences - 1] corresponde a X_s[num_sequences-1 : num_sequences-1+pars["win"]]
    
    if len(y_up) < pars["win"] + num_sequences:
        print(f"Trial {trial.number}: No hay suficientes datos en y_up para alinear con X_seq. y_up len: {len(y_up)}, required: {pars['win'] + num_sequences}. Saltando trial.")
        return -1e9
        
    up_s = y_up[pars["win"] : pars["win"] + num_sequences]
    dn_s = y_dn[pars["win"] : pars["win"] + num_sequences]
    clo_s = clo_m[pars["win"] : pars["win"] + num_sequences]
    atr_s = atr_m[pars["win"] : pars["win"] + num_sequences]


    # Validación adicional de longitudes
    if not (len(X_seq) == len(up_s) == len(dn_s) == len(clo_s) == len(atr_s)):
        print(f"Trial {trial.number}: Desajuste de longitud después de crear secuencias y etiquetas.")
        print(f"X_seq: {len(X_seq)}, up_s: {len(up_s)}, dn_s: {len(dn_s)}, clo_s: {len(clo_s)}, atr_s: {len(atr_s)}")
        return -1e9


    X_tr, X_val, up_tr, up_val, dn_tr, dn_val, cl_tr, cl_val, at_tr, at_val = \
        train_test_split(
            X_seq, up_s, dn_s, clo_s, atr_s,
            test_size=0.2, shuffle=False
        )

    if len(X_tr) == 0 or len(X_val) == 0:
        print(f"Trial {trial.number}: Conjunto de entrenamiento o validación vacío después del split. X_tr: {len(X_tr)}, X_val: {len(X_val)}. Saltando trial.")
        return -1e9


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
# La ruta de salida ahora es directamente el archivo JSON, no un directorio.
# args.output debe ser la ruta completa al archivo JSON, ej: gs://bucket/path/to/best_params.json
output_gcs_path = args.output # Se asume que args.output ya es la ruta completa al archivo.

best_params_content = json.dumps({
    **best,
    "pair": PAIR,
    "timeframe": TF,
    "features_path": args.features,
    "timestamp": datetime.utcnow().isoformat()
}, indent=2)

with tempfile.TemporaryDirectory() as tmpdir:
    # El nombre del archivo local es solo 'best_params.json' si output_gcs_path es el nombre del archivo.
    # Si output_gcs_path es un directorio, entonces Path(output_gcs_path).name sería incorrecto.
    # Asumimos que el nombre del archivo en GCS es el deseado.
    local_tmp_file_name = Path(output_gcs_path).name # e.g., "best_params.json"
    local_tmp_file = Path(tmpdir) / local_tmp_file_name
    local_tmp_file.write_text(best_params_content)
    upload_gs(local_tmp_file, output_gcs_path) # output_gcs_path es la URI completa del archivo

print(f"✅ best_params.json guardado en {output_gcs_path}")