#!/usr/bin/env python3
"""
evaluation.backtest
────────────────────
Compara estrategia base (LSTM) vs. filtrada (LSTM + PPO) y guarda:
    • trades_base.csv / trades_filtered.csv
    • metrics.json

Diseñado para ejecutarse en entornos de GCP (Vertex AI Custom Training).
Las rutas de entrada y salida son compatibles con gs://.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import warnings
import tempfile # Importado para manejo de archivos temporales

# ---------------- imports principales ------------------------------------
import numpy as np
# Parche NumPy ≥1.24
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import pandas as pd
import pandas_ta as ta
import joblib
import tensorflow as tf
from stable_baselines3 import PPO
from collections import deque

# Importar funciones de GCS y el cliente de Google Cloud Storage
from google.cloud import storage
from google.oauth2 import service_account

# Importa build_indicators (asumiendo que core.indicators está disponible en la imagen Docker)
from core.indicators import build_indicators

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- reproducibilidad ---------------------------------------
np.random.seed(42)
tf.random.set_seed(42)

# ---------------- constantes ---------------------------------------------
ATR_LEN = 14
COST_PIPS = 0.8  # comisiones/spread estimado

# ---------------- helpers GCS para el backtest ----------------------------

def gcs_client():
    """
    Si GOOGLE_APPLICATION_CREDENTIALS está definida, utiliza esas credenciales
    para acceder a Google Cloud Storage.
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
    Sube un archivo desde una ruta local a GCS.
    """
    bucket, blob = uri[5:].split("/", 1)
    gcs_client().bucket(bucket).blob(blob).upload_from_filename(str(local))

def maybe_local(path_or_uri: str) -> Path:
    """
    Verifica si la ruta es GCS o local, y descarga desde GCS si es necesario.
    """
    return download_gs(path_or_uri) if path_or_uri.startswith("gs://") else Path(path_or_uri)

# ---------------- helpers de carga de artefactos -------------------------

def load_artifacts(lstm_model_path: str, lstm_scaler_path: str, lstm_params_path: str, rl_model_path: str):
    """
    Carga los artefactos del modelo LSTM y el modelo PPO desde rutas (locales o GCS).
    """
    model = tf.keras.models.load_model(maybe_local(lstm_model_path), compile=False)
    scaler = joblib.load(maybe_local(lstm_scaler_path))
    hp = json.loads(maybe_local(lstm_params_path).read_text())
    ppo = PPO.load(maybe_local(rl_model_path))
    return model, scaler, hp, ppo

def prepare_df(features_path: str, hp: dict):
    """
    Carga el DataFrame de features y calcula los indicadores.
    """
    raw = pd.read_parquet(maybe_local(features_path)).reset_index(drop=True)
    ind = build_indicators(raw, hp, ATR_LEN)
    ind.bfill(inplace=True)
    return ind

def sequences(df: pd.DataFrame, model, scaler, hp: dict, tick: float):
    X_raw = df[scaler.feature_names_in_].values
    X = scaler.transform(X_raw)
    win = hp["win"]
    X_seq = np.stack([X[i-win:i] for i in range(win, len(X))]).astype(np.float32)

    pred = model.predict(X_seq, verbose=0).astype(np.float32)
    up, dn = pred[:,0], pred[:,1]

    emb = tf.keras.Model(model.input, model.layers[-2].output) \
            .predict(X_seq, verbose=0).astype(np.float32)

    closes  = df.close.values[win:]
    atr_arr = df[f"atr_{ATR_LEN}"].values[win:] / tick
    return up, dn, emb, closes, atr_arr

def to_obs(pred, emb, obs_dim):
    need = obs_dim - pred.shape[1]
    emb_pad = np.hstack([emb, np.zeros((len(emb), max(0, need)), dtype=np.float32)])
    return np.hstack([pred, emb_pad[:, :need]])

# ---------------- back-test logic ------------------------------------------
def backtest(up, dn, closes, atr, mask, hp, tick, use_filter):
    dmin, swin = hp["delta_min"], hp["smooth_win"]
    tup, tdn, rr = hp["min_thr_up"], hp["min_thr_dn"], hp["rr"]

    trades, eq, pos = [], 0.0, False
    dq = deque(maxlen=swin)
    for i,(u,d,p,atr_i,acc) in enumerate(zip(up,dn,closes,atr,mask)):
        mag, diff = max(u,d), abs(u-d)
        raw = 1 if u>d else -1
        cond = ((raw==1 and mag>=tup) or (raw==-1 and mag>=tdn)) and diff>=dmin
        dq.append(raw if cond else 0)
        buys, sells = dq.count(1), dq.count(-1)
        sig = 1 if buys>swin//2 else -1 if sells>swin//2 else 0

        if pos:  # gestionar trade abierto
            pnl = (p-entry)/tick if dir=="BUY" else (entry-p)/tick
            mfe = max(mfe, pnl) # Max Favorable Excursion
            mae = min(mae, pnl) # Max Adverse Excursion
            if pnl>=tp or pnl<=-sl:
                net = (tp if pnl>=tp else -sl) - COST_PIPS
                eq += net
                trades[-1].update({"exit":i,"pips":net,"eq":eq,
                               "MFE_atr":mfe/atr_i,"MAE_atr":-mae/atr_i,
                               "result":"TP" if pnl>=tp else "SL"})
                pos = False
            continue

        if sig==0 or (use_filter and not acc):  # sin apertura
            continue
        # abrir trade
        pos, entry, dir = True, p, "BUY" if sig==1 else "SELL"
        sl = (tup if sig==1 else tdn)*atr_i
        tp = rr*sl
        mfe = mae = 0.0
        trades.append({"entry":i,"dir":dir,"SL_atr":sl/atr_i,"TP_atr":rr,
                       "pips":0.0,"eq":eq,"result":"OPEN"})

    # cierre al final (si hay un trade abierto al final de los datos)
    if pos:
        pnl = (closes[-1]-entry)/tick if dir=="BUY" else (entry-closes[-1])/tick
        net = pnl - COST_PIPS
        eq += net
        trades[-1].update({"exit":len(closes)-1,"pips":net,"eq":eq,
                           "MFE_atr":mfe/atr[-1],"MAE_atr":-mae/atr[-1],
                           "result":"TP" if pnl>=0 else "SL"})

    out = pd.DataFrame(trades)
    if not out.empty: out.eq = out.eq.ffill() # Asegura que la curva de equity no tenga NaNs
    return out

def metrics(df: pd.DataFrame):
    if df.empty:
        return {
            "trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "expectancy": 0.0, "net_pips": 0.0, "sharpe": 0.0,
            "sortino": 0.0, "max_drawdown": 0.0, "avg_mfe": 0.0,
            "avg_mae": 0.0
        }
    
    wins = df[df.result=="TP"]
    losses = df[df.result=="SL"]
    
    total_trades = len(df)
    win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
    
    total_pips_wins = wins.pips.sum()
    total_pips_losses = abs(losses.pips.sum())
    
    profit_factor = total_pips_wins / total_pips_losses if total_pips_losses > 0 else np.inf
    
    avg_win = wins.pips.mean() if len(wins) > 0 else 0.0
    avg_loss = abs(losses.pips.mean()) if len(losses) > 0 else 0.0
    
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    net_pips = float(df.pips.sum())
    
    # Cálculos para Sharpe y Sortino Ratio (ajuste para series vacías o std dev 0)
    sharpe = df.pips.mean() / (df.pips.std() + 1e-9) * np.sqrt(252 * 24 * 4) # Asumiendo datos de 15min (4 barras por hora, 24h, 252 días de trading)
    
    negative_returns = df[df.pips < 0].pips
    sortino = df.pips.mean() / (negative_returns.std() + 1e-9) * np.sqrt(252 * 24 * 4) if len(negative_returns) > 0 else 0.0

    dd = (df.eq - df.eq.cummax()).min() if not df.empty else 0.0
    
    avg_mfe = df.MFE_atr.mean() if not df.empty else 0.0
    avg_mae = df.MAE_atr.mean() if not df.empty else 0.0
    
    return dict(
        trades=total_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        net_pips=net_pips,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=dd,
        avg_mfe=avg_mfe,
        avg_mae=avg_mae
    )


# ---------------- main ----------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Realiza backtesting de una estrategia de trading (LSTM + PPO).")
    ap.add_argument("--pair", required=True, help="Símbolo del par de trading (ej: EURUSD)")
    ap.add_argument("--timeframe", required=True, help="Timeframe de los datos (ej: 15minute)")
    ap.add_argument("--lstm-model-path", required=True, help="Ruta (gs:// o local) al archivo .h5 del modelo LSTM.")
    ap.add_argument("--lstm-scaler-path", required=True, help="Ruta (gs:// o local) al archivo .pkl del scaler del LSTM.")
    ap.add_argument("--lstm-params-path", required=True, help="Ruta (gs:// o local) al archivo .json de hiperparámetros del LSTM.")
    ap.add_argument("--rl-model-path", required=True, help="Ruta (gs:// o local) al archivo .zip del modelo PPO (RL).")
    ap.add_argument("--features-path", required=True, help="Ruta (gs:// o local) al archivo .parquet con los features para backtesting.")
    ap.add_argument("--output-dir", required=True, help="Carpeta (gs:// o local) donde guardar los resultados del backtesting.")
    args = ap.parse_args()

    tick = 0.01 if args.pair.endswith("JPY") else 0.0001

    # 1. Cargar artefactos y datos
    print(f"[{datetime.utcnow().isoformat()} UTC] Cargando artefactos y datos...")
    model, scaler, hp, ppo = load_artifacts(
        args.lstm_model_path,
        args.lstm_scaler_path,
        args.lstm_params_path,
        args.rl_model_path
    )
    df = prepare_df(args.features_path, hp)

    # 2. Generar secuencias y observaciones para el backtest
    up, dn, emb, closes, atr = sequences(df, model, scaler, hp, tick)
    obs = to_obs(np.column_stack([up, dn]), emb, ppo.observation_space.shape[0])
    actions, _ = ppo.predict(obs, deterministic=True)
    accept = actions.astype(bool)

    # 3. Realizar back-tests
    print(f"[{datetime.utcnow().isoformat()} UTC] Realizando backtests...")
    base_tr = backtest(up, dn, closes, atr, accept, hp, tick, False)
    filt_tr = backtest(up, dn, closes, atr, accept, hp, tick, True)

    # 4. Calcular métricas
    m_base = metrics(base_tr)
    m_filt = metrics(filt_tr)

    # 5. Imprimir métricas (para logs en GCP)
    print(f"[{datetime.utcnow().isoformat()} UTC] Métricas de Backtest:")
    # print_metrics("BASE", m_base) # No es necesario en producción, solo para depuración
    # print_metrics("FILTRADA", m_filt) # No es necesario en producción, solo para depuración
    print(json.dumps({"base_metrics": m_base, "filtered_metrics": m_filt}, indent=2))


    # 6. Guardar resultados
    print(f"[{datetime.utcnow().isoformat()} UTC] Guardando resultados...")
    
    # Crear un directorio temporal para guardar los archivos antes de subirlos a GCS
    temp_local_dir = Path(tempfile.mkdtemp())
    
    # Rutas locales temporales
    trades_base_path = temp_local_dir / "trades_base.csv"
    trades_filtered_path = temp_local_dir / "trades_filtered.csv"
    metrics_json_path = temp_local_dir / "metrics.json"

    # Guardar localmente los DataFrames y JSON
    if not base_tr.empty:
        base_tr.to_csv(trades_base_path, index=False)
    else:
        print("Advertencia: No se generaron trades para la estrategia BASE.")
        trades_base_path = None # Marcar como no disponible

    if not filt_tr.empty:
        filt_tr.to_csv(trades_filtered_path, index=False)
    else:
        print("Advertencia: No se generaron trades para la estrategia FILTRADA.")
        trades_filtered_path = None # Marcar como no disponible

    json.dump({"base": m_base, "filtered": m_filt},
              open(metrics_json_path, "w"), indent=2)

    # Determinar si la salida es local o GCS
    if args.output_dir.startswith("gs://"):
        gcs_output_prefix = args.output_dir.rstrip('/') + '/'
        if trades_base_path:
            upload_gs(trades_base_path, gcs_output_prefix + "trades_base.csv")
        if trades_filtered_path:
            upload_gs(trades_filtered_path, gcs_output_prefix + "trades_filtered.csv")
        upload_gs(metrics_json_path, gcs_output_prefix + "metrics.json")
        print(f"[{datetime.utcnow().isoformat()} UTC] Resultados subidos a {gcs_output_prefix}")
    else:
        local_output_dir = Path(args.output_dir)
        local_output_dir.mkdir(parents=True, exist_ok=True)
        if trades_base_path:
            trades_base_path.replace(local_output_dir / "trades_base.csv")
        if trades_filtered_path:
            trades_filtered_path.replace(local_output_dir / "trades_filtered.csv")
        metrics_json_path.replace(local_output_dir / "metrics.json")
        print(f"[{datetime.utcnow().isoformat()} UTC] Resultados guardados localmente en {local_output_dir}")

    # Limpiar el directorio temporal
    import shutil
    shutil.rmtree(temp_local_dir)


    print(f"[{datetime.utcnow().isoformat()} UTC] ✅ Back-test finalizado.")

if __name__ == "__main__":
    main()