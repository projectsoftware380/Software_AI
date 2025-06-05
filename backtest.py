#!/usr/bin/env python3
"""
evaluation.backtest
────────────────────
Compara la estrategia base (solo LSTM) frente a la filtrada
(LSTM + PPO) y guarda:

    • trades_base.csv / trades_filtered.csv
    • metrics.json

Preparado para Vertex AI Custom Training (rutas gs:// o locales).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from google.cloud import storage
from google.oauth2 import service_account
from stable_baselines3 import PPO

# ---------- Configuración global ------------------------------------------------
np.random.seed(42)
tf.random.set_seed(42)

ATR_LEN   = 14
COST_PIPS = 0.8                 # comisión + spread estimado
NAN_TOL   = 0                   # no toleramos NaNs

# ---------- Indicadores ---------------------------------------------------------
from indicators import build_indicators  # mismo módulo que usa el resto del proyecto

# ---------- utilidades GCS ------------------------------------------------------
def _gcs_client() -> storage.Client:
    creds_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_file and Path(creds_file).exists():
        creds = service_account.Credentials.from_service_account_file(creds_file)
        return storage.Client(credentials=creds)
    return storage.Client()

def _download_gs(uri: str) -> Path:
    bucket_name, blob_name = uri[5:].split("/", 1)
    local_path = Path(tempfile.mkdtemp()) / Path(blob_name).name
    _gcs_client().bucket(bucket_name).blob(blob_name).download_to_filename(local_path)
    return local_path

def _upload_gs(local: Path, uri: str) -> None:
    bucket_name, blob_name = uri[5:].split("/", 1)
    _gcs_client().bucket(bucket_name).blob(blob_name).upload_from_filename(str(local))

def _maybe_local(path_or_uri: str) -> Path:
    return _download_gs(path_or_uri) if path_or_uri.startswith("gs://") else Path(path_or_uri)

# ---------- helpers de carga ----------------------------------------------------
def load_artifacts(
    lstm_model_path: str,
    lstm_scaler_path: str,
    lstm_params_path: str,
    rl_model_path: str,
) -> Tuple[tf.keras.Model, joblib, dict, PPO]:
    model   = tf.keras.models.load_model(_maybe_local(lstm_model_path), compile=False)
    scaler  = joblib.load(_maybe_local(lstm_scaler_path))
    hp      = json.loads(_maybe_local(lstm_params_path).read_text())
    ppo     = PPO.load(_maybe_local(rl_model_path))
    return model, scaler, hp, ppo

def prepare_df(features_path: str, hp: dict) -> pd.DataFrame:
    """Carga el parquet → calcula indicadores → elimina todos los NaN."""
    raw = pd.read_parquet(_maybe_local(features_path)).reset_index(drop=True)
    df  = build_indicators(raw, hp, ATR_LEN)
    df.bfill(inplace=True)
    df.dropna(inplace=True)
    if df.isna().sum().sum() > NAN_TOL:
        raise ValueError("Persisten NaNs tras bfill/dropna; abortando entrenamiento.")
    return df

# ---------- generación de secuencias -------------------------------------------
def make_sequences(
    df: pd.DataFrame,
    model: tf.keras.Model,
    scaler,
    hp: dict,
    tick: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cols_needed = list(scaler.feature_names_in_)
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise KeyError(f"Columnas faltantes en el DataFrame para el scaler: {missing}")

    X_raw  = df[cols_needed].to_numpy(dtype=np.float32)
    X_scl  = scaler.transform(X_raw)
    win    = hp["win"]
    if len(X_scl) <= win:
        raise ValueError(f"No hay suficientes filas ({len(X_scl)}) para ventanas de tamaño {win}.")
    X_seq  = np.stack([X_scl[i - win : i] for i in range(win, len(X_scl))]).astype(np.float32)

    pred   = model.predict(X_seq, verbose=0).astype(np.float32)
    up, dn = pred[:, 0], pred[:, 1]

    emb_model = tf.keras.Model(model.input, model.layers[-2].output)
    emb       = emb_model.predict(X_seq, verbose=0).astype(np.float32)

    closes = df.close.values[win:].astype(np.float32)
    atr    = df[f"atr_{ATR_LEN}"].values[win:].astype(np.float32) / tick
    return up, dn, emb, closes, atr

def to_obs(pred: np.ndarray, emb: np.ndarray, obs_dim: int) -> np.ndarray:
    need = obs_dim - pred.shape[1]
    if need <= 0:
        return pred[:, :obs_dim]          # recorte defensivo
    emb_pad = np.hstack([emb, np.zeros((len(emb), need), dtype=np.float32)])
    return np.hstack([pred, emb_pad[:, :need]])

# ---------- back-test -----------------------------------------------------------
def backtest(
    up: np.ndarray,
    dn: np.ndarray,
    closes: np.ndarray,
    atr: np.ndarray,
    accept_mask: np.ndarray,
    hp: dict,
    tick: float,
    use_filter: bool,
) -> pd.DataFrame:
    dmin, swin = hp["delta_min"], hp["smooth_win"]
    tup, tdn, rr = hp["min_thr_up"], hp["min_thr_dn"], hp["rr"]

    trades, equity = [], 0.0
    in_position   = False
    dq            = deque(maxlen=swin)

    # variables del trade abierto
    entry_price = direction = sl = tp = mfe = mae = 0.0

    for i, (u, d, price, atr_i, acc) in enumerate(zip(up, dn, closes, atr, accept_mask)):
        mag, diff = (u, d)[u < d], abs(u - d)
        raw_signal = 1 if u > d else -1
        min_thr    = tup if raw_signal == 1 else tdn
        cond       = (mag >= min_thr) and (diff >= dmin)
        dq.append(raw_signal if cond else 0)

        buys, sells = dq.count(1), dq.count(-1)
        signal = 1 if buys > swin // 2 else -1 if sells > swin // 2 else 0

        # -- gestión de posición abierta -------------------------------------
        if in_position:
            pnl = (price - entry_price) / tick if direction == 1 else (entry_price - price) / tick
            mfe = max(mfe, pnl)
            mae = min(mae, pnl)
            if pnl >= tp or pnl <= -sl:
                net = (tp if pnl >= tp else -sl) - COST_PIPS
                equity += net
                trades[-1].update(
                    exit=i,
                    pips=net,
                    eq=equity,
                    MFE_atr=mfe / atr_i,
                    MAE_atr=-mae / atr_i,
                    result="TP" if pnl >= tp else "SL",
                )
                in_position = False
            continue

        # -- apertura de nueva posición --------------------------------------
        if signal == 0 or (use_filter and not acc):
            continue

        in_position  = True
        entry_price  = price
        direction    = 1 if signal == 1 else -1
        sl           = (tup if direction == 1 else tdn) * atr_i
        tp           = rr * sl
        mfe = mae    = 0.0
        trades.append(
            dict(
                entry=i,
                dir="BUY" if direction == 1 else "SELL",
                SL_atr=sl / atr_i,
                TP_atr=rr,
                pips=0.0,
                eq=equity,
                result="OPEN",
            )
        )

    # cierre forzado al final
    if in_position:
        pnl  = (closes[-1] - entry_price) / tick if direction == 1 else (entry_price - closes[-1]) / tick
        net  = pnl - COST_PIPS
        equity += net
        trades[-1].update(
            exit=len(closes) - 1,
            pips=net,
            eq=equity,
            MFE_atr=mfe / atr[-1],
            MAE_atr=-mae / atr[-1],
            result="TP" if pnl >= 0 else "SL",
        )

    df_trades = pd.DataFrame(trades)
    if not df_trades.empty:
        df_trades.eq.ffill(inplace=True)
    return df_trades

# ---------- métricas ------------------------------------------------------------
def calc_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict.fromkeys(
            [
                "trades",
                "win_rate",
                "profit_factor",
                "expectancy",
                "net_pips",
                "sharpe",
                "sortino",
                "max_drawdown",
                "avg_mfe",
                "avg_mae",
            ],
            0.0,
        ) | {"trades": 0}

    wins   = df[df.result == "TP"]
    losses = df[df.result == "SL"]

    trades_total = len(df)
    win_rate     = len(wins) / trades_total

    total_win_pips  = wins.pips.sum()
    total_loss_pips = abs(losses.pips.sum())
    profit_factor   = total_win_pips / total_loss_pips if total_loss_pips > 0 else np.inf

    avg_win  = wins.pips.mean()  if not wins.empty   else 0.0
    avg_loss = abs(losses.pips.mean()) if not losses.empty else 0.0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    net_pips = float(df.pips.sum())

    annualizer = np.sqrt(252 * 24 * 4)  # 15-min data → 4 barras/hora
    sharpe  = df.pips.mean() / (df.pips.std() + 1e-9) * annualizer
    neg_ret = df[df.pips < 0].pips
    sortino = (
        df.pips.mean() / (neg_ret.std() + 1e-9) * annualizer
        if not neg_ret.empty
        else 0.0
    )

    max_dd = (df.eq - df.eq.cummax()).min() if "eq" in df.columns else 0.0

    return dict(
        trades=trades_total,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        net_pips=net_pips,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        avg_mfe=df.MFE_atr.mean(),
        avg_mae=df.MAE_atr.mean(),
    )

# ---------- pipeline principal ---------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser("Back-tester LSTM + PPO")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--lstm-model-path", required=True)
    parser.add_argument("--lstm-scaler-path", required=True)
    parser.add_argument("--lstm-params-path", required=True)
    parser.add_argument("--rl-model-path", required=True)
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[WARN] argumentos ignorados: {unknown}")

    tick = 0.01 if args.pair.endswith("JPY") else 0.0001

    ts = datetime.utcnow().isoformat()
    print(f"[{ts} UTC] ➜ Cargando artefactos…")
    model, scaler, hp, ppo = load_artifacts(
        args.lstm_model_path,
        args.lstm_scaler_path,
        args.lstm_params_path,
        args.rl_model_path,
    )

    print(f"[{ts} UTC] ➜ Preparando DataFrame…")
    df = prepare_df(args.features_path, hp)

    print(f"[{ts} UTC] ➜ Generando secuencias…")
    up, dn, emb, closes, atr = make_sequences(df, model, scaler, hp, tick)
    obs = to_obs(np.column_stack([up, dn]), emb, ppo.observation_space.shape[0])
    accept_mask, _ = ppo.predict(obs, deterministic=True)
    accept_mask = accept_mask.astype(bool)

    print(f"[{ts} UTC] ➜ Ejecutando back-tests…")
    base_trades = backtest(up, dn, closes, atr, accept_mask, hp, tick, use_filter=False)
    filt_trades = backtest(up, dn, closes, atr, accept_mask, hp, tick, use_filter=True)

    metrics_base = calc_metrics(base_trades)
    metrics_filt = calc_metrics(filt_trades)

    # -------- persistir resultados ----------------------------------------
    out_dir_local = Path(tempfile.mkdtemp())
    paths = {}
    if not base_trades.empty:
        paths["trades_base.csv"] = out_dir_local / "trades_base.csv"
        base_trades.to_csv(paths["trades_base.csv"], index=False)
    if not filt_trades.empty:
        paths["trades_filtered.csv"] = out_dir_local / "trades_filtered.csv"
        filt_trades.to_csv(paths["trades_filtered.csv"], index=False)
    paths["metrics.json"] = out_dir_local / "metrics.json"
    json.dump({"base": metrics_base, "filtered": metrics_filt}, open(paths["metrics.json"], "w"), indent=2)

    # subir o mover
    if args.output_dir.startswith("gs://"):
        prefix = args.output_dir.rstrip("/") + "/"
        for fname, lpath in paths.items():
            _upload_gs(lpath, prefix + fname)
        print(f"[{ts} UTC] ✔ Resultados subidos a {prefix}")
    else:
        dest = Path(args.output_dir)
        dest.mkdir(parents=True, exist_ok=True)
        for fname, lpath in paths.items():
            shutil.move(lpath, dest / fname)
        print(f"[{ts} UTC] ✔ Resultados guardados en {dest}")

    shutil.rmtree(out_dir_local, ignore_errors=True)
    print(f"[{datetime.utcnow().isoformat()} UTC] ✅ Back-test completado sin errores.")

# -------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
