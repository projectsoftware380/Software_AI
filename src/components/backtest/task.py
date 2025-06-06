# src/components/backtest/task.py (CORREGIDO)
"""
Tarea del componente de backtesting y evaluación de modelos.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from stable_baselines3 import PPO

from src.shared import constants, gcs_utils, indicators

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ATR_LEN = 14
COST_PIPS = 0.8

def _log_kfp_metrics(metrics_path: Path, metrics_data: dict):
    """Escribe las métricas en un archivo JSON para que KFP las visualice."""
    kfp_metrics = {"metrics": []}
    
    # Usamos las métricas del modelo filtrado (LSTM+PPO) para el reporte principal
    filtered_metrics = metrics_data.get("filtered", {})

    for key, value in filtered_metrics.items():
        # Asegurarse de que el valor sea un número
        if not isinstance(value, (int, float)):
            continue
            
        # Determinar el formato
        metric_format = "PERCENTAGE" if "rate" in key else "RAW"
        
        kfp_metrics["metrics"].append({
            "name": key.replace("_", "-"),  # KFP prefiere nombres con guiones
            "numberValue": round(value, 5),
            "format": metric_format,
        })
    
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(kfp_metrics, f)
    logger.info(f"Métricas de KFP guardadas en: {metrics_path}")


def _load_artifacts(
    lstm_model_dir: str, rl_model_path: str
) -> Tuple[tf.keras.Model, any, dict, PPO]:
    """Carga todos los artefactos de modelo necesarios desde GCS."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        logger.info(f"Descargando artefactos a directorio temporal: {tmp_path}")
        
        model_path = gcs_utils.download_gcs_file(f"{lstm_model_dir}/model.h5", tmp_path)
        scaler_path = gcs_utils.download_gcs_file(f"{lstm_model_dir}/scaler.pkl", tmp_path)
        params_path = gcs_utils.download_gcs_file(f"{lstm_model_dir}/params.json", tmp_path)
        ppo_model_path = gcs_utils.download_gcs_file(rl_model_path, tmp_path)
        
        model = tf.keras.models.load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        hp = json.loads(params_path.read_text())
        ppo = PPO.load(ppo_model_path)
        
    logger.info("✔ Todos los artefactos han sido cargados.")
    return model, scaler, hp, ppo


def _prepare_backtest_data(features_path: str, hp: dict) -> pd.DataFrame:
    """Carga y prepara el DataFrame de backtesting."""
    logger.info(f"Cargando datos para backtest desde: {features_path}")
    local_features_path = gcs_utils.ensure_gcs_path_and_get_local(features_path)
    df_raw = pd.read_parquet(local_features_path).reset_index(drop=True)
    
    logger.info("Calculando indicadores para los datos de backtest...")
    df_ind = indicators.build_indicators(df_raw, hp, atr_len=ATR_LEN, drop_na=True)
    
    if df_ind.isna().any().any():
        raise ValueError("Persisten NaNs en los datos de backtest tras la limpieza.")
    
    logger.info(f"✔ Datos de backtest preparados con {len(df_ind):,} filas.")
    return df_ind


def _generate_predictions(
    df: pd.DataFrame, model: tf.keras.Model, scaler, hp: dict, tick: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Genera predicciones y otros arrays necesarios para el backtest."""
    cols_needed = list(scaler.feature_names_in_)
    X_raw = df[cols_needed].to_numpy(dtype=np.float32)
    X_scl = scaler.transform(X_raw)
    win = hp["win"]
    
    if len(X_scl) <= win:
        raise ValueError(f"No hay suficientes filas ({len(X_scl)}) para ventanas de tamaño {win}.")
        
    X_seq = np.stack([X_scl[i - win : i] for i in range(win, len(X_scl))]).astype(np.float32)

    logger.info("Generando predicciones y embeddings del modelo LSTM...")
    pred = model.predict(X_seq, verbose=0).astype(np.float32)
    up, dn = pred[:, 0], pred[:, 1]

    emb_model = tf.keras.Model(model.input, model.layers[-2].output)
    emb = emb_model.predict(X_seq, verbose=0).astype(np.float32)

    closes = df.close.values[win:].astype(np.float32)
    atr = df[f"atr_{ATR_LEN}"].values[win:].astype(np.float32) / tick
    return up, dn, emb, closes, atr


def _run_backtest_simulation(
    up: np.ndarray, dn: np.ndarray, closes: np.ndarray, atr: np.ndarray,
    accept_mask: np.ndarray, hp: dict, tick: float, use_filter: bool
) -> pd.DataFrame:
    """Motor de simulación de backtesting."""
    dmin, swin, tup, tdn, rr = hp["delta_min"], hp["smooth_win"], hp["min_thr_up"], hp["min_thr_dn"], hp["rr"]
    trades, equity, in_position = [], 0.0, False
    entry_price, direction, sl, tp, mfe, mae = 0.0, 0, 0.0, 0.0, 0.0, 0.0
    dq = deque(maxlen=swin)

    for i, (u, d, price, atr_i, acc) in enumerate(zip(up, dn, closes, atr, accept_mask)):
        mag, diff = max(u, d), abs(u - d)
        raw_signal, min_thr = (1, tup) if u > d else (-1, tdn)
        cond = (mag >= min_thr) and (diff >= dmin)
        dq.append(raw_signal if cond else 0)
        
        buys, sells = dq.count(1), dq.count(-1)
        signal = 1 if buys > swin // 2 else -1 if sells > swin // 2 else 0

        if in_position:
            pnl = ((price - entry_price) / tick) * direction
            mfe, mae = max(mfe, pnl), min(mae, pnl)
            if pnl >= tp or pnl <= -sl:
                net = (tp if pnl >= tp else -sl) - COST_PIPS
                equity += net
                trades[-1].update(exit=i, pips=net, eq=equity, MFE_atr=mfe / atr_i, MAE_atr=-mae / atr_i, result="TP" if pnl >= tp else "SL")
                in_position = False
            continue

        if signal != 0 and (not use_filter or acc):
            in_position, entry_price, direction = True, price, signal
            sl, tp, mfe, mae = min_thr * atr_i, rr * (min_thr * atr_i), 0.0, 0.0
            trades.append(dict(entry=i, dir="BUY" if direction == 1 else "SELL", SL_atr=sl / atr_i, TP_atr=rr, pips=0.0, eq=equity, result="OPEN"))

    if in_position:
        pnl = ((closes[-1] - entry_price) / tick) * direction - COST_PIPS
        equity += pnl
        trades[-1].update(exit=len(closes) - 1, pips=pnl, eq=equity, MFE_atr=mfe/atr[-1], MAE_atr=-mae/atr[-1], result="CLOSE")
        
    return pd.DataFrame(trades)


def _calculate_metrics(df: pd.DataFrame) -> dict:
    """Calcula las métricas de rendimiento a partir de un DataFrame de trades."""
    if df.empty or len(df[df.result != 'OPEN']) == 0:
        return {k: 0.0 for k in ["trades", "win_rate", "profit_factor", "expectancy", "net_pips", "sharpe", "sortino", "max_drawdown", "avg_mfe", "avg_mae"]}

    df_closed = df[df.result.isin(["TP", "SL", "CLOSE"])].copy()
    wins = df_closed[df_closed.pips > 0]
    losses = df_closed[df_closed.pips <= 0]
    
    total_win_pips = wins.pips.sum()
    total_loss_pips = abs(losses.pips.sum())
    
    annualizer = np.sqrt(252 * 24 * 4)  # Para datos de 15min
    
    return {
        "trades": float(len(df_closed)),
        "win_rate": float(len(wins) / len(df_closed)) if len(df_closed) > 0 else 0.0,
        "profit_factor": float(total_win_pips / total_loss_pips) if total_loss_pips > 0 else np.inf,
        "expectancy": float(df_closed.pips.mean()),
        "net_pips": float(df_closed.pips.sum()),
        "sharpe": float(df_closed.pips.mean() / (df_closed.pips.std() + 1e-9)) * annualizer,
        "sortino": float(df_closed.pips.mean() / (df_closed[df_closed.pips < 0].pips.std() + 1e-9)) * annualizer if len(losses) > 0 else np.inf,
        "max_drawdown": float((df_closed.eq - df_closed.eq.cummax()).min()) if not df_closed.eq.empty else 0.0,
        "avg_mfe": float(df_closed.MFE_atr.mean()),
        "avg_mae": float(df_closed.MAE_atr.mean()),
    }

def run_backtest(
    lstm_model_dir: str, rl_model_path: str, features_path: str, pair: str, output_gcs_dir: str, kfp_metrics_path: Path
) -> None:
    """Orquesta el proceso completo de backtesting."""
    tick = 0.01 if pair.endswith("JPY") else 0.0001
    model, scaler, hp, ppo = _load_artifacts(lstm_model_dir, rl_model_path)
    df_backtest = _prepare_backtest_data(features_path, hp)
    
    up, dn, emb, closes, atr = _generate_predictions(df_backtest, model, scaler, hp, tick)
    obs = np.hstack([np.column_stack([up, dn]), emb])
    accept_mask, _ = ppo.predict(obs, deterministic=True)
    
    logger.info("Ejecutando simulación para la estrategia BASE (solo LSTM)...")
    trades_base = _run_backtest_simulation(up, dn, closes, atr, accept_mask, hp, tick, use_filter=False)
    
    logger.info("Ejecutando simulación para la estrategia FILTRADA (LSTM + PPO)...")
    trades_filtered = _run_backtest_simulation(up, dn, closes, atr, accept_mask, hp, tick, use_filter=True)
    
    metrics_base = _calculate_metrics(trades_base)
    metrics_filtered = _calculate_metrics(trades_filtered)
    
    all_metrics = {"base": metrics_base, "filtered": metrics_filtered}
    logger.info(f"Métricas Filtradas: {json.dumps(metrics_filtered, indent=2)}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        if not trades_base.empty:
            trades_base.to_csv(tmp_path / "trades_base.csv", index=False)
            gcs_utils.upload_gcs_file(tmp_path / "trades_base.csv", f"{output_gcs_dir}/trades_base.csv")
        if not trades_filtered.empty:
            trades_filtered.to_csv(tmp_path / "trades_filtered.csv", index=False)
            gcs_utils.upload_gcs_file(tmp_path / "trades_filtered.csv", f"{output_gcs_dir}/trades_filtered.csv")
        
        metrics_path = tmp_path / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        gcs_utils.upload_gcs_file(metrics_path, f"{output_gcs_dir}/metrics.json")
    
    # Escribir las métricas para KFP
    _log_kfp_metrics(kfp_metrics_path, all_metrics)
        
    logger.info(f"🎉 Tarea completada. Resultados de backtest disponibles en: {output_gcs_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task de Backtesting de Estrategias.")
    parser.add_argument("--lstm-model-dir", required=True)
    parser.add_argument("--rl-model-path", required=True)
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    # Argumento AÑADIDO para recibir la ruta del artefacto de métricas de KFP
    parser.add_argument("--kfp-metrics-path", type=Path, required=True, help="Path to write the KFP Metrics artifact.")
    
    args = parser.parse_args()

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    output_directory = f"{constants.BACKTEST_RESULTS_PATH}/{args.pair}/{args.timeframe}/{timestamp}"
    
    run_backtest(
        lstm_model_dir=args.lstm_model_dir,
        rl_model_path=args.rl_model_path,
        features_path=args.features_path,
        pair=args.pair,
        output_gcs_dir=output_directory,
        kfp_metrics_path=args.kfp_metrics_path,
    )
    
    print(output_directory)