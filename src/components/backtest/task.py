# src/components/backtest/task.py
"""
Ejecuta el backtest (LSTM + filtro PPO) y sube resultados a GCS.
Genera:
    • /tmp/backtest_dir.txt  → carpeta final en GCS (para KFP)
    • /tmp/kfp_metrics.json  → métricas en formato Vertex UI
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from google.cloud import storage
from stable_baselines3 import PPO

from src.shared import constants, gcs_utils, indicators

# ─────────────────────────── logging ────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ────────────────── backtest constantes globales ─────────────────
ATR_LEN = 14
COST_PIPS = 0.8          # comisión para _run_backtest_simulation

# ───────────────────── helpers utilitarios ──────────────────────
def resolve_artifact_dir(base_dir: str) -> str:
    """
    Devuelve la carpeta que contiene model.h5.  Si `base_dir` ya la contiene,
    se devuelve tal cual; de lo contrario se busca 1 nivel por debajo.
    Lanza FileNotFoundError si no se encuentra.
    """
    client = storage.Client()
    bucket_name, prefix = base_dir.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)

    if bucket.blob(f"{prefix.rstrip('/')}/model.h5").exists():
        return base_dir.rstrip("/")

    for blob in bucket.list_blobs(prefix=prefix.rstrip("/") + "/", delimiter="/"):
        pp = PurePosixPath(blob.name)
        if pp.name == "model.h5":
            logger.info("✔ model.h5 hallado en %s", pp.parent.as_posix())
            return f"gs://{bucket_name}/{pp.parent.as_posix()}"

    raise FileNotFoundError(
        f"model.h5 no encontrado en {base_dir} ni en sus subdirectorios."
    )


def _load_artifacts(lstm_model_dir: str, rl_model_path: str) -> Tuple[tf.keras.Model, any, dict, PPO]:
    lstm_model_dir = resolve_artifact_dir(lstm_model_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        model = tf.keras.models.load_model(
            gcs_utils.download_gcs_file(f"{lstm_model_dir}/model.h5", tmp), compile=False
        )
        scaler = joblib.load(
            gcs_utils.download_gcs_file(f"{lstm_model_dir}/scaler.pkl", tmp)
        )
        hp = json.loads(
            gcs_utils.download_gcs_file(f"{lstm_model_dir}/params.json", tmp).read_text()
        )

        rl_local = gcs_utils.download_gcs_file(rl_model_path, tmp)
        if not rl_local.exists():
            raise FileNotFoundError(f"Modelo RL no encontrado en {rl_model_path}")
        ppo = PPO.load(rl_local)

    logger.info("✔ Artefactos LSTM + RL cargados.")
    return model, scaler, hp, ppo


def _prepare_backtest_data(features_path: str, hp: dict) -> pd.DataFrame:
    local = gcs_utils.ensure_gcs_path_and_get_local(features_path)
    if not Path(local).exists():
        raise FileNotFoundError(f"Parquet de features no encontrado: {features_path}")

    df_raw = pd.read_parquet(local)
    if df_raw.empty:
        raise ValueError("Parquet de features está vacío.")

    df_ind = indicators.build_indicators(df_raw, hp, atr_len=ATR_LEN, drop_na=True)
    if df_ind.isna().any().any():
        raise ValueError("Persisten NaNs tras calcular indicadores")

    logger.info("✔ Datos de backtest preparados: %s filas.", len(df_ind))
    return df_ind.reset_index(drop=True)

# ───── stubs de las funciones omitidas (reemplázalas por tu implementación) ─────
def _generate_predictions(df, model, scaler, hp, tick):
    raise NotImplementedError("Implementa _generate_predictions")

def _run_backtest_simulation(up, dn, closes, atr, accept_mask, hp, tick, *, use_filter):
    raise NotImplementedError("Implementa _run_backtest_simulation")

def _calculate_metrics(trades_df) -> dict:
    raise NotImplementedError("Implementa _calculate_metrics")

# ───────────────────── función orquestadora ──────────────────────
def run_backtest(
    *,
    lstm_model_dir: str,
    rl_model_path: str,
    features_path: str,
    pair: str,
    timeframe: str,
) -> tuple[str, str]:
    """
    Devuelve:
        output_dir → carpeta GCS con CSV + metrics.json
        kfp_metrics_path → ruta local al JSON para Vertex UI
    """
    tick = 0.01 if pair.endswith("JPY") else 0.0001
    model, scaler, hp, ppo = _load_artifacts(lstm_model_dir, rl_model_path)
    df_bt = _prepare_backtest_data(features_path, hp)

    # ── Generar predicciones y simulaciones ─────────────────────
    up, dn, emb, closes, atr = _generate_predictions(df_bt, model, scaler, hp, tick)
    if not (len(up) == len(dn) == len(emb) == len(closes)):
        raise ValueError("Longitudes inconsistentes entre vectores predichos.")

    obs_for_rl = np.hstack([np.column_stack([up, dn]), emb])
    accept_mask, _ = ppo.predict(obs_for_rl, deterministic=True)

    trades_base = _run_backtest_simulation(up, dn, closes, atr,
                                           np.ones_like(accept_mask), hp, tick, use_filter=False)
    trades_filt = _run_backtest_simulation(up, dn, closes, atr,
                                           accept_mask, hp, tick, use_filter=True)

    metrics = {
        "base": _calculate_metrics(trades_base),
        "filtered": _calculate_metrics(trades_filt),
    }

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    output_dir = f"{constants.BACKTEST_RESULTS_PATH}/{pair}/{timeframe}/{ts}"

    # ── Guardar resultados y métricas ───────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        if not trades_base.empty:
            csv_base = tmp / "trades_base.csv"
            trades_base.to_csv(csv_base, index=False)
            gcs_utils.upload_gcs_file(csv_base, f"{output_dir}/trades_base.csv")

        if not trades_filt.empty:
            csv_filt = tmp / "trades_filtered.csv"
            trades_filt.to_csv(csv_filt, index=False)
            gcs_utils.upload_gcs_file(csv_filt, f"{output_dir}/trades_filtered.csv")

        json_metrics = tmp / "metrics.json"
        json_metrics.write_text(json.dumps(metrics, indent=2))
        gcs_utils.upload_gcs_file(json_metrics, f"{output_dir}/metrics.json")

        # Formato Vertex-AI / KFP
        kfp_json = tmp / "kfp_metrics.json"
        kfp_json.write_text(json.dumps({
            "metrics": [
                {"name": f"filtered-{k}", "numberValue": (v if np.isfinite(v) else 0), "format": "RAW"}
                for k, v in metrics["filtered"].items()
            ]
        }))

    logger.info("🎯 Backtest finalizado. Resultados en %s", output_dir)
    return output_dir, str(kfp_json)

# ──────────────────────────── CLI (entrypoint) ────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser("Backtest task")
    p.add_argument("--lstm-model-dir", required=True)
    p.add_argument("--rl-model-path", required=True)
    p.add_argument("--features-path", required=True)
    p.add_argument("--pair", required=True)
    p.add_argument("--timeframe", required=True)
    args = p.parse_args()

    out_dir, kfp_metrics_file = run_backtest(
        lstm_model_dir=args.lstm_model_dir,
        rl_model_path=args.rl_model_path,
        features_path=args.features_path,
        pair=args.pair,
        timeframe=args.timeframe,
    )

    # –—— rutas para KFP ————————————————————————————
    Path("/tmp").mkdir(exist_ok=True)
    Path("/tmp/backtest_dir.txt").write_text(out_dir)
    Path("/tmp/kfp_metrics.json").write_text(Path(kfp_metrics_file).read_text())
