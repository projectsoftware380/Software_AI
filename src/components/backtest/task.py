# src/components/backtest/task.py
"""
Back-test completo de la estrategia (LSTM + filtro PPO) y subida de
resultados a GCS.

Genera:
  • /tmp/backtest_dir.txt     → carpeta final en GCS (para KFP)
  • /tmp/kfp_metrics.json     → métricas en formato Vertex AI UI
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Tuple, Dict, List

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from google.cloud import storage
from stable_baselines3 import PPO

from src.shared import constants, gcs_utils, indicators

# ─────────────────────────── logging ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ────────────────── back-test constantes globales ────────────────
ATR_LEN = 14           # debe coincidir con training
COST_PIPS = 0.8        # comisión round-turn

# ───────────────────── helpers utilitarios ──────────────────────
def resolve_artifact_dir(base_dir: str) -> str:
    """
    Devuelve la carpeta que contiene *model.h5*.  Si `base_dir`
    ya la contiene, se devuelve sin cambios; de lo contrario busca
    un sub-path 1 nivel por debajo.
    """
    client = storage.Client()
    bucket_name, prefix = base_dir.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)

    # caso 1: la carpeta ya apunta al modelo
    if bucket.blob(f"{prefix.rstrip('/')}/model.h5").exists():
        return base_dir.rstrip("/")

    # caso 2: buscamos 1 nivel más abajo
    for blob in bucket.list_blobs(prefix=prefix.rstrip("/") + "/", delimiter="/"):
        pp = PurePosixPath(blob.name)
        if pp.name == "model.h5":
            candidate = f"gs://{bucket_name}/{pp.parent.as_posix()}"
            logger.info("✔ model.h5 hallado en %s", candidate)
            return candidate

    raise FileNotFoundError(
        f"model.h5 no encontrado en {base_dir} ni en sus subdirectorios."
    )


def _load_artifacts(lstm_model_dir: str, rl_model_path: str) -> Tuple[
    tf.keras.Model, joblib.Memory, Dict, PPO
]:
    lstm_model_dir = resolve_artifact_dir(lstm_model_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        model = tf.keras.models.load_model(
            gcs_utils.download_gcs_file(f"{lstm_model_dir}/model.h5", tmp),
            compile=False,
        )
        scaler = joblib.load(
            gcs_utils.download_gcs_file(f"{lstm_model_dir}/scaler.pkl", tmp)
        )
        hp = json.loads(
            gcs_utils.download_gcs_file(f"{lstm_model_dir}/params.json", tmp).read_text()
        )

        rl_local = gcs_utils.download_gcs_file(rl_model_path, tmp)
        if not rl_local.exists():
            raise FileNotFoundError(f"Modelo RL no encontrado: {rl_model_path}")
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

    df_ind = indicators.build_indicators(
        df_raw, hp, atr_len=ATR_LEN, drop_na=True
    )
    if df_ind.isna().any().any():
        raise ValueError("Persisten NaNs tras calcular indicadores")

    logger.info("✔ Datos de back-test preparados: %s filas.", len(df_ind))
    return df_ind.reset_index(drop=True)

# ───────────────── funciones que faltaban ───────────────────────
def _to_sequences(mat: np.ndarray, win: int) -> np.ndarray:
    """Crea ventana deslizante 3-D para LSTM (n_seq, win, n_feat)."""
    if len(mat) <= win:
        return np.empty((0, win, mat.shape[1]), dtype=np.float32)
    return np.stack([mat[i - win:i] for i in range(win, len(mat))]).astype(np.float32)


def _generate_predictions(
    df: pd.DataFrame,
    model: tf.keras.Model,
    scaler,
    hp: dict,
    tick: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Devuelve:
      up, dn      → vectores de tamaño N-win
      emb         → matriz (N-win, emb_dim)
      closes      → precio de cierre alineado
      atr         → ATR normalizado por tick
    """
    feature_cols = scaler.feature_names_in_
    X_scaled = scaler.transform(df[feature_cols].values)
    X_seq = _to_sequences(X_scaled, hp["win"])
    if X_seq.size == 0:
        raise ValueError("DataFrame demasiado corto: no se pueden crear secuencias.")

    # predicciones
    preds = model.predict(X_seq, verbose=0, batch_size=1024).astype(np.float32)
    up, dn = preds[:, 0], preds[:, 1]

    # embeddings = penúltima capa
    emb_model = tf.keras.Model(model.input, model.layers[-2].output)
    emb = emb_model.predict(X_seq, verbose=0, batch_size=1024).astype(np.float32)

    closes = df.close.values[hp["win"] :]
    atr = (df[f"atr_{ATR_LEN}"].values / tick)[hp["win"] :]

    return up, dn, emb, closes, atr


def _run_backtest_simulation(
    up: np.ndarray,
    dn: np.ndarray,
    closes: np.ndarray,
    atr: np.ndarray,
    accept_mask: np.ndarray,
    hp: dict,
    tick: float,
    *,
    use_filter: bool,
) -> pd.DataFrame:
    """
    Simula operaciones horizon-based.  Devuelve DataFrame con una
    fila por trade: timestamp idx, dir, entry, exit, pnl_pips.
    """
    horizon = hp["horizon"]
    min_thr_up = hp["min_thr_up"]
    min_thr_dn = hp["min_thr_dn"]
    delta_min = hp["delta_min"]
    smooth_win = hp["smooth_win"]

    trades: List[dict] = []
    dq = deque(maxlen=smooth_win)

    for i in range(len(up) - horizon):
        u, d = up[i], dn[i]
        mag, diff = (u if u > d else d), abs(u - d)
        raw_dir = 1 if u > d else -1
        cond_base = (
            (raw_dir == 1 and u >= min_thr_up)
            or (raw_dir == -1 and d >= min_thr_dn)
        ) and diff >= delta_min

        dq.append(raw_dir if cond_base else 0)

        # suavizado para la señal “base”
        buys, sells = dq.count(1), dq.count(-1)
        base_signal = 1 if buys > smooth_win // 2 else -1 if sells > smooth_win // 2 else 0

        # filtro PPO (accept_mask) — en back-test base_signal se
        # multiplica por 1 o por 0 según lo acepte el filtro
        if base_signal != 0:
            final_signal = base_signal if accept_mask[i] == 1 else 0
        else:
            final_signal = 0

        if final_signal == 0:
            continue

        entry = closes[i]
        exit_ = closes[i + horizon]
        pnl_pips = ((exit_ - entry) / tick) * final_signal - COST_PIPS
        trades.append(
            {
                "idx": i,
                "dir": final_signal,
                "entry_price": entry,
                "exit_price": exit_,
                "pnl_pips": pnl_pips,
            }
        )

    return pd.DataFrame(trades)


def _calculate_metrics(trades_df: pd.DataFrame) -> Dict[str, float]:
    """Calcula métricas básicas de rendimiento."""
    if trades_df.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "total_pips": 0.0,
            "avg_pips": 0.0,
            "sharpe": 0.0,
        }

    pnl = trades_df.pnl_pips.values
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    total_pips = pnl.sum()
    win_rate = len(wins) / len(pnl)
    avg_pips = pnl.mean()
    # Sharpe sencillo (√N * mean / std)
    sharpe = math.sqrt(len(pnl)) * avg_pips / (pnl.std() + 1e-8)

    return {
        "trades": float(len(pnl)),
        "win_rate": float(win_rate),
        "total_pips": float(total_pips),
        "avg_pips": float(avg_pips),
        "sharpe": float(sharpe),
    }

# ───────────────────── función orquestadora ──────────────────────
def run_backtest(
    *,
    lstm_model_dir: str,
    rl_model_path: str,
    features_path: str,
    pair: str,
    timeframe: str,
) -> Tuple[str, str]:
    """
    Devuelve:
      • output_dir          → carpeta GCS con CSV + metrics.json
      • kfp_metrics_path    → ruta local al JSON para Vertex AI
    """
    tick = 0.01 if pair.endswith("JPY") else 0.0001
    model, scaler, hp, ppo = _load_artifacts(lstm_model_dir, rl_model_path)
    df_bt = _prepare_backtest_data(features_path, hp)

    # ── Generar predicciones ────────────────────────────────────
    up, dn, emb, closes, atr = _generate_predictions(df_bt, model, scaler, hp, tick)

    obs_for_rl = np.hstack([np.column_stack([up, dn]), emb])
    accept_mask, _ = ppo.predict(obs_for_rl, deterministic=True)

    # ── Simular estrategias ─────────────────────────────────────
    trades_base = _run_backtest_simulation(
        up, dn, closes, atr,
        np.ones_like(accept_mask), hp, tick,
        use_filter=False,
    )
    trades_filt = _run_backtest_simulation(
        up, dn, closes, atr,
        accept_mask, hp, tick,
        use_filter=True,
    )

    metrics = {
        "base": _calculate_metrics(trades_base),
        "filtered": _calculate_metrics(trades_filt),
    }

    # ── Guardar resultados a GCS ────────────────────────────────
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    output_dir = f"{constants.BACKTEST_RESULTS_PATH}/{pair}/{timeframe}/{ts}"

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

        # formato Vertex AI / KFP
        kfp_json = tmp / "kfp_metrics.json"
        kfp_json.write_text(
            json.dumps(
                {
                    "metrics": [
                        {
                            "name": f"filtered-{k}",
                            "numberValue": (v if np.isfinite(v) else 0),
                            "format": "RAW",
                        }
                        for k, v in metrics["filtered"].items()
                    ]
                }
            )
        )

    logger.info("🎯 Back-test finalizado. Resultados en %s", output_dir)
    return output_dir, str(kfp_json)


# ───────────────────────────── CLI ──────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser("Back-test task")
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

    # — rutas para KFP —
    Path("/tmp").mkdir(exist_ok=True)
    Path("/tmp/backtest_dir.txt").write_text(out_dir)
    Path("/tmp/kfp_metrics.json").write_text(Path(kfp_metrics_file).read_text())
