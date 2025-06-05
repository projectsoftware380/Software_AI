#!/usr/bin/env python3
"""
model_promotion_decision.py
───────────────────────────
Decide si se promociona un nuevo conjunto (LSTM + PPO) a producción.

Principales cambios
-------------------
* Añadidos imports que faltaban: **sys** y **tempfile**.
* Eliminada la definición duplicada de *PUBSUB_PROJECT_ID*.
* Añadido manejo de errores y ‘logging’ más claro en `copy_gcs_object`.
* Normalización de métricas:
  • Todas las puntuaciones pasan por `clip()` → rango [0-1].  
  • `trades_score`, `pf_score` y `exp_score` ya no pueden ser negativas.
* Fórmula del `expectancy_score` simplificada (objetivo ≥ 0 → “cuanto mayor, mejor”).
* Uso coherente de constantes y anotaciones.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
from google.cloud import pubsub_v1, storage
from google.oauth2 import service_account
from google.auth.exceptions import DefaultCredentialsError

# ────────────────────────────── logging ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────── configuración & umbrales ───────────────────────
MIN_TRADES_FOR_PROMOTION = 100
MIN_NET_PIPS_ABSOLUTE = 0.0
MIN_PROFIT_FACTOR_ABSOLUTE = 1.0
MIN_SHARPE_ABSOLUTE_FIRST_DEPLOY = 0.5

TARGETS = {
    "sharpe": 0.60,
    "pf": 1.75,
    "winrate": 0.50,
    "expectancy": 0.0,
    "trades": 150,
}

WEIGHTS = {
    "sharpe": 0.35,
    "dd": 0.30,
    "pf": 0.25,
    "win": 0.05,
    "exp": 0.03,
    "ntrades": 0.02,
}

MAX_DRAWDOWN_REL_TOLERANCE_FACTOR = 1.5
MAX_DRAWDOWN_ABS_THRESHOLD = 25.0
GLOBAL_PROMOTION_SCORE_THRESHOLD = 0.80

# Pub/Sub
PUBSUB_PROJECT_ID = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT", "")
SUCCESS_TOPIC_ID = "data-ingestion-success"
FAILURE_TOPIC_ID = "data-ingestion-failures"

# ──────────────────────────── helpers GCS ────────────────────────────
def get_gcs_client() -> storage.Client:
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and Path(creds_path).exists():
        creds = service_account.Credentials.from_service_account_file(creds_path)
        return storage.Client(credentials=creds)
    return storage.Client()  # intentará ADC


def download_gs(uri: str, dest_dir: Path) -> Path:
    bucket_name, blob_name = uri[5:].split("/", 1)
    local_path = dest_dir / Path(blob_name).name
    get_gcs_client().bucket(bucket_name).blob(blob_name).download_to_filename(local_path)
    logger.info(f"📥  Descargado {uri} → {local_path}")
    return local_path


def upload_gs_file(local_path: Path, gcs_uri: str) -> None:
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    get_gcs_client().bucket(bucket_name).blob(blob_name).upload_from_filename(str(local_path))
    logger.info(f"☁️  Subido {local_path.name} → {gcs_uri}")


def copy_gcs_object(source_uri: str, destination_uri: str) -> None:
    client = get_gcs_client()
    src_bucket, src_blob = source_uri[5:].split("/", 1)
    dst_bucket, dst_blob = destination_uri[5:].split("/", 1)
    try:
        client.bucket(src_bucket).copy_blob(
            client.bucket(src_bucket).blob(src_blob),
            client.bucket(dst_bucket),
            dst_blob,
        )
        logger.info(f"✔ Copiado {source_uri} → {destination_uri}")
    except Exception as e:
        logger.error(f"Error al copiar {source_uri} → {destination_uri}: {e}")
        raise


# ───────────────────────── helpers Pub/Sub ───────────────────────────
def get_pubsub_publisher_client() -> pubsub_v1.PublisherClient:
    return pubsub_v1.PublisherClient()


def publish_notification(topic_id: str, status: str, message: str, details: Dict[str, Any] | None = None) -> None:
    publisher = get_pubsub_publisher_client()
    topic_path = publisher.topic_path(PUBSUB_PROJECT_ID, topic_id)
    payload = {
        "status": status,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details or {},
    }
    try:
        message_id = publisher.publish(topic_path, json.dumps(payload).encode()).result()
        logger.info(f"Notificación enviada a {topic_id} (ID: {message_id})")
    except Exception as e:
        logger.error(f"Error publicando en Pub/Sub: {e}")


# ───────────────────────── utilidades varias ─────────────────────────
def clip(value: float) -> float:
    """Limita *value* al rango [0, 1]."""
    return max(0.0, min(value, 1.0))


def load_metrics(gcs_path: str, assume_missing_ok: bool = False) -> Dict[str, Any]:
    """
    Descarga y devuelve el JSON de métricas; si falta y *assume_missing_ok* → dict con valores ‘malos’.
    """
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        local_json = download_gs(gcs_path, tmp_dir)
        with open(local_json, "r") as fh:
            data = json.load(fh)
        return data.get("filtered", data)  # Preferimos las métricas filtradas
    except Exception as err:
        if assume_missing_ok:
            logger.warning(f"No se pudo leer {gcs_path}. Se asume primer despliegue. Detalle: {err}")
            return {
                "trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy": -1e9,
                "net_pips": -1e9,
                "sharpe": -1e9,
                "max_drawdown": -1e9,
            }
        logger.critical(f"Error crítico leyendo métricas: {err}")
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ────────────────────────────── main ─────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser("Decisión de promoción de modelos")
    parser.add_argument("--new-metrics-path", required=True)
    parser.add_argument("--current-production-metrics-path", required=True)
    parser.add_argument("--new-lstm-artifacts-dir", required=True)
    parser.add_argument("--new-rl-model-path", required=True)
    parser.add_argument("--production-base-dir", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    args = parser.parse_args()

    logger.info(f"→ Evaluando promoción para {args.pair}/{args.timeframe}")

    # 1 Cargar métricas
    try:
        new_mx = load_metrics(args.new_metrics_path)
        prod_mx = load_metrics(args.current_production_metrics_path, assume_missing_ok=True)
    except Exception:
        publish_notification(
            FAILURE_TOPIC_ID,
            "ERROR",
            "Error leyendo métricas (no se pudo continuar).",
            {"new": args.new_metrics_path, "prod": args.current_production_metrics_path},
        )
        sys.exit(1)

    # 2 Vetos rápidos de drawdown
    new_dd = abs(new_mx.get("max_drawdown", -1e9))
    prod_dd = abs(prod_mx.get("max_drawdown", -1e9))

    veto_reasons: list[str] = []
    if MAX_DRAWDOWN_ABS_THRESHOLD > 0 and new_dd > MAX_DRAWDOWN_ABS_THRESHOLD:
        veto_reasons.append(f"Max DD {new_dd:.2f} supera {MAX_DRAWDOWN_ABS_THRESHOLD:.2f}")
    if prod_dd > 0 and new_dd >= prod_dd * MAX_DRAWDOWN_REL_TOLERANCE_FACTOR:
        veto_reasons.append(f"Max DD {new_dd:.2f} ≥ {MAX_DRAWDOWN_REL_TOLERANCE_FACTOR}× del actual {prod_dd:.2f}")

    if veto_reasons:
        publish_notification(
            FAILURE_TOPIC_ID,
            "VETO",
            "Modelo vetado por drawdown.",
            {"reasons": veto_reasons, "new_metrics": new_mx, "current_metrics": prod_mx},
        )
        logger.warning("Modelo vetado:\n" + "\n".join(f" • {r}" for r in veto_reasons))
        sys.exit(0)

    # 3 Scores normalizados
    sharpe_score = clip(new_mx.get("sharpe", 0.0) / TARGETS["sharpe"])
    pf_score = clip(new_mx.get("profit_factor", 0.0) / TARGETS["pf"])
    win_score = clip(new_mx.get("win_rate", 0.0) / TARGETS["winrate"])
    exp_score = clip(max(0.0, new_mx.get("expectancy", -1e9)) / (abs(TARGETS["expectancy"]) + 1e-9) + 1e-9)
    trades_score = clip(new_mx.get("trades", 0) / TARGETS["trades"])
    dd_score = clip(
        1.0 - (new_dd - prod_dd) / (prod_dd * 0.20 + 1e-9)
        if prod_dd > 0 else 1.0
    )

    total_score = (
        sharpe_score * WEIGHTS["sharpe"]
        + dd_score * WEIGHTS["dd"]
        + pf_score * WEIGHTS["pf"]
        + win_score * WEIGHTS["win"]
        + exp_score * WEIGHTS["exp"]
        + trades_score * WEIGHTS["ntrades"]
    )

    logger.info(
        "Scores: "
        f"Sharpe={sharpe_score:.2f}, DD={dd_score:.2f}, PF={pf_score:.2f}, "
        f"Win={win_score:.2f}, Exp={exp_score:.2f}, Trades={trades_score:.2f} → "
        f"Total={total_score:.2f}"
    )

    # 4 Promoción / no-promoción
    if total_score >= GLOBAL_PROMOTION_SCORE_THRESHOLD:
        logger.info("✨ Criterios de promoción superados. Copiando artefactos…")
        try:
            # LSTM
            for artefact in ("model.h5", "scaler.pkl", "params.json"):
                copy_gcs_object(
                    f"{args.new_lstm_artifacts_dir.rstrip('/')}/{artefact}",
                    f"{args.production_base_dir.rstrip('/')}/{artefact}",
                )
            # PPO
            copy_gcs_object(
                args.new_rl_model_path,
                f"{args.production_base_dir.rstrip('/')}/ppo_filter_model.zip",
            )
            # métricas de producción
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_metrics = Path(tmpdir) / "metrics_production.json"
                json.dump({"filtered": new_mx}, open(tmp_metrics, "w"), indent=2)
                upload_gs_file(tmp_metrics, f"{args.production_base_dir.rstrip('/')}/metrics_production.json")

            publish_notification(
                SUCCESS_TOPIC_ID,
                "PROMOTED",
                f"{args.pair}/{args.timeframe} promovido a producción.",
                {"score": total_score, "new_metrics": new_mx},
            )
            logger.info("✔ Promoción completada.")
        except Exception as e:
            publish_notification(
                FAILURE_TOPIC_ID,
                "ERROR",
                "Error durante la copia de artefactos.",
                {"exception": str(e)},
            )
            logger.error(f"Error copiando artefactos: {e}", exc_info=True)
            sys.exit(1)
    else:
        publish_notification(
            FAILURE_TOPIC_ID,
            "NO_PROMOTED",
            f"Score {total_score:.2f} < umbral {GLOBAL_PROMOTION_SCORE_THRESHOLD:.2f}.",
            {"new_metrics": new_mx, "current_metrics": prod_mx, "score": total_score},
        )
        logger.info("Modelo no promovido.")
        sys.exit(0)


if __name__ == "__main__":
    main()
