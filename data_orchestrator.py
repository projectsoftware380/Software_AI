#!/usr/bin/env python3
"""
data_orchestrator.py
────────────────────
Orquesta la descarga y actualización de OHLC desde Polygon.io hacia GCS
llamando al sub-script `data_fetcher_vertexai.py`.

✅ Cambios clave
────────────────
1.  Resuelve la ruta absoluta del sub-script para evitar FileNotFoundError
    cuando el directorio de trabajo no es el que contiene este archivo.
2.  Parámetros sensibles (reintentos, fecha de inicio por defecto, bucket, …)
    obtienen _fallback_ seguro si no están presentes en el JSON de config.
3.  Mensajes de Pub/Sub enriquecidos con `request_id` y listado de pares
    procesados / fallidos.
4.  Control estricto de credenciales: se verifica la API-KEY y el acceso a
    GCS antes de lanzar procesos costosos.
5.  Limpieza y tipado ligero 👉 las importaciones/instalaciones “defensivas”
    se mantienen, pero sólo se ejecutan cuando son realmente necesarias.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# ── Logging global ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Google libs (se instalan on-the-fly sólo si faltan) ───────────────────
try:
    import gcsfs
    from google.cloud import pubsub_v1, storage
    from google.oauth2 import service_account
    from google.auth.exceptions import DefaultCredentialsError
except ImportError:  # imagen local mínima: instalamos lo justo
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "gcsfs", "google-cloud-storage", "google-cloud-pubsub"]
    )
    import gcsfs
    from google.cloud import pubsub_v1, storage
    from google.oauth2 import service_account
    from google.auth.exceptions import DefaultCredentialsError

import pandas as pd

# ── Constantes de entorno / rutas ─────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
GCS_BUCKET = os.getenv("GCS_BUCKET", "trading-ai-models-460823")
CONFIG_PATH = f"gs://{GCS_BUCKET}/config/data_fetch_config.json"
DATA_BASE_PATH = f"gs://{GCS_BUCKET}/data"

SUCCESS_TOPIC_ID = "data-ingestion-success"
FAILURE_TOPIC_ID = "data-ingestion-failures"

# Fallbacks si faltan en el JSON de configuración
DEFAULT_START_DATE = "2000-01-01"
DEFAULT_MAX_RETRIES = 5

# ── Helpers GCS & Pub/Sub ─────────────────────────────────────────────────
def _gcs_client() -> storage.Client:
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and Path(creds_path).exists():
        creds = service_account.Credentials.from_service_account_file(creds_path)
        return storage.Client(credentials=creds)
    return storage.Client()

def download_json_from_gcs(gcs_uri: str) -> Dict:
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    blob = _gcs_client().bucket(bucket_name).blob(blob_name)
    return json.loads(blob.download_as_text())

def parquet_exists(gcs_uri: str) -> bool:
    return gcsfs.GCSFileSystem(token="cloud").exists(gcs_uri)

def last_timestamp(gcs_uri: str) -> Optional[str]:
    try:
        df = pd.read_parquet(gcs_uri, engine="pyarrow", storage_options={"token": "cloud"})
        if "timestamp" in df.columns and not df["timestamp"].empty:
            return pd.to_datetime(df["timestamp"]).max().strftime("%Y-%m-%d")
    except Exception as exc:
        logger.warning(f"No se pudo leer último timestamp de {gcs_uri}: {exc}")
    return None

def publish(topic_id: str, payload: Dict, project_id: str) -> None:
    topic_path = pubsub_v1.PublisherClient().topic_path(project_id, topic_id)
    data = json.dumps(payload).encode()
    pubsub_v1.PublisherClient().publish(topic_path, data).result(timeout=60)

# ── Lógica de descarga por par/timeframe ──────────────────────────────────
def fetch_symbol_timeframe(
    symbol: str,
    timeframe: str,
    cfg: Dict,
    env: Dict[str, str],
) -> bool:
    parquet_path = f"{DATA_BASE_PATH}/{symbol}/{timeframe}/{symbol}_{timeframe}.parquet"

    # Determinar ventana temporal
    start_date = cfg.get("default_start_date_new_parquet", DEFAULT_START_DATE)
    if parquet_exists(parquet_path):
        last_ts = last_timestamp(parquet_path)
        if last_ts:
            dt_next = datetime.fromisoformat(last_ts).date() + timedelta(days=1)
            if dt_next > date.today():
                logger.info(f"{symbol}-{timeframe} ya está al día; se omite descarga.")
                return True
            start_date = dt_next.isoformat()
    end_date = date.today().isoformat()

    logger.info(f"{symbol}-{timeframe}: ventana {start_date} ➡ {end_date}")

    # Construir comando para el sub-script (ruta absoluta)
    fetcher_path = ROOT_DIR / "data_fetcher_vertexai.py"
    cmd = [
        sys.executable,
        str(fetcher_path),
        "--data-dir",
        DATA_BASE_PATH,
        "--symbols",
        symbol,
        "--timeframes",
        timeframe,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--polygon-key",
        env["POLYGON_API_KEY"],
    ]

    retries = cfg.get("max_retries", DEFAULT_MAX_RETRIES)
    for attempt in range(1, retries + 1):
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode == 0:
            logger.info(f"✅ Descarga completada para {symbol}-{timeframe}")
            return True

        logger.warning(f"⚠️  Fallo intento {attempt}/{retries} ({symbol}-{timeframe})")
        logger.debug(result.stderr)
        if "401 Unauthorized" in result.stderr:
            logger.critical("API Key inválida – se abortan reintentos.")
            break
        time.sleep(2 ** attempt)  # back-off exponencial

    logger.error(f"❌ Descarga fallida para {symbol}-{timeframe} tras {retries} reintentos")
    return False

# ── main ──────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Orquestador de descargas OHLC a GCS")
    ap.add_argument("--mode", choices=["scheduled", "on-demand"], default="scheduled")
    ap.add_argument("--message", help="JSON de Pub/Sub para modo on-demand")
    ap.add_argument("--project_id", required=True)
    args = ap.parse_args()

    # Cargar configuración global
    try:
        cfg = download_json_from_gcs(CONFIG_PATH)
    except Exception as exc:
        logger.critical(f"No se pudo leer configuración {CONFIG_PATH}: {exc}")
        sys.exit(1)

    # Resolver lista de pares
    if args.mode == "on-demand":
        try:
            msg = json.loads(args.message or "{}")
            symbols = msg.get("symbols", [])
            timeframes = msg.get("timeframes", [])
            request_id = msg.get("request_id")
        except json.JSONDecodeError:
            logger.error("Mensaje on-demand no es JSON válido.")
            sys.exit(1)
    else:
        symbols = cfg.get("symbols", [])
        timeframes = cfg.get("timeframes", [])
        request_id = None

    if not symbols or not timeframes:
        logger.error("Lista de símbolos/timeframes vacía; abortando.")
        sys.exit(1)

    # Validar credenciales
    polygon_key = os.getenv("POLYGON_API_KEY")
    if not polygon_key:
        logger.critical("POLYGON_API_KEY no establecido en el entorno.")
        sys.exit(1)

    env = os.environ.copy()
    env["POLYGON_API_KEY"] = polygon_key
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        env["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    processed: List[str] = []
    failed: List[Dict] = []

    for sym in symbols:
        for tf in timeframes:
            ok = fetch_symbol_timeframe(sym, tf, cfg, env)
            (processed if ok else failed).append(f"{sym}_{tf}" if ok else {"symbol": sym, "timeframe": tf})

    # Publicar resultados
    if not failed:
        publish(
            SUCCESS_TOPIC_ID,
            {
                "status": "success",
                "processed_pairs": processed,
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": request_id,
            },
            args.project_id,
        )
        logger.info("🎉 Todas las descargas finalizaron con éxito.")
    else:
        publish(
            FAILURE_TOPIC_ID,
            {
                "status": "failure",
                "processed_pairs": processed,
                "failed_pairs": failed,
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": request_id,
            },
            args.project_id,
        )
        logger.error("Algunas descargas fallaron. Ver detalles arriba.")

if __name__ == "__main__":
    main()
