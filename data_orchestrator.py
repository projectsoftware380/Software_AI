#!/usr/bin/env python3
"""
data_orchestrator.py
────────────────────
Borra (si existe) el Parquet histórico para cada símbolo/time-frame y
vuelve a descargar todos los datos OHLC mediante *data_fetcher_vertexai.py*.

Cambios clave (jun-2025)
────────────────────────
•  *Borrado duro*: cualquier Parquet previo se elimina antes de descargar.  
•  Se sigue propagando `--min-rows` al sub-script para asegurar descargas
   “no vacías”.  
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List

# ────────────────────── Google Cloud libs ──────────────────────
try:
    import gcsfs
    from google.cloud import storage, pubsub_v1
except ImportError:  # instalación auto-defensiva (solo en contenedores ligeros)
    import subprocess as _sp, sys as _sys
    _sp.check_call(
        [_sys.executable, "-m", "pip", "install", "-q", "gcsfs",
         "google-cloud-storage", "google-cloud-pubsub"]
    )
    import gcsfs
    from google.cloud import storage, pubsub_v1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ────────────────────────── Constantes ─────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
BUCKET   = os.getenv("GCS_BUCKET", "trading-ai-models-460823")
DATA_DIR = f"gs://{BUCKET}/data"

MIN_ROWS_DEFAULT = int(os.getenv("MIN_ROWS_DATA", "100000"))

# ─────────────────────── helpers GCS ───────────────────────────
_fs = gcsfs.GCSFileSystem(token="cloud")

def parquet_exists(uri: str) -> bool:
    return _fs.exists(uri)

def delete_gcs_blob(uri: str) -> None:
    bucket, blob = uri[5:].split("/", 1)
    storage.Client().bucket(bucket).blob(blob).delete()
    log.info("🗑️  Parquet eliminado: %s", uri)

# ────────────────── descarga de símbolo/tf ─────────────────────
def download_symbol_tf(
    symbol: str,
    timeframe: str,
    min_rows: int,
    env: Dict[str, str],
) -> bool:
    parquet_uri = f"{DATA_DIR}/{symbol}/{timeframe}/{symbol}_{timeframe}.parquet"

    # --- 1. Borrado duro ---------------------------------------------------
    if parquet_exists(parquet_uri):
        delete_gcs_blob(parquet_uri)

    # --- 2. Invocar downloader --------------------------------------------
    fetcher = ROOT_DIR / "data_fetcher_vertexai.py"
    cmd = [
        sys.executable, str(fetcher),
        "--data-dir", DATA_DIR,
        "--symbols", symbol,
        "--timeframes", timeframe,
        "--start-date", "2008-01-01",
        "--end-date", date.today().isoformat(),
        "--min-rows", str(min_rows),
        "--polygon-key", env["POLYGON_API_KEY"],
    ]
    log.info("🚀  Ejecutando: %s", " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    return result.returncode == 0

# ───────────────────────── main ────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_id", required=True)
    ap.add_argument("--symbols", nargs="+", default=["EURUSD"])
    ap.add_argument("--timeframes", nargs="+", default=["15minute"])
    ap.add_argument("--min-rows", type=int, default=MIN_ROWS_DEFAULT)
    args = ap.parse_args()

    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        log.critical("POLYGON_API_KEY no está definido en variables de entorno.")
        sys.exit(1)

    env = os.environ.copy()
    env["POLYGON_API_KEY"] = api_key

    success = True
    for sym in args.symbols:
        for tf in args.timeframes:
            success &= download_symbol_tf(sym, tf, args.min_rows, env)

    topic = "data-ingestion-success" if success else "data-ingestion-failures"
    payload = {
        "symbols": args.symbols,
        "timeframes": args.timeframes,
        "status": success,
    }
    pubsub_v1.PublisherClient().publish(
        pubsub_v1.PublisherClient().topic_path(args.project_id, topic),
        json.dumps(payload).encode(),
    ).result()

    if not success:
        sys.exit("Alguna descarga falló; ver logs.")

# -------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # fallo inesperado
        log.critical("❌  data_orchestrator terminó con error: %s", exc, exc_info=True)
        sys.exit(1)
