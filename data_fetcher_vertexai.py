#!/usr/bin/env python3
"""
data_fetcher_vertexai.py
────────────────────────
Descarga barras OHLC de Polygon.io y guarda un único Parquet
(local o GCS) con **suficientes filas**.  
Novedades (jun-2025)
────────────────────
•  Argumento `--min-rows` (def. 100 000).  
   Si la descarga reúne menos filas ⇒ se lanza `RuntimeError`.  
•  Si el Parquet de destino ya existe se borra antes de escribir
   (para evitar mezclar descargas antiguas con incompletas).  
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ────────────────────────────── logging ──────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────── GCS helpers (opcionales) ────────────────────
try:
    import gcsfs
    from google.cloud import storage
except ImportError:  # se instala en la imagen Docker
    gcsfs = None  # type: ignore
    storage = None  # type: ignore

def _remove_if_exists(uri: str) -> None:
    """Borra un archivo local o GCS si existe."""
    if uri.startswith("gs://"):
        if storage is None:
            raise RuntimeError("google-cloud-storage no disponible para borrar objetos.")
        bucket_name, blob_name = uri[5:].split("/", 1)
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_name)
        if blob.exists():
            log.info(f"🗑️  Eliminando Parquet previo {uri}")
            blob.delete()
    else:
        p = Path(uri)
        if p.exists():
            log.info(f"🗑️  Eliminando Parquet previo {p}")
            p.unlink()

def _save_parquet(df: pd.DataFrame, uri: str) -> None:
    """Guarda el DataFrame en local o GCS."""
    Path(uri).parent.mkdir(parents=True, exist_ok=True) if not uri.startswith("gs://") else None
    opts: Dict[str, Any] = {"index": False, "engine": "pyarrow"}
    if uri.startswith("gs://"):
        if gcsfs is None:
            raise RuntimeError("gcsfs no instalado → pip install gcsfs")
        opts["storage_options"] = {"token": "cloud"}
    df.to_parquet(uri, **opts)
    log.info(f"☁️  Parquet guardado en {uri} ({len(df):,} filas)")

# ───────────────────────── API helpers ───────────────────────────────
POLYGON_MAX_LIMIT = 50_000

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(requests.HTTPError),
    reraise=True,
)
def _fetch_window(
    session: requests.Session,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    api_key: str,
) -> List[Dict[str, Any]]:
    m = re.match(r"^(\d+)([a-zA-Z]+)$", timeframe)
    if not m:
        raise ValueError(f"Invalid timeframe {timeframe!r}")
    mult, span = m.groups()
    url = f"https://api.polygon.io/v2/aggs/ticker/C:{symbol}/range/{mult}/{span}/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": POLYGON_MAX_LIMIT,
        "apiKey": api_key,
    }
    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("results", [])

def _results_to_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    df = (
        pd.DataFrame(results)
        .rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "timestamp"})
        .dropna(subset=["open", "high", "low", "close"])
        .assign(timestamp=lambda d: d["timestamp"].astype("int64"))
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )
    return df

# ───────────────────────────── main ──────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Descarga OHLC de Polygon.io a Parquet.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--timeframes", nargs="+", required=True)
    ap.add_argument("--polygon-key")
    ap.add_argument("--start-date", default="2000-01-01")
    ap.add_argument("--end-date")
    ap.add_argument("--min-rows", type=int, default=100_000,
                    help="Filas mínimas requeridas para considerar válida la descarga (def. 100 000)")
    args = ap.parse_args()

    api_key = args.polygon_key or os.getenv("POLYGON_API_KEY")
    if not api_key:
        ap.error("Falta API-key de Polygon (--polygon-key o env POLYGON_API_KEY)")

    start = dt.date.fromisoformat(args.start_date)
    end = dt.date.fromisoformat(args.end_date) if args.end_date else dt.date.today()
    if end < start:
        ap.error("--end-date anterior a --start-date")

    session = requests.Session()
    for sym in args.symbols:
        for tf in args.timeframes:
            log.info("📥 %s | %s | %s → %s", sym, tf, start, end)
            dfs: List[pd.DataFrame] = []
            win_start = start
            while win_start <= end:
                win_end = min(win_start + dt.timedelta(days=30), end)
                try:
                    res = _fetch_window(session, sym, tf, win_start.isoformat(), win_end.isoformat(), api_key)
                    dfw = _results_to_df(res)
                    dfs.append(dfw) if not dfw.empty else None
                except Exception as e:
                    log.warning("⚠️  Error ventana %s-%s: %s", win_start, win_end, e)
                win_start = win_end + dt.timedelta(days=1)

            df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

            out_prefix = f"{args.data_dir.rstrip('/')}/{sym}/{tf}"
            out_path = f"{out_prefix}/{sym}_{tf}.parquet"
            _remove_if_exists(out_path)  # borra versión previa

            if len(df_all) < args.min_rows:
                raise RuntimeError(
                    f"Descarga incompleta: {len(df_all):,} filas (< {args.min_rows:,})."
                    " Se aborta para evitar Parquet vacío."
                )

            _save_parquet(df_all, out_path)

    session.close()
    log.info("🏁 Descargas completas sin errores.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.critical("❌  data_fetcher_vertexai terminó con error: %s", e, exc_info=True)
        sys.exit(1)
