#!/usr/bin/env python3
"""
prepare_opt_data.py
────────────────────
1. Calcula indicadores sobre el Parquet OHLC de entrada.
2. Recorta la ventana a los últimos N años (def. 5) **después** de asegurar
   que no queden NaNs.
3. Guarda el Parquet filtrado en
      gs://…/data_filtered_for_opt_v2/<pair>/<tf>/<TIMESTAMP>/…
4. Limpieza automática:
   • Examina los sub-directorios con sello YYYYMMDDHHMMSS.
   • Conservar SOLO el más reciente; borra todos los anteriores.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd
from indicators import build_indicators

# Dependencias GCS (auto-instalación si hace falta)
try:
    import gcsfs
    from google.cloud import storage
except ImportError:  # pragma: no cover
    import subprocess as _sp, sys as _sys
    _sp.check_call([_sys.executable, "-m", "pip", "install", "-q", "gcsfs", "google-cloud-storage"])
    import gcsfs
    from google.cloud import storage     # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ───────────────────────── GCS helpers ──────────────────────────
def _read_parquet(uri: str) -> pd.DataFrame:
    opts = {"engine": "pyarrow"}
    if uri.startswith("gs://"):
        opts["storage_options"] = {"token": "cloud"}
    return pd.read_parquet(uri, **opts)

def _write_parquet(df: pd.DataFrame, uri: str) -> None:
    opts = {"index": False, "engine": "pyarrow"}
    if uri.startswith("gs://"):
        opts["storage_options"] = {"token": "cloud"}
    else:
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(uri, **opts)

def _keep_only_latest(base_prefix: str) -> None:
    """
    Mantiene solo el sub-directorio con timestamp más reciente bajo
    `base_prefix` (terminado en /<pair>/<tf>/).
    """
    fs = gcsfs.GCSFileSystem(token="cloud")
    dirs = [
        d.rstrip("/")
        for d in fs.ls(base_prefix)
        if re.search(r"/\d{14}/?$", d)           # patrón YYYYMMDDHHMMSS
    ]
    if len(dirs) <= 1:
        return  # nada que limpiar

    # Orden descendente por timestamp (el primero es el más nuevo)
    dirs_sorted = sorted(dirs, key=lambda p: p.split("/")[-1], reverse=True)
    for old_dir in dirs_sorted[1:]:
        log.info("🗑️  Eliminando versión anterior %s", old_dir)
        fs.rm(old_dir, recursive=True)

# ─────────────────────────── main ──────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path",  required=True,
                    help="Parquet OHLC fuente (gs:// o local)")
    ap.add_argument("--output_path", required=True,
                    help="Ruta destino para el Parquet filtrado")
    ap.add_argument("--years", type=int, default=5,
                    help="Años de historial a conservar (def. 5)")
    args = ap.parse_args()

    # 1. Carga y limpieza básica
    df = _read_parquet(args.input_path).dropna(subset=["open", "high", "low", "close"])
    log.info("✔ Datos cargados: %,d filas", len(df))

    # 2. Indicadores antes del recorte temporal
    hp_dummy: Dict[str, int] = dict(sma_len=50, rsi_len=14, macd_fast=12,
                                    macd_slow=26, stoch_len=14)
    df = build_indicators(df, hp_dummy, inplace=True, drop_na=True)
    log.info("🔧 Indicadores calculados → %,d filas tras NaN drop", len(df))

    # 3. Recorte a N años
    end_ts = df["timestamp"].max()
    start_ts = pd.to_datetime(end_ts) - pd.DateOffset(years=args.years)
    df_window = df[df["timestamp"] >= start_ts]
    if df_window.empty:
        raise RuntimeError("Ventana temporal resultó vacía; revisa rango de fechas.")
    log.info("🗂  Ventana %s → %s  ⇒ %,d filas",
             start_ts.date(), pd.to_datetime(end_ts).date(), len(df_window))

    _write_parquet(df_window, args.output_path)
    log.info("☁️  Parquet filtrado guardado en %s", args.output_path)

    # 4. Limpieza de versiones antiguas
    if args.output_path.startswith("gs://"):
        # base_prefix = .../data_filtered_for_opt_v2/<pair>/<tf>/
        base_prefix = "/".join(args.output_path.split("/")[:-2]) + "/"
        _keep_only_latest(base_prefix)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:      # pragma: no cover
        log.critical("❌  prepare_opt_data error: %s", exc, exc_info=True)
        sys.exit(1)
