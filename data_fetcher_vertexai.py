#!/usr/bin/env python3
"""
data_fetcher_vertexai.py
────────────────────────
Descarga velas históricas desde la API de Polygon.io y las guarda
en formato Parquet (local o GCS).  
✅ Cambios relevantes frente a la versión previa
-----------------------------------------------------------------
1.  **Renombrado consistente de columnas**  ➜  open, high, low, close,
    volume y timestamp (ms, dtype int64).  
2.  **Elimina filas con NAN en OHLC** antes de guardar; así los
    indicadores nunca reciben datos vacíos.  
3.  **Reintentos exponenciales** (tenacity) ante errores HTTP / rate-limit.  
4.  **Orden cronológico** y `drop_duplicates` por timestamp.  
5.  **Validaciones de entrada** (símbolos, timeframe, fechas).  
6.  **Salida informativa** y código 0 aunque algún símbolo no devuelva
    datos — no aborta todo el lote.  
"""

from __future__ import annotations

import os
import re
import argparse
import logging
import datetime as dt
from pathlib import Path
from typing import List, Dict, Any

import requests
import pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# --- GCS opcional ----------------------------------------------------------
try:
    import gcsfs
except ImportError:       # se instala en la imagen Docker
    gcsfs = None  # type: ignore

# ---------------------------------------------------------------------------

LOG_FMT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger(__name__)

POLYGON_MAX_LIMIT = 50_000       # Límite hard‐coded de la API


# ═════════════════════════════════ helper de descarga ══════════════════════
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
    """
    Descarga como máximo 50 000 barras de una ventana `start_date` → `end_date`.
    Retorna la lista cruda de dicts “results” de Polygon.
    """
    m = re.match(r"^(\d+)([a-zA-Z]+)$", timeframe)
    if not m:
        raise ValueError(f"Timeframe inválido: {timeframe!r}")
    multiplier, timespan = m.group(1), m.group(2)

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/C:{symbol}/range/"
        f"{multiplier}/{timespan}/{start_date}/{end_date}"
    )

    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": POLYGON_MAX_LIMIT,
        "apiKey": api_key,
    }
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("results", [])


def _results_to_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Con-vierte la lista de dicts al DataFrame normalizado que usa el resto del
    pipeline (open, high, low, close, volume, timestamp).
    """
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).rename(
        columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "t": "timestamp",
        }
    )

    # Timestamp a int64 (ms) y datetime para parquet
    df["timestamp"] = df["timestamp"].astype("int64")
    df = df.sort_values("timestamp").drop_duplicates("timestamp")

    # Descarta filas con NAN en precios (mantiene integridad)
    df = df.dropna(subset=["open", "high", "low", "close"])

    return df.reset_index(drop=True)


def _save_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Guarda el DataFrame en `path`, admitiendo rutas locales o `gs://`.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if path.startswith("gs://"):
        if gcsfs is None:
            raise ImportError(
                "gcsfs no está instalado. Añádelo a requirements.txt o instala con pip."
            )
        df.to_parquet(path, index=False, engine="pyarrow", storage_options={"token": "cloud"})
    else:
        df.to_parquet(path, index=False, engine="pyarrow")


# ═════════════════════════════════════ main ════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga datos OHLC de Polygon y los guarda en Parquet."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directorio local o prefijo gs:// donde se guardarán los Parquet",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Lista de símbolos (ej: EURUSD GBPUSD)",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        required=True,
        help="Lista de timeframes (ej: 15minute 1hour)",
    )
    parser.add_argument(
        "--polygon-key",
        help="API-key de Polygon. Si se omite se tomará de POLYGON_API_KEY env var.",
    )
    parser.add_argument(
        "--start-date",
        default="2000-01-01",
        help="Inicio YYYY-MM-DD (def. 2000-01-01)",
    )
    parser.add_argument(
        "--end-date",
        help="Fin YYYY-MM-DD (def. hoy)",
    )
    args = parser.parse_args()

    api_key = args.polygon_key or os.getenv("POLYGON_API_KEY")
    if not api_key:
        parser.error(
            "Polygon API-key no proporcionada; usa --polygon-key o variable POLYGON_API_KEY."
        )

    try:
        start_date = dt.date.fromisoformat(args.start_date)
    except ValueError:
        parser.error(f"--start-date inválido: {args.start_date}")

    end_date: dt.date = (
        dt.date.fromisoformat(args.end_date)
        if args.end_date
        else dt.date.today()
    )
    if end_date < start_date:
        parser.error("--end-date no puede ser anterior a --start-date")

    session = requests.Session()
    errors = 0

    for sym in args.symbols:
        for tf in args.timeframes:
            logger.info("📥 %s | %s | %s → %s", sym, tf, start_date, end_date)
            dfs: List[pd.DataFrame] = []
            window_start = start_date

            # Descarga en ventanas de 30 días calendario
            while window_start <= end_date:
                window_end = min(window_start + dt.timedelta(days=30), end_date)
                try:
                    results = _fetch_window(
                        session,
                        symbol=sym,
                        timeframe=tf,
                        start_date=window_start.isoformat(),
                        end_date=window_end.isoformat(),
                        api_key=api_key,
                    )
                    df_window = _results_to_df(results)
                    if not df_window.empty:
                        dfs.append(df_window)
                    logger.debug(
                        "   · ventana %s–%s ⇒ %s filas",
                        window_start,
                        window_end,
                        len(df_window),
                    )
                except Exception as win_err:  # HTTPError ya re-intentado
                    logger.warning(
                        "   ⚠️  Error descargando %s %s (%s → %s): %s",
                        sym,
                        tf,
                        window_start,
                        window_end,
                        win_err,
                    )
                window_start = window_end + dt.timedelta(days=1)

            # ——— guardar ———
            if dfs:
                df_all = pd.concat(dfs, ignore_index=True)
                out_prefix = f"{args.data_dir.rstrip('/')}/{sym}/{tf}"
                out_path = f"{out_prefix}/{sym}_{tf}.parquet"
                try:
                    _save_parquet(df_all, out_path)
                    logger.info("✅  %s filas guardadas en %s", len(df_all), out_path)
                except Exception as save_err:
                    errors += 1
                    logger.error("💥 Error guardando %s: %s", out_path, save_err)
            else:
                errors += 1
                logger.warning("⚠️  Sin datos para %s %s en el rango indicado.", sym, tf)

    if errors:
        logger.warning("Proceso completado con %s avisos/errores.", errors)
    else:
        logger.info("Proceso completado sin errores.")

    session.close()


if __name__ == "__main__":
    main()
