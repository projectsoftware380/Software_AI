#!/usr/bin/env python3
import os
import re
import argparse
import logging
import requests
import pandas as pd
from pathlib import Path
import datetime

try:
    import gcsfs
except ImportError:
    gcsfs = None  # Para manejar si no se usa GCS en local


def fetch_symbol_timeframe_window(symbol: str, timeframe: str, start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    m = re.match(r"^(\d+)([a-zA-Z]+)$", timeframe)
    if not m:
        raise ValueError(f"Timeframe inválido: {timeframe}")
    multiplier, timespan = m.group(1), m.group(2)
    url_base = f"https://api.polygon.io/v2/aggs/ticker/C:{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"

    all_results = []
    offset = 0
    limit = 50000
    while True:
        params = {
            "adjusted": "true",
            "sort": "asc",
            "apiKey": api_key,
            "limit": limit,
            "offset": offset
        }
        resp = requests.get(url_base, params=params)
        resp.raise_for_status()
        data = resp.json().get("results", [])
        if not data:
            break
        all_results.extend(data)
        if len(data) < limit:
            break
        offset += len(data)
    if not all_results:
        return pd.DataFrame()
    df = pd.DataFrame(all_results)

    # 🆕 MODIFICACIÓN AQUÍ: Renombrar las columnas 'o', 'h', 'l', 'c' a 'open', 'high', 'low', 'close'
    # Esto asegura que los DataFrames guardados en Parquet ya tengan los nombres correctos
    # para tu script de indicadores.
    df = df.rename(columns={
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        't': 'timestamp' # También renombramos 't' a 'timestamp' de forma explícita aquí
    })

    # Si 'timestamp' ya se renombró arriba, esta verificación podría ser redundante
    # if "t" in df.columns: # Ya no es necesario si 't' siempre se renombra
    #    df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
    if 'timestamp' in df.columns: # Asegurarse de que sea datetime si no lo es ya
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    return df


def save_dataframe(df: pd.DataFrame, path: str):
    if path.startswith("gs://"):
        if gcsfs is None:
            raise ImportError("El módulo gcsfs no está instalado. Instálalo con `pip install gcsfs`.")
        df.to_parquet(path, index=False, engine="pyarrow", storage_options={"token": "cloud"})
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Descarga datos de mercado desde Polygon.io y los guarda en Parquet (local o GCS).")
    parser.add_argument("--data-dir", required=True, help="Directorio local o ruta gs:// donde se guardarán los Parquet")
    parser.add_argument("--symbols", nargs="+", required=True, help="Lista de símbolos (ej: EURUSD GBPUSD)")
    parser.add_argument("--timeframes", nargs="+", required=True, help="Lista de timeframes (ej: 15minute 1hour)")
    
    # MODIFICACIÓN CLAVE AQUÍ: Haz que polygon-key no sea required=True
    # y léela de la variable de entorno si no se proporciona como argumento.
    parser.add_argument("--polygon-key", type=str, help="API Key de Polygon.io. Se puede omitir si POLYGON_API_KEY env var está seteada.")
    
    parser.add_argument("--start-date", type=str, default="2000-01-01", help="Fecha de inicio (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="Fecha de fin (YYYY-MM-DD (por defecto hoy)")
    args = parser.parse_args()

    # Obtener la API Key
    api_key = args.polygon_key
    if not api_key: # Si no se proporcionó como argumento CLI
        api_key = os.getenv("POLYGON_API_KEY") # Intentar leer de la variable de entorno
    
    if not api_key:
        raise ValueError("API Key de Polygon.io no proporcionada ni en --polygon-key ni en la variable de entorno POLYGON_API_KEY.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger()

    start = datetime.date.fromisoformat(args.start_date)
    end = datetime.date.fromisoformat(args.end_date) if args.end_date else datetime.date.today()
    symbols = args.symbols
    timeframes = args.timeframes

    for sym in symbols:
        for tf in timeframes:
            logger.info(f"Procesando {sym} | {tf} | {start} → {end}")
            dfs = []
            current = start
            while current <= end:
                window_end = min(current + datetime.timedelta(days=30), end)
                logger.info(f"   🌐 Descargando ventana: {current} → {window_end}")
                df_window = fetch_symbol_timeframe_window(
                    symbol=sym,
                    timeframe=tf,
                    start_date=current.isoformat(),
                    end_date=window_end.isoformat(),
                    api_key=api_key # Usar la API Key obtenida
                )
                if not df_window.empty:
                    dfs.append(df_window)
                current = window_end + datetime.timedelta(days=1)
            if dfs:
                df_all = pd.concat(dfs, ignore_index=True)
                out_path = f"{args.data_dir.rstrip('/')}/{sym}_{tf}.parquet"
                save_dataframe(df_all, out_path)
                logger.info(f"   ✅ Guardado {len(df_all)} registros en {out_path}")
            else:
                logger.warning(f"   ⚠️  No se obtuvieron datos para {sym} {tf}")


if __name__ == "__main__":
    main()