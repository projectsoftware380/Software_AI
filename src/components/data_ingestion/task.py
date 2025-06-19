# src/components/data_ingestion/task.py
"""
Tarea del componente de ingestión de datos.

Responsabilidades:
1.  Iterar sobre una lista de pares de divisas predefinida en las constantes.
2.  Para cada par, eliminar el Parquet histórico y descargar los datos desde Dukascopy.
3.  Validar que se ha descargado un número mínimo de registros para cada par.
4.  Guardar los datos consolidados en un nuevo archivo Parquet en GCS.
5.  Enviar notificaciones de éxito o fracaso a un topic de Pub/Sub para cada par.

Este script está diseñado para ser ejecutado por un componente de KFP.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import asyncio
import pandas as pd
import dukascopy_python as dukascopy
from google.cloud import pubsub_v1
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Importar los módulos compartidos de la nueva estructura
from src.shared import constants, gcs_utils

# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# --- Lógica de Descarga de Datos desde Dukascopy ---


async def _fetch_window(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Descarga una ventana de datos desde Dukascopy."""
    tf_map = {"15minute": dukascopy.INTERVAL_MIN_15, "1hour": dukascopy.INTERVAL_HOUR_1}
    if timeframe not in tf_map:
        raise ValueError(f"Timeframe inválido: {timeframe!r}")

    start_dt = dt.datetime.fromisoformat(start_date)
    end_dt = dt.datetime.fromisoformat(end_date)

    df = await asyncio.to_thread(
        dukascopy.fetch,
        symbol,
        tf_map[timeframe],
        dukascopy.OFFER_SIDE_BID,
        start_dt,
        end_dt,
    )

    df = df.reset_index()
    df = df.rename(columns={"timestamp": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).view("int64") // 1_000_000
    df = df[["open", "high", "low", "close", "volume", "timestamp"]]
    return df




# --- Orquestación Principal de la Tarea ---
def run_ingestion(
    pair: str,
    timeframe: str,
    project_id: str,
    gcs_data_path: str,
    start_date: str,
    end_date: str,
    min_rows: int,
) -> bool:
    """
    Orquesta el proceso completo de ingestión de datos para un par/timeframe.
    """
    status_success = False
    try:
        # 1. Definir ruta de salida y eliminar versión anterior
        parquet_uri = f"{gcs_data_path}/{pair}/{timeframe}/{pair}_{timeframe}.parquet"
        if gcs_utils.gcs_path_exists(parquet_uri):
            gcs_utils.delete_gcs_blob(parquet_uri)
        else:
            logger.info(f"No existía un Parquet previo en {parquet_uri}. Se creará uno nuevo.")

        # 2. Descargar datos por ventanas
        logger.info(f"📥 Descargando {pair} | {timeframe} | {start_date} → {end_date}")
        all_dfs: List[pd.DataFrame] = []
        win_start = dt.date.fromisoformat(start_date)
        final_end = dt.date.fromisoformat(end_date)

        while win_start <= final_end:
            win_end = min(win_start + dt.timedelta(days=30), final_end)
            try:
                df_window = asyncio.run(
                    _fetch_window(
                        pair,
                        timeframe,
                        win_start.isoformat(),
                        win_end.isoformat(),
                    )
                )
                if not df_window.empty:
                    all_dfs.append(df_window)
            except Exception as e:
                logger.warning(
                    f"⚠️  Error en ventana {win_start}-{win_end} para {pair}: {e}"
                )
            win_start = win_end + dt.timedelta(days=1)

        # 3. Consolidar y validar
        if not all_dfs:
            raise RuntimeError(f"No se descargaron datos para {pair}. La lista de DataFrames está vacía.")

        df_full = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Consolidados {len(df_full):,} registros en total para {pair}.")

        if len(df_full) < min_rows:
            raise RuntimeError(
                f"Descarga incompleta para {pair}: {len(df_full):,} filas (< {min_rows:,}). "
                "Se aborta para evitar un Parquet inválido."
            )

        # 4. Guardar en GCS
        with tempfile.TemporaryDirectory() as tmpdir:
            local_parquet_path = Path(tmpdir) / f"{pair}_{timeframe}.parquet"
            df_full.to_parquet(local_parquet_path, index=False, engine="pyarrow")
            gcs_utils.upload_gcs_file(local_parquet_path, parquet_uri)

        status_success = True
        logger.info(f"🏁 Descarga para {pair}/{timeframe} completada con éxito.")

    except Exception as e:
        logger.critical(f"❌ Fallo crítico en la ingestión de {pair}/{timeframe}: {e}", exc_info=True)
        status_success = False

    finally:
        # 5. Enviar notificación a Pub/Sub
        topic_id = constants.SUCCESS_TOPIC_ID if status_success else constants.FAILURE_TOPIC_ID
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(project_id, topic_id)
        
        payload = {
            "source": "data_ingestion_component",
            "status": "SUCCESS" if status_success else "FAILURE",
            "pair": pair,
            "timeframe": timeframe,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        
        try:
            future = publisher.publish(topic_path, json.dumps(payload).encode("utf-8"))
            future.result(timeout=60)
            logger.info(f"Notificación enviada a topic '{topic_id}' para {pair}.")
        except Exception as pubsub_err:
            logger.error(f"Error al enviar notificación a Pub/Sub para {pair}: {pubsub_err}")

    return status_success


# --- Punto de Entrada para Ejecución como Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task de Ingestión de Datos para KFP.")
    
    # ``--pair`` se mantiene opcional para compatibilidad con pruebas unitarias.
    parser.add_argument("--pair")
    parser.add_argument("--timeframe", required=True, help="Timeframe, ej: 15minute")
    parser.add_argument("--project-id", default=constants.PROJECT_ID)
    parser.add_argument("--gcs-data-path", default=constants.DATA_PATH)
    parser.add_argument("--start-date", default="2010-01-01")
    parser.add_argument("--end-date", default=dt.date.today().isoformat())
    parser.add_argument("--min-rows", type=int, default=100_000)
    parser.add_argument(
        "--completion-message-path",
        type=Path,
        default=Path(tempfile.gettempdir()) / "ingest_done.txt",
        help="Ruta de archivo donde se escribirá el mensaje de salida."
    )
    
    args = parser.parse_args()

    # Obtener la lista de pares del archivo de constantes
    pairs_to_ingest = list(constants.SPREADS_PIP.keys())
    logger.info(f"Iniciando ingestión para los siguientes pares: {pairs_to_ingest}")
    
    # Dukascopy no requiere autenticación con API key
    
    results = {}
    for pair in pairs_to_ingest:
        logger.info(f"--- Procesando par: {pair} ---")
        success = run_ingestion(
            pair=pair,
            timeframe=args.timeframe,
            project_id=args.project_id,
            gcs_data_path=args.gcs_data_path,
            start_date=args.start_date,
            end_date=args.end_date,
            min_rows=args.min_rows,
        )
        results[pair] = "SUCCESS" if success else "FAILURE"

    # Preparar mensaje final de salida
    successful_pairs = [p for p, s in results.items() if s == "SUCCESS"]
    failed_pairs = [p for p, s in results.items() if s == "FAILURE"]
    
    message = f"Ingestión completada. Éxito: {len(successful_pairs)} pares. Fallo: {len(failed_pairs)} pares."
    if failed_pairs:
        message += f" Pares fallidos: {', '.join(failed_pairs)}"

    args.completion_message_path.parent.mkdir(parents=True, exist_ok=True)
    args.completion_message_path.write_text(message)
    
    # Registrar el resumen final y salir siempre con código 0 para evitar
    # que el pipeline falle por pares individuales.
    if failed_pairs:
        logger.warning(
            "Ingestión completada con fallos en los pares: %s", ", ".join(failed_pairs)
        )
    else:
        logger.info("Ingestión completada sin fallos.")

    sys.exit(0)
