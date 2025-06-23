# src/components/data_ingestion/task.py
"""
Tarea del componente de ingestión de datos. (Versión con Logging Robusto)

Responsabilidades:
1.  Obtener la API Key de Polygon desde Google Secret Manager.
2.  Iterar sobre una lista de pares de divisas predefinida en las constantes.
3.  Para cada par, eliminar el Parquet histórico y descargar los datos desde Polygon.io.
4.  Validar que se ha descargado un número mínimo de registros para cada par.
5.  Guardar los datos consolidados en un nuevo archivo Parquet en GCS.
6.  Enviar notificaciones de éxito o fracaso a un topic de Pub/Sub para cada par.

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

import pandas as pd
import requests
from google.cloud import pubsub_v1, secretmanager, storage
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


# --- Lógica de Acceso a Secret Manager ---
def get_polygon_api_key(
    project_id: str, secret_name: str, version: str = "latest"
) -> str:
    """
    Obtiene la API Key de Polygon desde Google Secret Manager.
    """
    # [LOG] Punto de control de inicio de la función.
    logger.info("Iniciando obtención de API Key desde Secret Manager...")
    try:
        logger.info(
            f"Accediendo al secreto: projects/{project_id}/secrets/{secret_name}/versions/{version}"
        )
        client = secretmanager.SecretManagerServiceClient()
        secret_path = client.secret_version_path(project_id, secret_name, version)
        response = client.access_secret_version(request={"name": secret_path})
        api_key = response.payload.data.decode("UTF-8")
        logger.info(f"✅ API Key obtenida exitosamente de Secret Manager ({secret_name}).")
        return api_key
    except Exception as e:
        # [LOG] Error contextualizado.
        logger.error(f"❌ Error al obtener la API Key de Secret Manager: {e}", exc_info=True)
        raise RuntimeError(
            f"Fallo al obtener la API Key de Secret Manager '{secret_name}': {e}"
        )

# --- Lógica de Descarga de Datos ---
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
    # ... (lógica interna sin cambios) ...
    match = re.match(r"^(\d+)([a-zA-Z]+)$", timeframe)
    if not match:
        raise ValueError(f"Timeframe inválido: {timeframe!r}")
    mult, span = match.groups()

    url = f"https://api.polygon.io/v2/aggs/ticker/C:{symbol}/range/{mult}/{span}/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": POLYGON_MAX_LIMIT,
        "apiKey": api_key,
    }
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json().get("results", [])


def _results_to_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    # ... (lógica interna sin cambios) ...
    if not results:
        return pd.DataFrame()
    df = (
        pd.DataFrame(results)
        .rename(
            columns={
                "o": "open", "h": "high", "l": "low", "c": "close",
                "v": "volume", "t": "timestamp",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
        .assign(timestamp=lambda d: pd.to_datetime(d["timestamp"], unit="ms"))
        .set_index("timestamp")
        .sort_index()
    )
    df = df[~df.index.duplicated(keep="first")]
    return df

def upload_df_to_gcs_and_verify(df: pd.DataFrame, gcs_uri: str, local_path: Path):
    # ... (lógica interna sin cambios, ya tiene buen logging) ...
    logger.info(f"Guardando DataFrame en archivo local temporal: {local_path}")
    df.to_parquet(local_path, index=True, engine="pyarrow")
    
    logger.info(f"Subiendo archivo a GCS: {gcs_uri}")
    gcs_utils.upload_gcs_file(local_path, gcs_uri)
    
    logger.info(f"Verificando la existencia del objeto en GCS: {gcs_uri}")
    if not gcs_utils.gcs_path_exists(gcs_uri):
        raise FileNotFoundError(
            f"¡VERIFICACIÓN FALLIDA! El objeto no se encontró en GCS después de la subida: {gcs_uri}"
        )
    logger.info(f"✅ VERIFICACIÓN EXITOSA: El objeto existe en {gcs_uri}.")


# --- Orquestación Principal de la Tarea ---
def run_ingestion(
    pair: str,
    timeframe: str,
    project_id: str,
    gcs_data_path: str,
    polygon_secret: str,
    start_date: str,
    end_date: str,
    min_rows: int,
    api_key: str,
) -> bool:
    status_success = False
    try:
        # [LOG] Registrar la ruta de salida que se va a utilizar.
        parquet_uri = f"{gcs_data_path}/{pair}/{timeframe}/{pair}_{timeframe}.parquet"
        logger.info(f"Ruta de destino del artefacto: {parquet_uri}")

        if gcs_utils.gcs_path_exists(parquet_uri):
            logger.info(f"El archivo previo existe. Eliminándolo...")
            gcs_utils.delete_gcs_blob(parquet_uri)
            logger.info("Archivo previo eliminado.")
        else:
            logger.info("No existía un Parquet previo. Se creará uno nuevo.")

        logger.info(f"📥 Iniciando descarga para {pair} | {timeframe} | {start_date} → {end_date}")
        all_dfs: List[pd.DataFrame] = []
        win_start = dt.date.fromisoformat(start_date)
        final_end = dt.date.fromisoformat(end_date)
        session = requests.Session()

        while win_start <= final_end:
            win_end = min(win_start + dt.timedelta(days=30), final_end)
            try:
                results = _fetch_window(
                    session, pair, timeframe, win_start.isoformat(), win_end.isoformat(), api_key
                )
                df_window = _results_to_df(results)
                if not df_window.empty:
                    all_dfs.append(df_window)
            except Exception as e:
                logger.warning(f"⚠️  Error en ventana {win_start}-{win_end} para {pair}: {e}")
            win_start = win_end + dt.timedelta(days=1)
        
        session.close()

        if not all_dfs:
            raise RuntimeError(f"No se descargaron datos para {pair}. La lista de DataFrames está vacía.")

        df_full = pd.concat(all_dfs, ignore_index=False).drop_duplicates()
        # [LOG] Registrar el estado de los datos después de la consolidación.
        logger.info(f"Consolidados {len(df_full):,} registros en total para {pair}.")

        if len(df_full) < min_rows:
            raise RuntimeError(
                f"Descarga incompleta para {pair}: {len(df_full):,} filas (< {min_rows:,}). "
                "Se aborta para evitar un Parquet inválido."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            local_parquet_path = Path(tmpdir) / f"{pair}_{timeframe}.parquet"
            upload_df_to_gcs_and_verify(df_full, parquet_uri, local_parquet_path)

        status_success = True
        logger.info(f"🏁 Descarga y guardado para {pair}/{timeframe} completados con éxito.")

    except Exception as e:
        # [LOG] Captura de error fatal con contexto completo.
        logger.critical(f"❌ Fallo crítico en la ingestión de {pair}/{timeframe}: {e}", exc_info=True)
        status_success = False

    finally:
        # [LOG] El bloque finally asegura que esto siempre se ejecute.
        logger.info("Iniciando proceso de notificación a Pub/Sub...")
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
            logger.info(f"✅ Notificación enviada exitosamente a topic '{topic_id}' para {pair}.")
        except Exception as pubsub_err:
            logger.error(f"❌ Error al enviar notificación a Pub/Sub para {pair}: {pubsub_err}")

    return status_success

# --- Punto de Entrada para Ejecución como Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task de Ingestión de Datos para KFP.")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--project-id", default=constants.PROJECT_ID)
    parser.add_argument("--gcs-data-path", default=constants.RAW_DATA_PATH)
    parser.add_argument("--polygon-secret-name", default=constants.POLYGON_API_KEY_SECRET_NAME)
    parser.add_argument("--start-date", default="2010-01-01")
    parser.add_argument("--end-date", default=dt.date.today().isoformat())
    parser.add_argument("--min-rows", type=int, default=100_000)
    parser.add_argument("--completion-message-path", type=Path, required=True)
    
    args = parser.parse_args()

    # [LOG] Registro de los argumentos recibidos al iniciar el script.
    logger.info("Componente 'data_ingestion' iniciado con los siguientes argumentos:")
    for key, value in vars(args).items():
        logger.info(f"  - {key}: {value}")

    api_key = get_polygon_api_key(args.project_id, args.polygon_secret_name)
    
    success = run_ingestion(
        pair=args.pair,
        timeframe=args.timeframe,
        project_id=args.project_id,
        gcs_data_path=args.gcs_data_path,
        polygon_secret=args.polygon_secret_name,
        start_date=args.start_date,
        end_date=args.end_date,
        min_rows=args.min_rows,
        api_key=api_key
    )
    
    message = f"Ingestion for {args.pair}: {'SUCCESS' if success else 'FAILURE'}"
    args.completion_message_path.parent.mkdir(parents=True, exist_ok=True)
    args.completion_message_path.write_text(message)
    logger.info(f"Mensaje de finalización escrito: '{message}'")
    
    if not success:
        sys.exit(1)
    else:
        sys.exit(0)