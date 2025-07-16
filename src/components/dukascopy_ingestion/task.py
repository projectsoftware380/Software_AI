# src/components/dukascopy_ingestion/task.py
"""
Tarea del componente de ingestión de datos desde Dukascopy.
Puede funcionar en modo KFP (guardando localmente para que KFP lo suba)
o en modo local (subiendo directamente a GCS).
"""

import argparse
import datetime as dt
import logging
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

from src.components.dukascopy_ingestion.dukascopy_utils import DukascopyDownloader, DataProcessor
from src.shared.gcs_utils import upload_gcs_file # Importar la utilidad de GCS
from src.shared.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def download_data_from_dukascopy(symbol: str, timeframe: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    downloader = DukascopyDownloader()
    processor = DataProcessor()
    logger.info(f"Descargando datos de Dukascopy para {symbol} ({timeframe}) desde {start_date} hasta {end_date}...")
    raw_data = downloader.download_data(symbol, timeframe, start_date, end_date)
    if not raw_data:
        logger.warning(f"No se obtuvieron datos crudos para {symbol}.")
        return pd.DataFrame()
    df = processor.process_raw_data(raw_data, symbol, timeframe)
    logger.info(f"Datos de Dukascopy procesados: {len(df)} filas.")
    return df

def save_df_locally_and_verify(df: pd.DataFrame, local_output_path: Path):
    logger.info(f"Guardando DataFrame en archivo local: {local_output_path}")
    try:
        local_output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(local_output_path, index=True, engine="pyarrow")
        if not local_output_path.exists():
            raise FileNotFoundError(f"¡VERIFICACIÓN FALLIDA! El archivo no se encontró: {local_output_path}")
        logger.info(f"✅ VERIFICACIÓN EXITOSA: El archivo existe en {local_output_path}.")
    except Exception as e:
        logger.error(f"Error al guardar o verificar el archivo local: {e}", exc_info=True)
        raise

def run_ingestion(
    pair: str,
    timeframe: str,
    end_date_str: str,
    project_id: str | None = None,
    output_gcs_path: str | None = None,
    # Argumentos de KFP (ahora opcionales)
    local_output_dir: str | None = None,
    output_local_path_output: str | None = None,
    completion_message_path: str | None = None,
) -> bool:
    is_local_run = output_gcs_path is not None
    logger.info(f"Iniciando la tarea de ingestión para {pair}/{timeframe}. Modo de ejecución: {'Local' if is_local_run else 'KFP'}")
    
    status_success = False
    final_message = ""
    
    try:
        end_date = dt.date.fromisoformat(end_date_str)
        start_date = (end_date - dt.timedelta(days=365 * 5))

        df_full = download_data_from_dukascopy(pair, timeframe, start_date, end_date)
        if df_full.empty:
            raise RuntimeError(f"No se descargaron datos de Dukascopy para {pair}.")

        # --- Lógica de guardado ---
        output_file_name = f"dukascopy_{pair}_{timeframe}.parquet"

        if is_local_run:
            # Modo local: guardar en un directorio temporal y subir a GCS
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = Path(tmpdir) / output_file_name
                save_df_locally_and_verify(df_full, local_path)
                
                gcs_destination_path = f"{output_gcs_path}/{pair}/{timeframe}/{output_file_name}"
                logger.info(f"Subiendo {local_path} a {gcs_destination_path}...")
                upload_gcs_file(local_path, gcs_destination_path)
                final_message = f"Ingestión para {pair} completada. {len(df_full)} filas subidas a {gcs_destination_path}"
        else:
            # Modo KFP: guardar en la ruta que KFP proporciona
            output_dir_path = Path(local_output_dir)
            local_path = output_dir_path / output_file_name
            save_df_locally_and_verify(df_full, local_path)
            final_message = f"Ingestión para {pair} completada. {len(df_full)} filas guardadas en {local_path}"
            # Escribir las rutas de salida para KFP
            if output_local_path_output:
                Path(output_local_path_output).write_text(str(local_path))

        status_success = True
        logger.info(f"🏁 {final_message}")

    except Exception as e:
        final_message = f"❌ Fallo crítico en la ingestión de {pair}/{timeframe}: {e}"
        logger.exception(final_message)
        status_success = False

    finally:
        # Escribir mensaje de completado para KFP
        if completion_message_path:
            Path(completion_message_path).parent.mkdir(parents=True, exist_ok=True)
            Path(completion_message_path).write_text(final_message, encoding="utf-8")

    return status_success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task de Ingestión de Datos desde Dukascopy.")
    # Argumentos comunes
    parser.add_argument("--pair", required=True, help="Par de divisas (ej. EURUSD).")
    parser.add_argument("--timeframe", required=True, help="Timeframe (ej. m1, h1).")
    parser.add_argument("--end-date", dest="end_date_str", default=dt.date.today().isoformat(), help="Fecha de fin (YYYY-MM-DD).")
    
    # Argumentos para ejecución local/híbrida
    parser.add_argument("--project-id", required=False, help="Google Cloud project ID (para modo local).")
    parser.add_argument("--output-gcs-path", required=False, help="Ruta base en GCS para guardar el resultado (para modo local).")

    # Argumentos para ejecución en KFP (se vuelven opcionales)
    parser.add_argument("--local-output-dir", required=False, help="Directorio local de salida (provisto por KFP).")
    parser.add_argument("--output-local-path-output", required=False, help="Ruta al archivo de salida para la ruta del parquet (provisto por KFP).")
    parser.add_argument("--completion-message-path", required=False, help="Ruta al archivo de salida para el mensaje de completado (provisto por KFP).")

    args = parser.parse_args()

    # Validar argumentos según el modo de ejecución
    if args.output_gcs_path: # Modo local
        if not args.project_id:
            raise ValueError("--project-id es requerido para la ejecución en modo local/híbrido.")
    elif not all([args.local_output_dir, args.output_local_path_output, args.completion_message_path]):
        raise ValueError("Se requieren los argumentos --local-output-dir, --output-local-path-output y --completion-message-path para ejecución en modo KFP.")

    logger.info("Componente 'dukascopy_ingestion' iniciado con argumentos: %s", vars(args))

    success = run_ingestion(
        pair=args.pair,
        timeframe=args.timeframe,
        end_date_str=args.end_date_str,
        project_id=args.project_id,
        output_gcs_path=args.output_gcs_path,
        local_output_dir=args.local_output_dir,
        output_local_path_output=args.output_local_path_output,
        completion_message_path=args.completion_message_path,
    )
    
    if not success:
        sys.exit(1)