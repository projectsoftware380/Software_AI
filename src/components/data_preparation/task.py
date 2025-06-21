S# task.py: Componente de Preparación de Datos
# -----------------------------------------------------------------------------
# Este script orquesta la carga de datos crudos, su limpieza, resampling,
# la creación de un conjunto de hold-out para validación final, y la subida
# de los artefactos resultantes (datasets de entrenamiento y hold-out) a
# Google Cloud Storage.
# -----------------------------------------------------------------------------

import argparse
import logging
from datetime import datetime
from pathlib import Path
import tempfile

import pandas as pd
from google.cloud import storage

# Módulos internos
from src.shared import constants, gcs_utils
from src.shared.gcs_utils import clean_and_resample, create_holdout_set

# Configuración del logging para una salida clara y estandarizada
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


def upload_df_to_gcs_and_verify(df: pd.DataFrame, gcs_uri: str) -> None:
    """
    Sube un DataFrame a una URI de GCS como Parquet y verifica que la subida
    fue exitosa.

    Args:
        df: El DataFrame de pandas a subir.
        gcs_uri: La ruta completa en GCS (gs://...) donde se guardará el archivo.

    Raises:
        FileNotFoundError: Si después de la operación de escritura, el archivo
                           no se puede encontrar en GCS.
        Exception: Cualquier otro error durante el proceso de subida.
    """
    logging.info(f"Intentando subir DataFrame a {gcs_uri}...")
    try:
        # Usa pandas para escribir el archivo directamente a GCS.
        # gcsfs debe estar instalado para que esto funcione.
        df.to_parquet(gcs_uri, engine="pyarrow", index=True)
        logging.info(
            f"La operación de escritura a {gcs_uri} se completó sin errores."
        )

        # --- Bloque de Verificación Crítico ---
        logging.info(f"Verificando la existencia del objeto en GCS: {gcs_uri}")
        client = storage.Client()
        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            # Si el archivo no existe después de la subida, es un error fatal.
            # Esto detendrá el pipeline en el lugar correcto.
            raise FileNotFoundError(
                f"¡VERIFICACIÓN FALLIDA! El objeto no se encontró en GCS "
                f"después de la subida: {gcs_uri}"
            )

        logging.info(f"✅ VERIFICACIÓN EXITOSA: El objeto existe en {gcs_uri}.")
        # --- Fin de la Verificación ---

    except Exception as e:
        logging.error(f"Error fatal durante la subida o verificación a GCS: {e}")
        raise


def run_data_preparation(
    pair: str,
    timeframe: str,
    years_to_keep: int,
    holdout_months: int,
    cleanup: bool,
    prepared_data_path_output: str,
    holdout_data_path_output: str,
):
    """
    Orquesta todo el proceso de preparación de datos.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Descargar los datos crudos desde GCS
        local_raw_path = gcs_utils.download_gcs_file(
            f"gs://{constants.BUCKET_NAME}/{constants.RAW_DATA_PATH}/{timeframe}/{pair}.parquet",
            tmp_path,
        )
        if local_raw_path is None:
            raise FileNotFoundError(f"No se pudo descargar el archivo de datos crudos para {pair}.")
            
        df_raw = pd.read_parquet(local_raw_path)

        # Procesar y dividir los datos
        df_processed = clean_and_resample(df_raw, timeframe)
        df_train, df_holdout = create_holdout_set(df_processed, holdout_months)
        logging.info(
            f"Set de entrenamiento: {df_train.shape[0]} filas | "
            f"Set de Hold-out: {df_holdout.shape[0]} filas"
        )

        # Crear rutas de salida versionadas con timestamp
        version_ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        output_base_path = f"gs://{constants.BUCKET_NAME}/{constants.FEATURES_PATH}/{pair}/{timeframe}/{version_ts}"

        prepared_data_path = f"{output_base_path}/{pair}_{timeframe}_train_opt.parquet"
        holdout_data_path = f"{output_base_path}/{pair}_{timeframe}_holdout.parquet"

        # Subir los dataframes a GCS con la nueva función de verificación
        upload_df_to_gcs_and_verify(df_train, prepared_data_path)
        upload_df_to_gcs_and_verify(df_holdout, holdout_data_path)

        # Escribir las rutas de salida para que KFP las pase a los siguientes componentes
        Path(prepared_data_path_output).parent.mkdir(parents=True, exist_ok=True)
        Path(prepared_data_path_output).write_text(prepared_data_path)

        Path(holdout_data_path_output).parent.mkdir(parents=True, exist_ok=True)
        Path(holdout_data_path_output).write_text(holdout_data_path)

        logging.info(
            "Rutas de salida escritas para KFP: "
            f"Train='{prepared_data_path}', Holdout='{holdout_data_path}'"
        )

        if cleanup:
            logging.info("La lógica de limpieza de versiones antiguas no está implementada.")
            # Aquí iría la lógica para encontrar y eliminar directorios antiguos
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepara datos para entrenamiento y backtesting."
    )
    parser.add_argument("--pair", required=True, help="Par de divisas a procesar.")
    parser.add_argument("--timeframe", required=True, help="Timeframe de los datos.")
    parser.add_argument(
        "--years-to-keep",
        type=int,
        default=5,
        help="Años de datos a mantener antes del holdout.",
    )
    parser.add_argument(
        "--holdout-months",
        type=int,
        default=3,
        help="Meses para reservar para el set de hold-out.",
    )
    parser.add_argument(
        "--cleanup", type=bool, default=True, help="Activar limpieza de versiones antiguas."
    )
    # Argumentos de salida de KFP
    parser.add_argument(
        "--prepared-data-path-output",
        required=True,
        help="Ruta donde guardar la URI del dataset de entrenamiento.",
    )
    parser.add_argument(
        "--holdout-data-path-output",
        required=True,
        help="Ruta donde guardar la URI del dataset de hold-out.",
    )

    args = parser.parse_args()

    run_data_preparation(
        pair=args.pair,
        timeframe=args.timeframe,
        years_to_keep=args.years_to_keep,
        holdout_months=args.holdout_months,
        cleanup=args.cleanup,
        prepared_data_path_output=args.prepared_data_path_output,
        holdout_data_path_output=args.holdout_data_path_output,
    )