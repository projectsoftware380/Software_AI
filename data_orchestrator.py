#!/usr/bin/env python3
"""
data_orchestrator.py
────────────────────
Orquesta la descarga y actualización de datos de mercado históricos
desde Polygon.io a Google Cloud Storage (GCS) usando data_fetcher_vertexai.py.

Diseñado para ser ejecutado como un Cloud Run Job o un componente de Vertex AI Pipeline.
Puede ser activado por Cloud Scheduler (actualización programada) o por mensajes de Pub/Sub
(solicitudes bajo demanda).

Funcionalidades:
1.  Lee la configuración de símbolos/timeframes desde un JSON en GCS.
2.  Para cada par/timeframe, verifica si el archivo Parquet existe en GCS.
3.  Determina la fecha de inicio de la descarga (completa si no existe, incremental si existe).
4.  Invoca data_fetcher_vertexai.py con la lógica de reintentos.
5.  Publica mensajes de éxito/fallo en topics de Pub/Sub.
6.  Lee la API Key de Polygon.io de Secret Manager (inyectada como variable de entorno).
"""

import os
import sys
import json
import time
import subprocess
import argparse
from datetime import datetime, date, timedelta
from pathlib import Path
import logging

# Instalar gcsfs y google-cloud-storage si no están presentes
# (Aunque en la imagen Docker ya deberían estar instalados, es una buena práctica defensiva)
try:
    import gcsfs
    from google.cloud import storage
    from google.cloud import pubsub_v1
    from google.oauth2 import service_account
    from google.auth.exceptions import DefaultCredentialsError
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gcsfs", "google-cloud-storage", "google-cloud-pubsub"])
    import gcsfs
    from google.cloud import storage
    from google.cloud import pubsub_v1
    from google.oauth2 import service_account
    from google.auth.exceptions import DefaultCredentialsError

import pandas as pd # Necesario para leer el último timestamp del parquet

# ---------------- Configuración Global -------------------------------------
# Configura logging para que sea visible en Cloud Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Nombre del bucket de GCS. Se puede sobreescribir con una variable de entorno.
# Usamos el nombre que ya confirmaste: trading-ai-models-460823
GCS_BUCKET = os.getenv("GCS_BUCKET", "trading-ai-models-460823") 
CONFIG_PATH = f"gs://{GCS_BUCKET}/config/data_fetch_config.json"
DATA_BASE_PATH = f"gs://{GCS_BUCKET}/data"

# ID del proyecto de GCP. Se obtiene de las variables de entorno de Cloud Run.
PUBSUB_PROJECT_ID = os.getenv("GCP_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT"))
# Topics de Pub/Sub (asegúrate de que existan en tu proyecto GCP)
SUCCESS_TOPIC_ID = "data-ingestion-success"
FAILURE_TOPIC_ID = "data-ingestion-failures"
# Topic para solicitudes bajo demanda (el orquestador estará suscrito a este)
REQUEST_TOPIC_ID = "data-ingestion-requests"

# ---------------- Helpers para GCS y Pub/Sub -----------------------------

def get_gcs_client():
    """
    Obtiene un cliente de GCS, usando credenciales de cuenta de servicio
    si GOOGLE_APPLICATION_CREDENTIALS está definida, o por defecto si es en GCP.
    """
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            creds = service_account.Credentials.from_service_account_file(
                os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            )
            return storage.Client(credentials=creds)
        except Exception as e:
            logger.error(f"Error al cargar credenciales de GOOGLE_APPLICATION_CREDENTIALS: {e}")
            # Si hay un problema con el archivo, intentar con credenciales predeterminadas de GCP
            pass 
    # Intentar obtener credenciales por defecto (funciona en GCP, o si gcloud auth application-default login se usó localmente)
    try:
        return storage.Client()
    except DefaultCredentialsError as e:
        logger.error(f"No se pudieron obtener credenciales predeterminadas de GCP: {e}. Asegúrate de ejecutar 'gcloud auth application-default login' o de configurar GOOGLE_APPLICATION_CREDENTIALS.")
        raise # Relanzar para abortar si no hay credenciales

def get_pubsub_publisher_client():
    """Obtiene un cliente de Pub/Sub para publicar mensajes."""
    # El cliente de Pub/Sub automáticamente busca las credenciales predeterminadas.
    return pubsub_v1.PublisherClient()

def download_config_from_gcs(gcs_path: str) -> dict:
    """Descarga y lee un archivo JSON de configuración desde GCS."""
    try:
        client = get_gcs_client()
        bucket_name, blob_name = gcs_path[5:].split("/", 1)
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        config_content = blob.download_as_text()
        return json.loads(config_content)
    except Exception as e:
        logger.error(f"Error al descargar o parsear la configuración de GCS desde {gcs_path}: {e}")
        raise

def check_parquet_exists(gcs_parquet_path: str) -> bool:
    """Verifica si un archivo Parquet existe en GCS."""
    try:
        # gcsfs.GCSFileSystem.exists() intentará usar las credenciales adecuadas.
        fs = gcsfs.GCSFileSystem(token="cloud") # 'token="cloud"' le dice a gcsfs que busque las credenciales de GCP.
        return fs.exists(gcs_parquet_path)
    except Exception as e:
        # Aquí, si falla el chequeo de existencia, significa que no hay credenciales para GCSFS
        # o que hay un problema de red al acceder al servicio de credenciales.
        logger.critical(f"Error CRÍTICO al verificar la existencia de {gcs_parquet_path}: {e}. La ejecución de GCS fallará.")
        raise # Abortar si no se puede verificar la existencia

def get_latest_timestamp_from_parquet(gcs_parquet_path: str) -> str:
    """Lee el último timestamp de un archivo Parquet en GCS."""
    try:
        # Usar gcsfs para que pandas pueda leer directamente de GCS.
        # Depende de que las credenciales de GCS estén configuradas correctamente.
        df = pd.read_parquet(gcs_parquet_path, engine="pyarrow", storage_options={"token": "cloud"})
        if "timestamp" in df.columns and not df["timestamp"].empty:
            # Asegurarse de que el timestamp sea un tipo datetime antes de encontrar el máximo
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            latest_ts = df["timestamp"].max()
            # Devolver la fecha como string YYYY-MM-DD
            return latest_ts.strftime('%Y-%m-%d')
        logger.warning(f"No se encontró la columna 'timestamp' o está vacía en {gcs_parquet_path}. Asumiendo que no se pudo leer el último timestamp.")
        return None
    except Exception as e:
        logger.error(f"Error al leer el último timestamp de {gcs_parquet_path}: {e}. Esto podría ser por problemas de credenciales o archivo corrupto.")
        return None

def publish_message(topic_id: str, message_data: dict):
    """Publica un mensaje JSON en un topic de Pub/Sub."""
    publisher = get_pubsub_publisher_client()
    topic_path = publisher.topic_path(PUBSUB_PROJECT_ID, topic_id)
    message_json = json.dumps(message_data)
    message_bytes = message_json.encode("utf-8")

    try:
        future = publisher.publish(topic_path, message_bytes)
        message_id = future.result()
        logger.info(f"Mensaje publicado en {topic_id} con ID: {message_id}")
    except Exception as e:
        logger.error(f"Error al publicar mensaje en {topic_id}: {e}")

# ---------------- Lógica Principal de Orquestación -----------------------

def process_data_fetch(symbol: str, timeframe: str, config: dict, env_vars_for_subprocess: dict):
    """
    Procesa la descarga de datos para un símbolo y timeframe específico,
    incluyendo la lógica de reintentos.
    env_vars_for_subprocess: Diccionario de variables de entorno a pasar al subproceso.
    """
    gcs_parquet_file = f"{DATA_BASE_PATH}/{symbol}/{timeframe}/{symbol}_{timeframe}.parquet"
    start_date_to_fetch = None
    end_date_to_fetch = date.today().isoformat()

    logger.info(f"Procesando {symbol} | {timeframe}...")

    # Determinar start_date
    try:
        if check_parquet_exists(gcs_parquet_file):
            latest_ts_str = get_latest_timestamp_from_parquet(gcs_parquet_file)
            if latest_ts_str:
                # Añadir un día para no duplicar el último registro
                start_date_to_fetch = (datetime.fromisoformat(latest_ts_str).date() + timedelta(days=1)).isoformat()
                logger.info(f"  Parquet existente. Último dato: {latest_ts_str}. Iniciando descarga incremental desde: {start_date_to_fetch}")
                # Si el start_date calculado es posterior a la fecha actual, los datos ya están al día.
                if datetime.fromisoformat(start_date_to_fetch).date() > date.today():
                    logger.info(f"  Datos para {symbol} | {timeframe} ya están actualizados. Saltando descarga.")
                    return True # Éxito, no hay nada que hacer
            else:
                logger.warning(f"  No se pudo obtener el último timestamp del Parquet existente. Forzando descarga completa para {symbol} | {timeframe}.")
                start_date_to_fetch = config["default_start_date_new_parquet"]
        else:
            logger.info(f"  Parquet no existente. Iniciando descarga completa desde: {config['default_start_date_new_parquet']}")
            start_date_to_fetch = config["default_start_date_new_parquet"]
    except Exception as e:
        logger.critical(f"Error al determinar el estado del parquet para {symbol} | {timeframe}: {e}. Abortando este par.")
        return False # No podemos ni determinar el estado del parquet, es un fallo crítico para este par.


    # Reintentos para la llamada a data_fetcher_vertexai.py
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"  Intento {attempt}/{max_retries}: Llamando a data_fetcher_vertexai.py para {symbol} | {timeframe} ({start_date_to_fetch} a {end_date_to_fetch})...")
            
            command = [
                sys.executable,  # Usa el intérprete de Python actual
                "data_fetcher_vertexai.py", # Asegúrate que este script es ejecutable o está en PATH
                "--data-dir", DATA_BASE_PATH,
                "--symbols", symbol,
                "--timeframes", timeframe,
                "--start-date", start_date_to_fetch,
                "--end-date", end_date_to_fetch,
                "--polygon-key", os.getenv("POLYGON_API_KEY") # API Key inyectada por Secret Manager
            ]
            
            # Capturar la salida para logging
            # Pasa las variables de entorno al subproceso
            result = subprocess.run(command, capture_output=True, text=True, check=False, env=env_vars_for_subprocess)

            if result.returncode == 0:
                logger.info(f"  ✅ data_fetcher_vertexai.py completado para {symbol} | {timeframe}.")
                logger.info(f"  STDOUT: {result.stdout}")
                return True # Éxito
            else:
                logger.error(f"  ❌ data_fetcher_vertexai.py falló para {symbol} | {timeframe} (Intento {attempt}):")
                logger.error(f"  STDERR: {result.stderr}")
                if "429 Too Many Requests" in result.stderr:
                    logger.warning("  Posible límite de tasa de Polygon.io. Reintentando...")
                elif "401 Unauthorized" in result.stderr:
                    logger.critical("  API Key de Polygon.io inválida. Deteniendo reintentos.")
                    return False # Error fatal, no reintentar
                elif "Invalid gcloud credentials" in result.stderr or "Failed to retrieve http://metadata.google.internal" in result.stderr:
                    logger.critical("  Error de credenciales de GCS en subproceso. Asegúrate de que GOOGLE_APPLICATION_CREDENTIALS esté configurada correctamente.")
                    return False # Error fatal, no reintentar este tipo de error

        except FileNotFoundError:
            logger.critical("  Error: data_fetcher_vertexai.py no encontrado. Asegúrate de que esté en la imagen Docker.")
            return False # Error fatal

        except Exception as e:
            logger.error(f"  Error inesperado durante el intento {attempt} para {symbol} | {timeframe}: {e}")

        # Espera exponencial antes del próximo reintento
        time.sleep(2 ** attempt)

    logger.error(f"  ❌ Fallo persistente para {symbol} | {timeframe} después de {max_retries} reintentos.")
    return False # Fallo después de todos los reintentos

# ---------------- Función Principal del Agente de Datos --------------------

def main():
    parser = argparse.ArgumentParser(description="Agente de Orquestación de Descarga de Datos de Mercado.")
    parser.add_argument("--mode", default="scheduled", choices=["scheduled", "on-demand"],
                        help="Modo de ejecución: 'scheduled' para todos los pares en config, 'on-demand' para un mensaje de Pub/Sub.")
    parser.add_argument("--message", type=str, help="Mensaje JSON de Pub/Sub para modo bajo demanda.")
    args = parser.parse_args()

    logger.info(f"[{datetime.utcnow().isoformat()} UTC] Iniciando Agente de Datos en modo: {args.mode}")

    # Cargar configuración desde GCS
    try:
        config = download_config_from_gcs(CONFIG_PATH)
        logger.info(f"Configuración cargada: {config}")
    except Exception as e:
        logger.critical(f"No se pudo cargar la configuración. Abortando: {e}")
        sys.exit(1)

    symbols_to_process = []
    timeframes_to_process = []
    callback_topic_id = None
    request_id = None

    # Lógica para procesar el mensaje de Pub/Sub si el modo es 'on-demand'
    if args.mode == "on-demand":
        if not args.message:
            logger.error("Modo 'on-demand' requiere un mensaje JSON de Pub/Sub (o simulación para prueba local).")
            sys.exit(1)
        try:
            # En un entorno real de Cloud Run, el mensaje entrante debe ser decodificado de base64.
            # El activador de Pub/Sub de Cloud Run pasa el mensaje real decodificado en el cuerpo.
            # Aquí, asumimos que args.message ya es un JSON string.
            message_data = json.loads(args.message) 

            symbols_to_process = message_data.get("symbols", [])
            timeframes_to_process = message_data.get("timeframes", [])
            callback_topic_id = message_data.get("callback_topic")
            request_id = message_data.get("request_id")
            if not symbols_to_process or not timeframes_to_process:
                logger.error("Mensaje 'on-demand' inválido: 'symbols' o 'timeframes' vacíos.")
                sys.exit(1)
            logger.info(f"Procesando solicitud bajo demanda para símbolos: {symbols_to_process}, timeframes: {timeframes_to_process}")
        except json.JSONDecodeError as e:
            logger.error(f"El mensaje de Pub/Sub no es un JSON válido: {e}")
            sys.exit(1)
    else: # scheduled mode
        symbols_to_process = config.get("symbols", [])
        timeframes_to_process = config.get("timeframes", [])
        if not symbols_to_process or not timeframes_to_process:
            logger.error("Configuración 'scheduled' inválida: 'symbols' o 'timeframes' vacíos.")
            sys.exit(1)
        logger.info(f"Procesando todos los símbolos/timeframes configurados: {symbols_to_process}, {timeframes_to_process}")


    all_succeeded = True
    processed_pairs = []
    failed_pairs = []

    # Preparar las variables de entorno que se pasarán al subproceso (data_fetcher_vertexai.py)
    # Esto es CRUCIAL para que el subproceso tenga las credenciales de GCS y la API Key.
    env_for_subprocess = os.environ.copy() # Copia todas las variables de entorno actuales
    # Asegúrate de que la API Key de Polygon esté en el entorno del subproceso
    if not env_for_subprocess.get("POLYGON_API_KEY"):
        logger.critical("Error: La variable de entorno POLYGON_API_KEY NO está configurada para el subproceso. La descarga fallará.")
        all_succeeded = False # Considerar un fallo global si la API key falta desde el inicio.
    # En el entorno local, asegura que GOOGLE_APPLICATION_CREDENTIALS se pase al subproceso
    # para la autenticación de GCS. En GCP, esto no es necesario ya que las credenciales
    # de la cuenta de servicio son automáticas.
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        env_for_subprocess["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    else:
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS no está configurada para la prueba local. Esto solo es un aviso en GCP.")


    for symbol in symbols_to_process:
        for timeframe in timeframes_to_process:
            # Si POLYGON_API_KEY no se encontró al inicio, no tiene sentido procesar.
            if not env_for_subprocess.get("POLYGON_API_KEY"):
                failed_pairs.append({"symbol": symbol, "timeframe": timeframe, "reason": "Missing API Key env var"})
                continue

            # Pasa las variables de entorno preparadas a process_data_fetch
            success = process_data_fetch(symbol, timeframe, config, env_vars_for_subprocess=env_for_subprocess)
            if success:
                processed_pairs.append(f"{symbol}_{timeframe}")
            else:
                all_succeeded = False
                failed_pairs.append({"symbol": symbol, "timeframe": timeframe, "reason": "Persistent fetch failure"})

    # Publicar notificaciones finales
    if all_succeeded:
        logger.info("🎉 Todas las descargas de datos completadas con éxito.")
        publish_message(SUCCESS_TOPIC_ID, {
            "status": "success",
            "message": "Datos de mercado actualizados con éxito.",
            "processed_pairs": processed_pairs,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id # Incluir request_id si es modo bajo demanda
        })
        # Si es una solicitud bajo demanda con callback, notificar allí también
        if callback_topic_id:
            publish_message(callback_topic_id, {
                "status": "data_ready",
                "message": "Datos solicitados listos para entrenamiento.",
                "symbols_timeframes": processed_pairs,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            })
    else:
        logger.error("❌ Algunas descargas de datos fallaron persistentemente.")
        publish_message(FAILURE_TOPIC_ID, {
            "status": "failure",
            "message": "Fallo en la actualización de datos de mercado.",
            "failed_pairs": failed_pairs,
            "processed_pairs": processed_pairs, # Mostrar los que sí se procesaron
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id # Incluir request_id si es modo bajo demanda
        })

    logger.info(f"[{datetime.utcnow().isoformat()} UTC] Agente de Datos finalizado.")

if __name__ == "__main__":
    main()