# src/shared/logging_config.py
import logging
import os
# from google.cloud.logging.handlers import CloudLoggingHandler # Descomentar si se usa GCP
# from google.cloud.logging import Client as CloudLoggingClient # Descomentar si se usa GCP

def setup_logging(level=logging.INFO, enable_cloud_logging=False):
    """
    Configura el sistema de logging para el proyecto.
    Puede enviar logs a la consola y opcionalmente a Google Cloud Logging.
    """
    # Evitar configurar el logger raíz múltiples veces
    if logging.getLogger().handlers:
        return

    # Configuración básica para la consola
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()]
    )
    
    # Configurar el logger raíz
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Opcional: Integración con Google Cloud Logging
    # Descomentar y configurar si se desea enviar logs a GCP
    # if enable_cloud_logging and os.getenv("GOOGLE_CLOUD_PROJECT"):
    #     try:
    #         client = CloudLoggingClient()
    #         handler = CloudLoggingHandler(client)
    #         # Usar un formato JSON para Cloud Logging para mejor parseo
    #         formatter = logging.Formatter(
    #             '{"timestamp": "%(asctime)s", "severity": "%(levelname)s", "name": "%(name)s", "message": "%(message)s", "trace": "%(pathname)s:%(lineno)d"}'
    #         )
    #         handler.setFormatter(formatter)
    #         root_logger.addHandler(handler)
    #         logging.info("Logging configurado para enviar a Google Cloud Logging.")
    #     except Exception as e:
    #         logging.warning(f"No se pudo configurar Google Cloud Logging: {e}")
