import logging
from python_json_logger import jsonlogger

def setup_logging():
    """
    Configura el logging para que emita en formato JSON,
    compatible con Cloud Logging.
    """
    logger = logging.getLogger()
    
    # Eliminar handlers existentes para evitar duplicados
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logHandler = logging.StreamHandler()
    
    # El formato especial de GCP para que reconozca los campos.
    # https://cloud.google.com/logging/docs/agent/logging/configuration#special-fields
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s %(lineno)d '
        '%(pathname)s',
        rename_fields={
            "levelname": "severity", # GCP usa 'severity' en lugar de 'levelname'
            "asctime": "timestamp"
        }
    )
    
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)
    logger.setLevel(logging.INFO)