# scripts/create_gcs_bucket.py
"""
Crea un bucket en Google Cloud Storage usando la biblioteca de Python.
"""

import argparse
import logging
from google.cloud import storage
from google.api_core import exceptions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_bucket(bucket_name: str, project_id: str, location: str):
    """Crea un nuevo bucket de GCS."""
    storage_client = storage.Client(project=project_id)

    try:
        logging.info(f"Intentando crear el bucket '{bucket_name}' en el proyecto '{project_id}' y región '{location}'...")
        bucket = storage_client.create_bucket(bucket_name, location=location)
        logging.info(f"✅ Bucket '{bucket.name}' creado exitosamente.")
    except exceptions.Conflict:
        logging.warning(f"⚠️ El bucket '{bucket_name}' ya existe. No se requiere ninguna acción.")
    except Exception as e:
        logging.error(f"❌ No se pudo crear el bucket '{bucket_name}'. Error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crear un bucket de GCS.")
    parser.add_argument("--project-id", required=True, help="ID del proyecto de Google Cloud.")
    parser.add_argument("--bucket-name", required=True, help="Nombre del bucket a crear.")
    parser.add_argument("--location", default="europe-west1", help="Ubicación del bucket.")
    args = parser.parse_args()

    create_bucket(args.bucket_name, args.project_id, args.location)
