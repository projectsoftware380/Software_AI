
import os
import subprocess
from datetime import datetime

# --- Configuración ---
PROJECT_ID = "trading-ai-460823"
REGION = "europe-west1"
REPO_NAME = "data-ingestion-repo"
IMAGE_NAME = "data-ingestion-agent"

# --- 1. Generar Tag y URI de la Imagen ---
print("--- Paso 1: Generando etiqueta de versión y URI de la imagen ---")
version_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
common_image_uri = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO_NAME}/{IMAGE_NAME}:{version_tag}"
print(f"URI de la imagen: {common_image_uri}")

# --- 2. Construir y Subir la Imagen Docker ---
print("\n--- Paso 2: Construyendo y subiendo la imagen Docker ---")
try:
    # Construir la imagen
    build_command = ["docker", "build", "--no-cache", "-t", common_image_uri, "."]
    print(f"Ejecutando: {' '.join(build_command)}")
    subprocess.run(build_command, check=True)

    # Subir la imagen
    push_command = ["docker", "push", common_image_uri]
    print(f"Ejecutando: {' '.join(push_command)}")
    subprocess.run(push_command, check=True)
    print("Imagen Docker subida correctamente.")
except subprocess.CalledProcessError as e:
    print(f"Error durante el proceso de Docker: {e}")
    exit(1)
except FileNotFoundError:
    print("Error: 'docker' no se encuentra. Asegúrate de que Docker esté instalado y en el PATH del sistema.")
    exit(1)


# --- 3. Compilar y Desplegar la Pipeline ---
print("\n--- Paso 3: Compilando y desplegando la pipeline ---")
try:
    deploy_command = [
        "python", "-m", "src.deploy.cli",
        "--common-image-uri", common_image_uri,
        "--output-json", "algo_trading_mlops_pipeline_v5_final.json",
        "--timeframe", "1h",
        "--n-trials-arch", "20",
        "--n-trials-logic", "30",
        "--backtest-years-to-keep", "5",
        "--holdout-months", "3"
    ]
    print(f"Ejecutando: {' '.join(deploy_command)}")
    subprocess.run(deploy_command, check=True)
    print("Pipeline desplegado correctamente.")
except subprocess.CalledProcessError as e:
    print(f"Error durante el despliegue de la pipeline: {e}")
    exit(1)
except FileNotFoundError:
    print("Error: 'python' no se encuentra. Asegúrate de que Python esté instalado y en el PATH del sistema.")
    exit(1)

print("\nProceso de despliegue completado.")
