# Dockerfile
# ---------
# Este archivo define la imagen Docker que se usará como entorno de ejecución
# para los componentes de la pipeline de KFP. Contiene todas las dependencias
# y el código fuente necesarios.

# 1. Imagen Base
# Se utiliza una imagen oficial de Python 3.10, versión "slim", que es ligera
# pero contiene las herramientas necesarias. Basada en Debian Bookworm.
FROM python:3.10-slim-bookworm

# 2. Variables de Entorno
# Se configuran variables de entorno para optimizar la ejecución en contenedores.
ENV PYTHONUNBUFFERED=1 \
    # Desactiva el caché de pip para mantener la imagen final más pequeña.
    PIP_NO_CACHE_DIR=on \
    # Suprime las advertencias sobre la versión de pip.
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # CRÍTICO: Añade el directorio /app al PYTHONPATH.
    # Esto permite que los scripts dentro del contenedor puedan hacer imports
    # absolutos desde la raíz del proyecto, como `from src.shared import constants`.
    PYTHONPATH=/app

# 3. Directorio de Trabajo
# Establece el directorio de trabajo por defecto dentro del contenedor.
WORKDIR /app

# 4. Instalación de Dependencias
# Se copia primero solo el archivo de requisitos para aprovechar el caché de capas de Docker.
# Docker solo reconstruirá esta capa si `requirements.txt` cambia.
COPY requirements.txt .

# Se actualiza pip y se instalan las dependencias.
# Se añade `--extra-index-url` para PyTorch, como se indica en los comentarios
# del archivo requirements.txt, para asegurar que se instalen las versiones de CPU.
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 5. Copia del Código Fuente
# Se copia todo el directorio `src` con el código modularizado al contenedor.
COPY src/ ./src/

# 6. Configuración del Entrypoint
# Se copia el script `run.sh`, que sirve como punto de entrada para el job
# de entrenamiento del LSTM en Vertex AI.
COPY run.sh .

# Se le dan permisos de ejecución al script.
RUN chmod +x run.sh

# Se establece `run.sh` como el punto de entrada del contenedor.
# Cuando Vertex AI inicie este contenedor para un Custom Job, ejecutará este
# script, pasándole los argumentos definidos en el componente `train_lstm_launcher`.
ENTRYPOINT ["./run.sh"]