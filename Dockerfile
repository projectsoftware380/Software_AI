# Dockerfile
# -----------
# Define la imagen Docker para el entorno de ejecución de los componentes de la pipeline en Vertex AI.
# Contiene el sistema operativo, las dependencias de Python y el código fuente de la aplicación.

# 1. IMAGEN BASE
# Se parte de una imagen oficial y ligera de Python 3.10, basada en Debian (Bookworm).
# Es una base estable y común para aplicaciones de Python.
FROM python:3.10-slim-bookworm

# 2. VARIABLES DE ENTORNO
# Se configuran para optimizar la ejecución de Python y pip en un entorno de contenedor.
ENV PYTHONUNBUFFERED=1 \
    # Deshabilita el caché de pip para reducir el tamaño final de la imagen.
    PIP_NO_CACHE_DIR=on \
    # Suprime las advertencias sobre la versión de pip durante la instalación.
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # CRÍTICO: Añade el directorio de trabajo '/app' al PYTHONPATH.
    # Esto permite que los módulos de Python (ej: 'from src.shared import ...')
    # se importen de forma absoluta desde cualquier script dentro del contenedor.
    PYTHONPATH=/app

# 3. DIRECTORIO DE TRABAJO
# Establece '/app' como el directorio por defecto para todos los comandos subsiguientes (COPY, RUN, etc.).
WORKDIR /app

# 4. INSTALACIÓN DE DEPENDENCIAS DE PYTHON
# Se copia únicamente el archivo de requisitos para aprovechar el sistema de caché de capas de Docker.
# Esta capa solo se reconstruirá si el contenido de 'requirements.txt' cambia.
COPY requirements.txt .

# Se actualiza pip y se instalan las dependencias listadas.
# La opción '--no-cache-dir' es redundante por la variable de entorno, pero es una buena práctica mantenerla.
# Se añade el índice extra para PyTorch (versión CPU), como se especifica en 'requirements.txt'.
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 5. COPIA DEL CÓDIGO FUENTE
# Se copia el código fuente del proyecto al directorio de trabajo del contenedor.
# Esto sirve como una versión 'base' o 'fallback' del código.
# La ruta de destino './src/' es relativa al WORKDIR, resultando en '/app/src/'.
COPY src/ ./src/

# 6. CONFIGURACIÓN DEL PUNTO DE ENTRADA (ENTRYPOINT)
# Se copia el script 'run.sh', que actuará como el punto de entrada principal del contenedor.
COPY run.sh .

# Se otorgan permisos de ejecución al script 'run.sh' para que el sistema pueda invocarlo.
RUN chmod +x run.sh

# Se establece 'run.sh' como el ENTRYPOINT. Cuando Vertex AI inicia un job con este contenedor,
# ejecutará este script. El script está diseñado para recibir comandos como argumentos
# (gracias a 'exec "$@"'), como la llamada al módulo de entrenamiento del LSTM.
ENTRYPOINT ["./run.sh"]