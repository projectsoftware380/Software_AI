# -----------------------------------------------------------------------------
# Dockerfile Optimizado para Pipeline de MLOps con GPU
# -----------------------------------------------------------------------------

# Paso 1: Usar la imagen OFICIAL de TensorFlow para GPU como base.
FROM tensorflow/tensorflow:2.15.0-gpu

# Paso 2: Instalar dependencias del sistema operativo y Google Cloud SDK.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        libcairo2-dev \
        curl \
        gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

# AJUSTE CLAVE: Define el proyecto de GCP como variable de entorno.
# Esto soluciona el error "Project not passed" de forma definitiva.
ENV GOOGLE_CLOUD_PROJECT=trading-ai-460823

# Paso 3: Establecer el directorio de trabajo.
WORKDIR /app

# Paso 4: Copiar solo el archivo de requerimientos primero.
# Esta es la clave para el caché de capas - las dependencias solo se reinstalan
# cuando requirements.txt cambia, no cuando cambia el código.
COPY requirements.txt .

# Paso 5: Instalar PyTorch de forma aislada y robusta.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install \
        --no-cache-dir \
        --default-timeout=100 \
        --retries 10 \
        torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Paso 6: Instalar el resto de las dependencias de Python.
# Esta capa se cachea independientemente y solo se reconstruye si requirements.txt cambia.
RUN sed -i '/tensorflow/d' requirements.txt && \
    pip install \
        --no-cache-dir \
        --default-timeout=100 \
        --retries 10 \
        -r requirements.txt

# Paso 7: Copiar solo los archivos de configuración y utilidades compartidas.
# Estos archivos cambian menos frecuentemente que el código principal.
COPY pyproject.toml .
COPY src/shared/ ./src/shared/

# Paso 8: Copiar el resto del código del proyecto.
# Esta capa se reconstruye cada vez que el código cambia, pero las dependencias
# ya están instaladas y cacheadas en las capas anteriores.
COPY . .

# Paso 9: Definir un comando por defecto.
CMD ["/bin/bash"]