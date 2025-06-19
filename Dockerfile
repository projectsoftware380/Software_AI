# -----------------------------------------------------------------------------
# Dockerfile Optimizado para Pipeline de MLOps con GPU
# -----------------------------------------------------------------------------

# Paso 1: Usar la imagen OFICIAL de TensorFlow para GPU como base.
# Esta imagen es el estándar de la industria, mantenida por el equipo de
# TensorFlow. Contiene Python 3.10, TensorFlow 2.15.0 y, lo más importante,
# todas las librerías CUDA (v12.2) y cuDNN necesarias pre-configuradas.
FROM tensorflow/tensorflow:2.15.0-gpu

# Paso 2: Instalar dependencias del sistema operativo.
# 'build-essential' y 'pkg-config' son necesarios para compilar algunas
# librerías de Python. 'libcairo2-dev' es dependencia de librerías de gráficos.
# Se limpian los cachés de apt para mantener la imagen ligera.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        libcairo2-dev && \
    rm -rf /var/lib/apt/lists/*

# Paso 3: Establecer el directorio de trabajo.
# Todas las operaciones posteriores se realizarán dentro de /app.
WORKDIR /app

# Paso 4: Copiar solo el archivo de requerimientos primero.
# Esto optimiza el uso del caché de Docker. Este paso solo se re-ejecutará
# si el archivo requirements.txt cambia.
COPY requirements.txt .

# Paso 5: Instalar PyTorch de forma aislada y robusta.
# Se instala PyTorch por separado para asegurar que se usa la versión
# compatible con CUDA 12.1, que es compatible con la CUDA 12.2 de la imagen
# base. Apuntar al index-url de PyTorch es más directo y fiable.
# Se incluyen reintentos y un timeout más largo para evitar fallos por
# problemas de red, como el que ocurrió con 'triton'.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install \
        --no-cache-dir \
        --default-timeout=100 \
        --retries 10 \
        torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Paso 6: Instalar el resto de las dependencias de Python.
# 1. Se elimina 'tensorflow' del requirements.txt con 'sed' para no
#    sobreescribir la versión optimizada de la imagen base.
# 2. Se instalan las librerías restantes, de nuevo con reintentos y timeout
#    para máxima robustez en la descarga.
RUN sed -i '/tensorflow/d' requirements.txt && \
    pip install \
        --no-cache-dir \
        --default-timeout=100 \
        --retries 10 \
        -r requirements.txt

# Paso 7: Copiar todo el código del proyecto al contenedor.
# Este paso se ejecuta después de instalar todas las dependencias para que,
# si solo cambias el código de tu aplicación, la construcción sea casi
# instantánea gracias al caché de Docker.
COPY . .

# Paso 8: Definir un comando por defecto.
# Este comando es un placeholder y será ignorado y sobreescrito por el
# orquestador (Kubeflow/Vertex AI) al ejecutar un componente del pipeline.
CMD ["/bin/bash"]
