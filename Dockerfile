# -----------------------------------------------------------------------------
# Dockerfile para pipeline MLOps con soporte GPU
# -----------------------------------------------------------------------------

FROM tensorflow/tensorflow:2.15.0-gpu

# Dependencias del sistema y Google Cloud CLI.
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

# La identidad/proyecto GCP se proporcionan en runtime mediante variables de
# entorno o credenciales de workload; no se vinculan a una cuenta concreta.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copiar dependencias primero para aprovechar el cache de Docker.
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install \
        --no-cache-dir \
        --default-timeout=100 \
        --retries 10 \
        torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

RUN sed -i '/tensorflow/d' requirements.txt && \
    pip install \
        --no-cache-dir \
        --default-timeout=100 \
        --retries 10 \
        -r requirements.txt

COPY pyproject.toml .
COPY src/shared/ ./src/shared/
COPY . .

CMD ["/bin/bash"]
