# Imagen base: TF 2.16 + CUDA 12 + Python 3.11
FROM tensorflow/tensorflow:2.16.1-gpu

ENV DEBIAN_FRONTEND=noninteractive

# ── Paquetes de sistema ──────────────────────────────────────────────
# build-essential      → gcc / g++ / make (necesario para compilar C)
# pkg-config           → localiza las cflags/libs de cairo
# libcairo2-dev        → cabeceras + pkg-config *.pc
# libcairo2            → runtime (ya venía, pero lo incluimos por claridad)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        libcairo2 \
        libcairo2-dev && \
    rm -rf /var/lib/apt/lists/*

# ── Carpeta de trabajo ───────────────────────────────────────────────
WORKDIR /app

# ── Dependencias Python ──────────────────────────────────────────────
COPY requirements.txt .

RUN sed -i '/^tensorflow/d' requirements.txt && \
    python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip check                       # Falla rápido si algo está roto

# ── Copia del código (ajusta a tu estructura) ────────────────────────
COPY . .

CMD ["python", "main.py"]
