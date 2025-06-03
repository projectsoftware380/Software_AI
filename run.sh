#!/bin/bash

# Descargar el código desde GCS
echo "Descargando scripts desde GCS..."
gsutil -m cp gs://trading-ai-models-460823/code/*.py . || exit 1

# Ejecutar el script principal (ya descargado)
echo "Ejecutando el script..."
python train_lstm.py
