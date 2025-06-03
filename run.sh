#!/bin/bash
set -e

echo "📥 Descargando código desde GCS..."
gsutil -m cp gs://trading-ai-models-460823/code/*.py ./

echo "🚀 Ejecutando script principal..."
python train_lstm.py