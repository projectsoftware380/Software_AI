# tests/test_components.py
"""
Pruebas unitarias y de integración para los componentes de la pipeline.

Este archivo utiliza pytest y unittest.mock para probar la lógica de cada
script `task.py` de forma aislada. Las dependencias externas (como las
llamadas a APIs de GCP o a servicios de datos) se "mockean" para que las
pruebas sean rápidas, determinísticas y no dependan de credenciales o
recursos en la nube.
"""

import sys
import subprocess
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

# --- Constantes para Pruebas ---
TEST_PAIR = "EURUSD"
TEST_TIMEFRAME = "15minute"
TEST_PROJECT_ID = "test-project"
TEST_GCS_BUCKET = "test-bucket"

from src.components.dukascopy_ingestion.task import run_ingestion

# --- Pruebas para el Componente: dukascopy_ingestion ---

@patch('src.components.dukascopy_ingestion.task.upload_gcs_file')
@patch('src.components.dukascopy_ingestion.task.DukascopyDownloader')
def test_dukascopy_ingestion_task_success(mock_downloader, mock_upload, tmp_path):
    """
    Prueba el flujo de éxito del script de ingestión de datos de Dukascopy.
    Verifica que se llame a la función de subida a GCS con los argumentos correctos.
    """
    # 1. Configurar el mock del descargador de Dukascopy
    dummy_data = pd.DataFrame({
        'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=10, freq='h')),
        'open': np.random.rand(10), 'high': np.random.rand(10),
        'low': np.random.rand(10), 'close': np.random.rand(10)
    })
    mock_downloader.return_value.download_data.return_value = [
        {'timestamp': '2023-01-01 00:00:00', 'open': 1.0, 'high': 1.1, 'low': 0.9, 'close': 1.05, 'volume': 100},
        {'timestamp': '2023-01-01 01:00:00', 'open': 1.05, 'high': 1.15, 'low': 0.95, 'close': 1.10, 'volume': 120},
    ]

    # 2. Ejecutar la función directamente
    success = run_ingestion(
        pair=TEST_PAIR,
        timeframe="m1",
        end_date_str="2023-01-10",
        project_id=TEST_PROJECT_ID,
        output_gcs_path=f"gs://{TEST_GCS_BUCKET}/data",
        local_output_dir=str(tmp_path)
    )

    # 3. Realizar aserciones (verificaciones)
    assert success is True
    mock_upload.assert_called_once()
    final_gcs_path = mock_upload.call_args[0][1]
    assert f"gs://{TEST_GCS_BUCKET}/data/{TEST_PAIR}/m1/dukascopy_{TEST_PAIR}_m1.parquet" in final_gcs_path




from src.components.data_preparation.task import run_data_preparation

# --- Pruebas para el Componente: data_preparation ---

@patch('src.components.data_preparation.task.save_df_to_gcs_and_verify')
@patch('src.components.data_preparation.task.pd.read_parquet')
def test_data_preparation_task(mock_read_parquet, mock_save_df, tmp_path):
    """
    Prueba el flujo de éxito del script de preparación de datos.
    Simula la descarga de un parquet y verifica que se suba un resultado.
    """
    # 1. Crear datos de prueba y guardarlos en un archivo parquet local temporal
    dummy_data = pd.DataFrame({
        'timestamp': pd.to_datetime(pd.date_range(start='2020-01-01', periods=100, freq='h')),
        'open': np.random.rand(100), 'high': np.random.rand(100),
        'low': np.random.rand(100), 'close': np.random.rand(100),
        'volume': np.random.rand(100)
    })
    mock_read_parquet.return_value = dummy_data

    # 2. Ejecutar la función directamente
    run_data_preparation(
        years_to_keep=1,
        holdout_months=6,
        input_data_path=f"gs://{TEST_GCS_BUCKET}/data/{TEST_PAIR}/{TEST_TIMEFRAME}/input.parquet",
        output_gcs_path=f"gs://{TEST_GCS_BUCKET}/prepared_data",
    )

    # 3. Aserciones
    mock_read_parquet.assert_called_once()
    assert mock_save_df.call_count == 2



# --- Pruebas para Componentes de ML (ej. model_promotion) ---

from src.components.model_promotion.task import run_model_promotion

# --- Pruebas para Componentes de ML (ej. model_promotion) ---

@patch('src.components.model_promotion.task.gcs_utils')
@patch('src.components.model_promotion.task.json.load')
def test_model_promotion_task_promotes(mock_gcs_utils, mock_json_load, tmp_path):
    """Prueba el caso en que el modelo SÍ es promovido."""
    # 1. Configurar mocks para simular métricas excelentes
    new_metrics_content = { "sharpe_ratio": 2.5 }
    prod_metrics_content = { "sharpe_ratio": 1.0 }

    mock_json_load.side_effect = [new_metrics_content, prod_metrics_content]

    

    def mock_download_side_effect(gcs_uri, local_dir):
        target_file_path = local_dir / Path(gcs_uri).name
        if "new_metrics" in gcs_uri:
            target_file_path.write_text(json.dumps(new_metrics_content))
        elif "metrics.json" in gcs_uri: # Para las métricas de producción
            target_file_path.write_text(json.dumps(prod_metrics_content))
        return str(target_file_path)

    mock_gcs_utils.download_gcs_file.side_effect = mock_download_side_effect
    mock_gcs_utils.gcs_path_exists.return_value = True # Simular que el modelo en producción existe

    # 2. Ejecutar la función directamente
    run_model_promotion(
        new_metrics_dir="gs://fake/metrics/dir",
        new_lstm_artifacts_dir="gs://fake/lstm/dir",
        new_filter_model_path="gs://fake/filter/model.pkl",
        pair=TEST_PAIR,
        timeframe=TEST_TIMEFRAME,
        production_base_dir="gs://fake/production"
    )
    
    # 3. Aserciones
    # Se debe haber llamado a la copia de artefactos
    assert mock_gcs_utils.copy_gcs_directory.call_count > 0
    assert mock_gcs_utils.copy_gcs_blob.call_count > 0
