# tests/test_shared.py
"""
Pruebas unitarias para los módulos compartidos en `src/shared/`.

Este archivo contiene pruebas para:
- `indicators.py`: Verifica que los indicadores técnicos se calculen correctamente.
- `gcs_utils.py`: Verifica que las funciones de utilidad de GCS interactúen
  correctamente con el cliente de Google Cloud Storage (usando mocks).

No se prueban los archivos de constantes, ya que no contienen lógica ejecutable.
"""

from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np
import pytest

# Módulos del proyecto a probar
from src.shared import indicators, gcs_utils


# --- Fixtures de Pytest: Datos de prueba reutilizables ---

@pytest.fixture
def sample_ohlc_df() -> pd.DataFrame:
    """
    Crea un DataFrame de pandas con datos OHLC de ejemplo para usar en las pruebas.
    Contiene suficientes datos para que se puedan calcular todos los indicadores.
    """
    date_range = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D'))
    data = {
        'timestamp': date_range,
        'open': np.linspace(100, 150, 100),
        'high': np.linspace(102, 155, 100),
        'low': np.linspace(98, 148, 100),
        'close': np.linspace(101, 149, 100),
    }
    return pd.DataFrame(data)


# --- Pruebas para el módulo: indicators.py ---

class TestIndicators:
    """Agrupa todas las pruebas relacionadas con el cálculo de indicadores."""

    def test_build_indicators_adds_columns(self, sample_ohlc_df):
        """Verifica que la función añade las columnas de indicadores esperadas."""
        dummy_params = {
            "sma_len": 20, "rsi_len": 14, "macd_fast": 12,
            "macd_slow": 26, "stoch_len": 14
        }
        df_with_indicators = indicators.build_indicators(sample_ohlc_df, dummy_params)

        expected_cols = [
            "sma_20", "rsi_14", "macd_12_26", "stoch_k_14", "atr_14"
        ]
        for col in expected_cols:
            assert any(col in s for s in df_with_indicators.columns)

    def test_build_indicators_handles_na(self, sample_ohlc_df):
        """Verifica que, por defecto, la función elimina filas con NaNs."""
        dummy_params = {"sma_len": 20, "rsi_len": 14, "macd_fast": 12, "macd_slow": 26, "stoch_len": 14}
        
        # Con drop_na=True (default)
        df_dropped = indicators.build_indicators(sample_ohlc_df, dummy_params, drop_na=True)
        assert df_dropped.isna().sum().sum() == 0
        assert len(df_dropped) < len(sample_ohlc_df)

        # Con drop_na=False
        df_not_dropped = indicators.build_indicators(sample_ohlc_df, dummy_params, drop_na=False)
        assert df_not_dropped.isna().sum().sum() > 0
        assert len(df_not_dropped) == len(sample_ohlc_df)


# --- Pruebas para el módulo: gcs_utils.py ---

class TestGcsUtils:
    """
    Agrupa las pruebas para las utilidades de GCS.
    Utiliza `unittest.mock.patch` para simular el cliente de GCS.
    """

    @patch('src.shared.gcs_utils._get_gcs_client')
    def test_upload_gcs_file(self, mock_get_gcs_client, tmp_path):
        """Verifica que `upload_gcs_file` llama a los métodos correctos del cliente."""
        # 1. Preparar el entorno de la prueba
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_get_gcs_client.return_value.bucket.return_value = mock_bucket
        mock_get_gcs_client.return_value.bucket.return_value.blob.return_value = mock_blob
        
        local_file = tmp_path / "test.txt"
        local_file.write_text("hello world")
        gcs_uri = "gs://my-test-bucket/path/to/test.txt"

        # 2. Ejecutar la función a probar
        gcs_utils.upload_gcs_file(local_file, gcs_uri)

        # 3. Realizar aserciones
        mock_get_gcs_client.assert_called_once()
        mock_bucket.blob.assert_called_once_with("path/to/test.txt")
        mock_blob.upload_from_filename.assert_called_once_with(str(local_file))

    @patch('src.shared.gcs_utils._get_gcs_client')
    def test_download_gcs_file(self, mock_get_gcs_client, tmp_path):
        """Verifica que `download_gcs_file` llama a los métodos correctos del cliente."""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_get_gcs_client.return_value.bucket.return_value = mock_bucket
        mock_get_gcs_client.return_value.bucket.return_value.blob.return_value = mock_blob
        mock_blob.exists.return_value = True

        gcs_uri = "gs://my-test-bucket/path/to/download.txt"
        destination_dir = tmp_path

        # Ejecutar la función
        result_path = gcs_utils.download_gcs_file(gcs_uri, destination_dir)

        # Aserciones
        mock_get_gcs_client.assert_called_once()
        mock_bucket.blob.assert_called_once_with("path/to/download.txt")
        mock_blob.exists.assert_called_once()
        expected_local_path = destination_dir / "download.txt"
        mock_blob.download_to_filename.assert_called_once_with(expected_local_path)
        assert result_path == expected_local_path

    