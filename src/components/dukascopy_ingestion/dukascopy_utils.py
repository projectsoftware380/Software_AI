# src/components/dukascopy_ingestion/dukascopy_utils.py
"""
Utilidades para la descarga y procesamiento de datos de Dukascopy.
Adaptado de Descargar_Datos/src/dukascopy_downloader.py y Descargar_Datos/src/data_processor.py
"""

import logging
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
import dukascopy_python
from dukascopy_python.instruments import (
    INSTRUMENT_FX_MAJORS_EUR_USD, INSTRUMENT_FX_MAJORS_GBP_USD, INSTRUMENT_FX_MAJORS_USD_JPY,
    INSTRUMENT_FX_MAJORS_USD_CHF, INSTRUMENT_FX_MAJORS_AUD_USD, INSTRUMENT_FX_MAJORS_USD_CAD,
    INSTRUMENT_FX_MAJORS_NZD_USD
)

logger = logging.getLogger(__name__)

# Mapeo de símbolos Dukascopy-python (solo los disponibles)
SYMBOL_MAP = {
    'EURUSD': INSTRUMENT_FX_MAJORS_EUR_USD,
    'GBPUSD': INSTRUMENT_FX_MAJORS_GBP_USD,
    'USDJPY': INSTRUMENT_FX_MAJORS_USD_JPY,
    'USDCHF': INSTRUMENT_FX_MAJORS_USD_CHF,
    'AUDUSD': INSTRUMENT_FX_MAJORS_AUD_USD,
    'USDCAD': INSTRUMENT_FX_MAJORS_USD_CAD,
    'NZDUSD': INSTRUMENT_FX_MAJORS_NZD_USD
}

# Mapeo de timeframes
TIMEFRAME_MAP = {
    'm1': dukascopy_python.INTERVAL_MIN_1,
    'm5': dukascopy_python.INTERVAL_MIN_5,
    'm15': dukascopy_python.INTERVAL_MIN_15,
    'm30': dukascopy_python.INTERVAL_MIN_30,
    'h1': dukascopy_python.INTERVAL_HOUR_1,
    'h4': dukascopy_python.INTERVAL_HOUR_4,
    'd1': dukascopy_python.INTERVAL_DAY_1
}

OFFER_SIDE = dukascopy_python.OFFER_SIDE_BID

class DukascopyDownloader:
    """
    Clase para descargar datos históricos de Forex desde Dukascopy usando dukascopy-python.
    """
    def __init__(self):
        pass

    def get_available_symbols(self) -> List[str]:
        return list(SYMBOL_MAP.keys())

    def validate_symbol(self, symbol: str) -> bool:
        return symbol.upper() in SYMBOL_MAP

    def download_data(self, symbol: str, timeframe: str, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        try:
            if not self.validate_symbol(symbol):
                logger.error(f"Símbolo {symbol} no está disponible en Dukascopy")
                return []
            if timeframe not in TIMEFRAME_MAP:
                logger.error(f"Timeframe {timeframe} no es válido")
                return []
            logger.info(f"Iniciando descarga de {symbol} ({timeframe})")
            logger.info(f"Período: {start_date} a {end_date}")
            instrument = SYMBOL_MAP[symbol.upper()]
            interval = TIMEFRAME_MAP[timeframe]
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.max.time())
            df = dukascopy_python.fetch(
                instrument=instrument,
                interval=interval,
                offer_side=OFFER_SIDE,
                start=start_dt,
                end=end_dt
            )
            if df is None or df.empty:
                logger.warning("No se encontraron datos para el período especificado")
                return []
            # Convertir DataFrame a lista de dicts
            return df.reset_index().to_dict(orient='records')
        except Exception as e:
            logger.error(f"Error durante la descarga de Dukascopy: {e}", exc_info=True)
            return []

class DataProcessor:
    """
    Clase para procesar datos de Forex.
    """
    
    def process_raw_data(self, raw_data: List[Dict[str, Any]], symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Procesa los datos crudos de Dukascopy y los convierte a DataFrame.
        """
        if not raw_data:
            return pd.DataFrame()
        
        print(f"DEBUG: raw_data type: {type(raw_data)}, value: {raw_data}")
        df = pd.DataFrame(raw_data)
        
        column_mapping = {
            'timestamp': 'timestamp',
            'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
            'volume': 'volume'
        }
        
        # Renombrar columnas
        df = df.rename(columns={v: k for k, v in column_mapping.items() if v in df.columns})
        
        required_columns = ['timestamp', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Columnas faltantes en los datos procesados: {missing_columns}")
            return pd.DataFrame()
        
        if 'timestamp' in df.columns:
            if df['timestamp'].dtype == 'object':
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except Exception as e:
                    logger.error(f"Error al convertir timestamps: {e}", exc_info=True)
                    return pd.DataFrame()
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        
        return df
