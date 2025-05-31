#!/usr/bin/env python3
"""

────────────────
Utilidad centralizada para añadir indicadores técnicos a un
DataFrame OHLC.  Pensada para ser importada desde cualquier
script de entrenamiento o back-test.

Uso:
    from core.indicators import build_indicators
    df_ind = build_indicators(df_raw, params)          # ← df_raw no se modifica
"""

from __future__ import annotations
import pandas as pd
import pandas_ta as ta
from typing import Dict, Tuple

# cache en memoria para evitar recomputar los mismos indicadores
_INDICATOR_CACHE: Dict[Tuple[int, Tuple[int, ...]], pd.DataFrame] = {}

def _cache_key(df: pd.DataFrame, params: dict, atr_len: int) -> Tuple[int, Tuple[int, ...]]:
    """Clave: id del DF original + valores de hiperparámetros + atr_len."""
    return (
        id(df),                               # referencia única del objeto
        (
            params["sma_len"],
            params["rsi_len"],
            params["macd_fast"],
            params["macd_slow"],
            params["stoch_len"],
            atr_len,
        ),
    )

def build_indicators(
    df: pd.DataFrame,
    params: dict,
    atr_len: int = 14,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Calcula SMA, RSI, MACD, Stochastic y ATR según los hiperparámetros
    entregados en `params` y devuelve un nuevo DataFrame con las columnas
    añadidas.  Cachea resultados por sesión para acelerar llamadas repetidas.

    Parameters
    ----------
    df : pd.DataFrame
        Debe contener columnas: open, high, low, close.
    params : dict
        Debe incluir:
            sma_len, rsi_len, macd_fast, macd_slow, stoch_len
    atr_len : int, default 14
        Periodo de ATR a calcular.
    inplace : bool, default False
        Si True, añade columnas sobre el mismo DataFrame;
        si False, devuelve una copia.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas de indicadores añadidas.
    """
    key = _cache_key(df, params, atr_len)
    if key in _INDICATOR_CACHE:
        return _INDICATOR_CACHE[key]

    out = df if inplace else df.copy()

    # SMA y RSI
    out[f"sma_{params['sma_len']}"] = ta.sma(out.close, length=params["sma_len"])
    out[f"rsi_{params['rsi_len']}"] = ta.rsi(out.close, length=params["rsi_len"])

    # MACD
    macd = ta.macd(
        out.close,
        fast=params["macd_fast"],
        slow=params["macd_slow"],
        signal=9,
    )
    out[
        [
            f"macd_{params['macd_fast']}_{params['macd_slow']}",
            f"macd_signal_{params['macd_fast']}_{params['macd_slow']}",
            f"macd_hist_{params['macd_fast']}_{params['macd_slow']}",
        ]
    ] = macd

    # Stochastic
    stoch = ta.stoch(
        out.high,
        out.low,
        out.close,
        k=params["stoch_len"],
        d=3,
    )
    out[[f"stoch_k_{params['stoch_len']}", f"stoch_d_{params['stoch_len']}"]] = stoch

    # ATR
    out[f"atr_{atr_len}"] = ta.atr(out.high, out.low, out.close, length=atr_len)

    # Rellena NaN iniciales
    out.bfill(inplace=True)

    _INDICATOR_CACHE[key] = out
    return out
