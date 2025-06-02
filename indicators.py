#!/usr/bin/env python3
"""
Añade indicadores técnicos a un DataFrame OHLC de forma **robusta**:
SMA, RSI, MACD, Stochastic y ATR.  Nunca devuelve `None` y la caché
solo guarda objetos `pd.DataFrame` válidos.

Uso:
    from core.indicators import build_indicators
    df_ind = build_indicators(df_raw, params, atr_len=14)
"""

from __future__ import annotations
from typing import Dict, Tuple, Callable, List
import pandas as pd
import pandas_ta as ta

# ─────────────────────────  Caché en memoria  ──────────────────────────
_INDICATOR_CACHE: Dict[Tuple[int, Tuple[int, ...]], pd.DataFrame] = {}


def _cache_key(df: pd.DataFrame, p: dict, atr_len: int) -> Tuple[int, Tuple[int, ...]]:
    """Clave: id del DataFrame + hiperparámetros relevantes."""
    return (
        id(df),
        (p["sma_len"], p["rsi_len"], p["macd_fast"], p["macd_slow"], p["stoch_len"], atr_len),
    )


# ─────────────────────  Helpers de cálculo seguro  ─────────────────────
def _nan_frame(index: pd.Index, columns: List[str]) -> pd.DataFrame:
    """DataFrame lleno de NA para usar como relleno de emergencia."""
    return pd.DataFrame({c: pd.Series([pd.NA] * len(index), index=index) for c in columns})


def _safe_calc(
    name: str,
    func: Callable[[], pd.Series | pd.DataFrame | None],
    index: pd.Index,
    expected_cols: List[str],
) -> pd.DataFrame:
    """
    Intenta ejecutar `func()`.  
    Si hay error o devuelve `None`, entrega DataFrame de NaNs con `expected_cols`.
    Asegura que el resultado final sea DataFrame con las columnas correctas.
    """
    try:
        res = func()
        if res is None:
            raise ValueError(f"{name} devolvió None")
        # Normalizar a DataFrame con columnas deseadas
        if isinstance(res, pd.Series):
            res = res.to_frame(name=expected_cols[0])
        res = res.copy()
        res.columns = expected_cols
        return res
    except Exception as e:
        print(f"⚠️  {name} falló ({e}); columnas rellenadas con NaN")
        return _nan_frame(index, expected_cols)


# ─────────────────────────  Función principal  ─────────────────────────
def build_indicators(
    df: pd.DataFrame,
    params: dict,
    atr_len: int = 14,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Calcula indicadores y los agrega al DataFrame (o a una copia).
    Garantiza:
      • Nunca retorna `None`.
      • Si un indicador individual falla, lo reemplaza por NaNs y sigue.
    """
    key = _cache_key(df, params, atr_len)

    # -------- Cache ----------
    if key in _INDICATOR_CACHE:
        cached = _INDICATOR_CACHE[key]
        if cached is not None:
            return cached
        print(f"⚠️  Entrada corrupta (None) en caché para clave {key}. Recalculando…")
        del _INDICATOR_CACHE[key]

    out = df if inplace else df.copy()

    # -------- Indicadores ----------
    # SMA
    sma_cols = [f"sma_{params['sma_len']}"]
    out[sma_cols] = _safe_calc(
        "SMA",
        lambda: ta.sma(out.close, length=params["sma_len"]),
        out.index,
        sma_cols,
    )

    # RSI
    rsi_cols = [f"rsi_{params['rsi_len']}"]
    out[rsi_cols] = _safe_calc(
        "RSI",
        lambda: ta.rsi(out.close, length=params["rsi_len"]),
        out.index,
        rsi_cols,
    )

    # MACD (devuelve 3 columnas)
    macd_cols = [
        f"macd_{params['macd_fast']}_{params['macd_slow']}",
        f"macd_signal_{params['macd_fast']}_{params['macd_slow']}",
        f"macd_hist_{params['macd_fast']}_{params['macd_slow']}",
    ]
    out[macd_cols] = _safe_calc(
        "MACD",
        lambda: ta.macd(
            out.close,
            fast=params["macd_fast"],
            slow=params["macd_slow"],
            signal=9,
        ),
        out.index,
        macd_cols,
    )

    # Stochastic (2 columnas)
    stoch_cols = [f"stoch_k_{params['stoch_len']}", f"stoch_d_{params['stoch_len']}"]
    out[stoch_cols] = _safe_calc(
        "Stochastic",
        lambda: ta.stoch(
            out.high,
            out.low,
            out.close,
            k=params["stoch_len"],
            d=3,
        ),
        out.index,
        stoch_cols,
    )

    # ATR (1 columna)
    atr_cols = [f"atr_{atr_len}"]
    out[atr_cols] = _safe_calc(
        "ATR",
        lambda: ta.atr(out.high, out.low, out.close, length=atr_len),
        out.index,
        atr_cols,
    )

    # -------- Post-procesado ----------
    out.bfill(inplace=True)

    # -------- Validación final ----------
    if not isinstance(out, pd.DataFrame):
        raise ValueError("build_indicators devolvió un objeto no-DataFrame")

    # Guardar en caché y retornar
    _INDICATOR_CACHE[key] = out
    return out
