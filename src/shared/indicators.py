#!/usr/bin/env python3
"""
indicators.py
────────────────────────────────────────────────────────────────────────────
Calcula indicadores técnicos SMA, RSI, MACD, Stochastic y ATR de forma
robusta.  Nunca devuelve None y puede garantizar que el DataFrame final
no contenga NaNs (útil antes de escalar o entrenar).

Uso rápido
----------
>>> from indicators import build_indicators
>>> df_ok = build_indicators(df_raw, params_dict)
"""

from __future__ import annotations

from typing import Dict, Tuple, Callable, List
import logging

import pandas as pd

try:
    import pandas_ta as ta
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pandas-ta no está instalado. Añádelo a requirements.txt → pandas-ta==0.3.14b0"
    ) from e

# ─────────────────────────────  logging  ──────────────────────────────
log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

# ─────────────────────────  Caché en memoria  ─────────────────────────
_INDICATOR_CACHE: Dict[Tuple[int, Tuple[int, ...]], pd.DataFrame] = {}


def _cache_key(
    df: pd.DataFrame, p: dict, atr_len: int, drop_na: bool, ffill: bool
) -> Tuple[int, Tuple[int, ...]]:
    """Clave basada en ``id`` del DataFrame, hiper-parámetros y opciones."""
    return (
        id(df),
        (
            p["sma_len"],
            p["rsi_len"],
            p["macd_fast"],
            p["macd_slow"],
            p["stoch_len"],
            atr_len,
            int(drop_na),
            int(ffill),
        ),
    )


# ─────────────────────  Helpers de cálculo seguro  ─────────────────────
def _nan_frame(index: pd.Index, cols: List[str]) -> pd.DataFrame:
    """DataFrame de NaNs (back-up cuando un cálculo falla)."""
    return pd.DataFrame({c: pd.NA for c in cols}, index=index)


def _safe_calc(
    name: str,
    func: Callable[[], pd.Series | pd.DataFrame | None],
    index: pd.Index,
    expected_cols: List[str],
) -> pd.DataFrame:
    """
    Ejecuta `func()` de forma segura; si falla o devuelve None se llenan NaNs.
    Además normaliza la salida a DataFrame con los nombres `expected_cols`.
    """
    try:
        res = func()
        if res is None:
            raise ValueError(f"{name} devolvió None")

        # Normalizar → DataFrame con columnas esperadas
        if isinstance(res, pd.Series):
            res = res.to_frame()

        if res.shape[1] != len(expected_cols):
            raise ValueError(
                f"{name}: nº de columnas inesperado "
                f"(obtenido {res.shape[1]}, esperado {len(expected_cols)})"
            )

        res = res.copy()
        res.columns = expected_cols
        return res

    except Exception as exc:
        log.warning("⚠️  %s falló (%s); se rellenan NaNs", name, exc)
        return _nan_frame(index, expected_cols)


# ─────────────────────────  Función principal  ─────────────────────────
def build_indicators(
    df: pd.DataFrame,
    params: dict,
    atr_len: int = 14,
    *,
    inplace: bool = False,
    drop_na: bool = True,
    ffill: bool = False,
) -> pd.DataFrame:
    """
    Calcula indicadores y los agrega al DataFrame (o a una copia).

    Parámetros
    ----------
    df : DataFrame con columnas open, high, low, close (minúsculas).
    params : dict con longitudes / hiper-parámetros.
    atr_len : periodo ATR.
    inplace : si True modifica `df` en sitio.
    drop_na : elimina filas con cualquier NaN al final.
    ffill    : aplica ffill además de bfill antes del drop_na.

    Devuelve
    --------
    DataFrame con los indicadores añadidos y, opcionalmente, sin NaNs.
    """
    # ─────────────────── Validación / renombre OHLC ────────────────────
    ohlc_map = {"o": "open", "h": "high", "l": "low", "c": "close"}
    missing = {"open", "high", "low", "close"} - set(df.columns)
    if missing and all(abbr in df.columns for abbr in ohlc_map):
        df = df.rename(columns=ohlc_map)
        missing = {"open", "high", "low", "close"} - set(df.columns)

    if missing:
        raise KeyError(
            f"Faltan columnas OHLC obligatorias: {', '.join(sorted(missing))}"
        )

    key = _cache_key(df, params, atr_len, drop_na, ffill)

    # ─────────────────────────  Cache  ────────────────────────────────
    if key in _INDICATOR_CACHE and _INDICATOR_CACHE[key] is not None:
        return _INDICATOR_CACHE[key]

    out = df if inplace else df.copy()

    # ───────────────────────  Indicadores  ────────────────────────────
    sma_len = params["sma_len"]
    rsi_len = params["rsi_len"]
    macd_fast, macd_slow = params["macd_fast"], params["macd_slow"]
    stoch_len = params["stoch_len"]

    # SMA
    sma_cols = [f"sma_{sma_len}"]
    out[sma_cols] = _safe_calc(
        "SMA",
        lambda: ta.sma(out.close, length=sma_len),
        out.index,
        sma_cols,
    )

    # RSI
    rsi_cols = [f"rsi_{rsi_len}"]
    out[rsi_cols] = _safe_calc(
        "RSI",
        lambda: ta.rsi(out.close, length=rsi_len),
        out.index,
        rsi_cols,
    )

    # MACD
    macd_cols = [
        f"macd_{macd_fast}_{macd_slow}",
        f"macd_signal_{macd_fast}_{macd_slow}",
        f"macd_hist_{macd_fast}_{macd_slow}",
    ]
    out[macd_cols] = _safe_calc(
        "MACD",
        lambda: ta.macd(out.close, fast=macd_fast, slow=macd_slow, signal=9),
        out.index,
        macd_cols,
    )

    # Stochastic
    stoch_cols = [f"stoch_k_{stoch_len}", f"stoch_d_{stoch_len}"]
    out[stoch_cols] = _safe_calc(
        "Stochastic",
        lambda: ta.stoch(out.high, out.low, out.close, k=stoch_len, d=3),
        out.index,
        stoch_cols,
    )

    # ATR
    atr_cols = [f"atr_{atr_len}"]
    out[atr_cols] = _safe_calc(
        "ATR",
        lambda: ta.atr(out.high, out.low, out.close, length=atr_len),
        out.index,
        atr_cols,
    )

    # ───────────────────  Post-procesado NaNs  ────────────────────────
    # Only fill missing values when explicitely requested.  Dropping NaNs first
    # ensures the initial warm‑up rows are removed, which keeps behaviour
    # predictable for tests.
    if drop_na:
        before = len(out)
        out = out.dropna().reset_index(drop=True)
        removed = before - len(out)
        if removed:
            log.info("🔍 build_indicators: %d filas con NaN eliminadas", removed)
    if ffill:
        out.ffill(inplace=True)

    # ─────────────────── Validación final & cache  ────────────────────
    if out.isna().any().any():
        log.warning("⚠️  Aún quedan NaNs tras el procesamiento.")

    _INDICATOR_CACHE[key] = out
    return out
