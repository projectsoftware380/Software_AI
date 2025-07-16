# src/config/pipeline_config.py

from dataclasses import dataclass

@dataclass
class PipelineConfig:
    timeframe: str
    n_trials_arch: int
    n_trials_logic: int
    backtest_years_to_keep: int
    holdout_months: int
    end_date: str # Fecha de fin para la ingesta de datos
    # Añadir otros parámetros globales de la pipeline aquí
