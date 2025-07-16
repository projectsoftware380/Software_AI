"""
Sistema de Análisis de Logs en Tiempo Real para Pipeline de Trading Forex
=======================================================================

Este módulo proporciona análisis automático de logs de GCP para identificar
mejoras en el código del pipeline de trading automatizado.

Componentes principales:
- RealtimeLogAnalyzer: Clase principal para streaming y análisis
- PatternDetector: Detección de patrones problemáticos
- SuggestionEngine: Generación de sugerencias de mejora
- AlertManager: Gestión de alertas por severidad
"""

from .analyzer import GCPLogStreamer, RealtimeLogAnalyzer
from .config import LogAnalyzerConfig

__version__ = "1.0.0"
__all__ = [
    "GCPLogStreamer",
    "LogAnalyzerConfig",
    "RealtimeLogAnalyzer"
] 