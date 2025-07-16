"""
Configuración del Sistema de Análisis de Logs
============================================

Maneja la configuración del analizador de logs incluyendo:
- Filtros de GCP Logging
- Patrones de detección
- Configuración de alertas
- Variables de entorno
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path
import logging

from ..shared.constants import PROJECT_ID


@dataclass
class LogAnalyzerConfig:
    """Configuración mínima para el streamer de logs"""
    project_id: str = field(default_factory=lambda: os.getenv("GCP_PROJECT_ID", ""))
    log_filters: Dict[str, str] = field(default_factory=lambda: {
        "resource_type": "k8s_container",
        "pod_app_label": "trading-pipeline",
        "min_severity": "INFO",
        "namespace": "trading"
    })

    @classmethod
    def from_yaml(cls, config_path: str) -> "LogAnalyzerConfig":
        if not Path(config_path).exists():
            return cls()
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

    def get_gcp_filter_string(self) -> str:
        filters = []
        resource_type = self.log_filters.get("resource_type")
        if resource_type and resource_type != "global":
            filters.append(f'resource.type="{resource_type}"')
        pod_app_label = self.log_filters.get("pod_app_label")
        if pod_app_label:
            filters.append(f'labels."k8s-pod/app"="{pod_app_label}"')
        min_severity = self.log_filters.get("min_severity")
        if min_severity:
            filters.append(f'severity>={min_severity}')
        namespace = self.log_filters.get("namespace")
        if namespace:
            filters.append(f'resource.labels.namespace_name="{namespace}"')
        if not filters:
            filters.append('severity>=INFO')
        return " AND ".join(filters)


def create_default_config(config_path: str = "log_analyzer_config.yaml") -> LogAnalyzerConfig:
    """Crea archivo de configuración por defecto"""
    config = LogAnalyzerConfig()
    config.to_yaml(config_path)
    logging.info(f"Archivo de configuración creado: {config_path}")
    return config 