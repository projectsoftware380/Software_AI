

# src/log_analyzer/alerts.py

from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from .patterns import DetectedPattern
from .suggestions import ImprovementSuggestion

class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

class Alert:
    """
    Clase para representar una alerta generada.
    """
    def __init__(self, alert_id: str, title: str, description: str, severity: Severity,
                 timestamp: datetime, related_pattern: DetectedPattern, 
                 suggested_actions: List[ImprovementSuggestion]):
        self.id = alert_id
        self.title = title
        self.description = description
        self.severity = severity
        self.timestamp = timestamp
        self.related_pattern = related_pattern
        self.suggested_actions = suggested_actions
        self.status = "ACTIVE"

class AlertManager:
    """
    Clase para gestionar la creación y el estado de las alertas.
    """
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}

    def create_alert(self, pattern: DetectedPattern, 
                     suggestions: List[ImprovementSuggestion]) -> Alert:
        """
        Crea una nueva alerta basada en un patrón detectado y sugerencias.
        """
        alert_id = f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        title = f"Alerta: {pattern.pattern_type} detectado"
        description = f"Se ha detectado el patrón '{pattern.pattern_type}' con severidad {pattern.severity.value}. " \
                      f"Impacto: {pattern.impact_analysis}"
        
        alert = Alert(
            alert_id=alert_id,
            title=title,
            description=description,
            severity=pattern.severity,
            timestamp=datetime.now(),
            related_pattern=pattern,
            suggested_actions=suggestions
        )
        self.alerts[alert_id] = alert
        return alert

    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen de las alertas activas.
        """
        summary = {
            "total_alerts": len(self.alerts),
            "active_alerts": len([a for a in self.alerts.values() if a.status == "ACTIVE"]),
            "by_severity": {},
            "by_status": {}
        }
        for alert in self.alerts.values():
            summary["by_severity"][alert.severity.value] = summary["by_severity"].get(alert.severity.value, 0) + 1
            summary["by_status"][alert.status] = summary["by_status"].get(alert.status, 0) + 1
        return summary

    def resolve_alert(self, alert_id: str) -> Optional[Alert]:
        """
        Marca una alerta como resuelta.
        """
        alert = self.alerts.get(alert_id)
        if alert:
            alert.status = "RESOLVED"
        return alert

