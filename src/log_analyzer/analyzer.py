"""
Analizador de Logs en Tiempo Real para Pipeline de Trading
=========================================================

Clase principal que coordina:
- Streaming de logs desde GCP
- Detección de patrones problemáticos
- Generación de sugerencias de mejora
- Gestión de alertas
"""

import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor
import threading

from google.cloud.logging_v2.services.logging_service_v2 import LoggingServiceV2Client
from google.cloud.logging_v2.types import TailLogEntriesRequest
from google.api_core import exceptions as google_exceptions

from .config import LogAnalyzerConfig
from .patterns import PatternDetector, DetectedPattern
from .suggestions import SuggestionEngine, ImprovementSuggestion
from .alerts import AlertManager, Alert, Severity


class GCPLogStreamer:
    """Clase simplificada para obtener logs en tiempo real desde GCP"""
    def __init__(self, config: LogAnalyzerConfig):
        self.config = config
        self.running = False
        self.logging_client = LoggingServiceV2Client()

    def stream_logs(self):
        """Generador que produce logs en tiempo real desde GCP"""
        filter_string = self.config.get_gcp_filter_string()
        last_timestamp = None
        self.running = True
        while self.running:
            current_filter = filter_string
            if last_timestamp:
                current_filter += f' AND timestamp>"{last_timestamp}"'
            try:
                request = {
                    "resource_names": [f"projects/{self.config.project_id}"],
                    "filter": current_filter,
                    "order_by": "timestamp desc",
                    "page_size": 10
                }
                page_result = self.logging_client.list_log_entries(request)
                entries = list(page_result)
                for entry in reversed(entries):
                    log_dict = self._log_entry_to_dict(entry)
                    yield log_dict
                    if entry.timestamp:
                        last_timestamp = entry.timestamp.ToDatetime().isoformat()
                time.sleep(2)
            except google_exceptions.PermissionDenied:
                logging.exception("Permisos insuficientes para acceder a logs de GCP")
                break
            except google_exceptions.NotFound:
                logging.exception("Proyecto o recurso no encontrado")
                break
            except Exception as e:
                logging.exception(f"Error obteniendo logs: {e}")
                time.sleep(5)

    def stop(self):
        self.running = False

    def _log_entry_to_dict(self, log_entry):
        # Incluye detalles adicionales del log
        log_dict = {
            "timestamp": log_entry.timestamp.ToDatetime().isoformat() if log_entry.timestamp else None,
            "severity": getattr(log_entry.severity, 'name', None),
            "message": getattr(log_entry, 'text_payload', None) or str(getattr(log_entry, 'json_payload', '')),
            "resource": str(getattr(log_entry, 'resource', None)),
            "labels": dict(getattr(log_entry, 'labels', {})),
        }
        # Incluye cualquier otro campo relevante
        if hasattr(log_entry, 'insert_id'):
            log_dict["insert_id"] = getattr(log_entry, 'insert_id', None)
        if hasattr(log_entry, 'log_name'):
            log_dict["log_name"] = getattr(log_entry, 'log_name', None)
        return log_dict


class RealtimeLogAnalyzer:
    """Analizador principal de logs en tiempo real"""
    
    def __init__(self, config: LogAnalyzerConfig):
        self.config = config
        self.running = False
        self.paused = False
        
        # Componentes del sistema
        self.pattern_detector = PatternDetector(config)
        self.suggestion_engine = SuggestionEngine()
        self.alert_manager = AlertManager()
        
        # Cliente de GCP Logging (streaming real)
        self.logging_client = None
        self._setup_gcp_client()
        
        # Estadísticas
        self.stats = {
            "logs_processed": 0,
            "patterns_detected": 0,
            "suggestions_generated": 0,
            "alerts_created": 0,
            "start_time": None,
            "last_log_time": None
        }
        
        # Callbacks personalizables
        self.on_pattern_detected: Optional[Callable] = None
        self.on_suggestion_generated: Optional[Callable] = None
        self.on_alert_created: Optional[Callable] = None
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.lock = threading.Lock()
        
        # Configurar logging (ya se hace globalmente)
        # self._setup_logging()
    
    def _setup_gcp_client(self):
        """Configura el cliente de GCP Logging para streaming real"""
        try:
            self.logging_client = LoggingServiceV2Client()
            logging.info(f"Cliente GCP Logging (streaming) configurado para proyecto: {self.config.project_id}")
        except Exception as e:
            logging.exception(f"Error configurando cliente GCP: {e}")
            raise
    
    
    
    def start_streaming(self) -> None:
        """Inicia el streaming de logs en tiempo real"""
        if self.running:
            logging.warning("El analizador ya está ejecutándose")
            return
        
        self.running = True
        self.stats["start_time"] = datetime.utcnow()
        
        logging.info("🚀 Iniciando análisis de logs en tiempo real...")
        logging.info(f"Proyecto: {self.config.project_id}")
        logging.info(f"Filtro: {self.config.get_gcp_filter_string()}")
        
        # Configurar signal handlers para graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            self._stream_logs()
        except KeyboardInterrupt:
            logging.info("Interrupción recibida, deteniendo analizador...")
        except Exception as e:
            logging.exception(f"Error en streaming de logs: {e}")
        finally:
            self.stop_streaming()
    
    def _stream_logs(self):
        """Método principal de streaming de logs en tiempo real"""
        filter_string = self.config.get_gcp_filter_string()
        logging.info(f"📡 Conectando a GCP Logging (streaming) con filtro: {filter_string}")
        
        # Contador para mostrar que está esperando
        wait_counter = 0
        last_timestamp = None
        
        try:
            logging.info("⏳ Esperando logs en tiempo real... (Presiona Ctrl+C para detener)")
            
            while self.running:
                if self.paused:
                    time.sleep(1)
                    continue
                
                # Construir filtro con timestamp si tenemos uno anterior
                current_filter = filter_string
                if last_timestamp:
                    current_filter += f" AND timestamp>\"{last_timestamp}\""
                
                # Obtener logs recientes
                try:
                    request = {
                        "resource_names": [f"projects/{self.config.project_id}"],
                        "filter": current_filter,
                        "order_by": "timestamp desc",
                        "page_size": 10
                    }
                    
                    logging.info("[DEBUG] Iteración del stream: consultando logs de GCP")
                    
                    # Usar list_log_entries en lugar de tail_log_entries
                    page_result = self.logging_client.list_log_entries(request)
                    entries = list(page_result)
                    
                    if not entries:
                        logging.info("[DEBUG] Respuesta recibida pero sin logs nuevos.")
                    else:
                        logging.info(f"[DEBUG] Encontrados {len(entries)} logs nuevos")
                        # Procesar logs en orden cronológico (más antiguos primero)
                        for entry in reversed(entries):
                            self._process_log_entry(entry)
                            with self.lock:
                                self.stats["logs_processed"] += 1
                                self.stats["last_log_time"] = datetime.utcnow()
                            
                            # Actualizar timestamp del último log procesado
                            if entry.timestamp:
                                last_timestamp = entry.timestamp.ToDatetime().isoformat()
                        
                        if self.stats["logs_processed"] % 10 == 0:
                            self._log_progress()
                
                except Exception as e:
                    logging.exception(f"Error obteniendo logs: {e}")
                    time.sleep(5)  # Esperar antes de reintentar
                    continue
                        
        except google_exceptions.PermissionDenied:
            logging.exception("❌ Permisos insuficientes para acceder a logs de GCP")
            logging.error("Verifica que el service account tenga permisos de Logging")
        except google_exceptions.NotFound:
            logging.exception("❌ Proyecto o recurso no encontrado")
        except Exception as e:
            logging.exception(f"❌ Error en streaming de logs: {e}")
            self._handle_streaming_error(e)
    
    def _process_log_entry(self, log_entry):
        """Procesa una entrada de log individual"""
        try:
            # Convertir log entry a dict para procesamiento
            entry_dict = self._log_entry_to_dict(log_entry)
            
            # Detectar patrones
            pattern = self.pattern_detector.process_log_entry(entry_dict)
            
            if pattern:
                with self.lock:
                    self.stats["patterns_detected"] += 1
                
                # Generar sugerencias
                suggestions = self.suggestion_engine.generate_suggestions(pattern)
                
                with self.lock:
                    self.stats["suggestions_generated"] += len(suggestions)
                
                # Crear alerta
                alert = self.alert_manager.create_alert(pattern, suggestions)
                
                if alert:
                    with self.lock:
                        self.stats["alerts_created"] += 1
                
                # Ejecutar callbacks
                self._execute_callbacks(pattern, suggestions, alert)
                
                # Log detallado si está en modo verbose
                if self.config.verbose:
                    self._log_detailed_analysis(pattern, suggestions, alert)
        
        except Exception as e:
            logging.exception(f"Error procesando entrada de log: {e}")
    
    def _log_entry_to_dict(self, log_entry) -> Dict[str, Any]:
        """Convierte log entry de GCP a diccionario"""
        entry_dict = {
            "timestamp": log_entry.timestamp.ToDatetime().isoformat(),
            "severity": log_entry.severity.name,
            "resource": {
                "type": log_entry.resource.type,
                "labels": dict(log_entry.resource.labels)
            }
        }
        
        # Extraer payload
        if log_entry.text_payload:
            entry_dict["textPayload"] = log_entry.text_payload
        elif log_entry.json_payload:
            entry_dict["jsonPayload"] = json.loads(log_entry.json_payload)
        
        # Extraer labels
        if log_entry.labels:
            entry_dict["labels"] = dict(log_entry.labels)
        
        return entry_dict
    
    def _execute_callbacks(self, pattern: DetectedPattern, suggestions: List[ImprovementSuggestion], alert: Optional[Alert]):
        """Ejecuta callbacks personalizados"""
        if self.on_pattern_detected:
            try:
                self.on_pattern_detected(pattern)
            except Exception as e:
                logging.exception(f"Error en callback on_pattern_detected: {e}")
        
        if self.on_suggestion_generated:
            try:
                self.on_suggestion_generated(suggestions)
            except Exception as e:
                logging.exception(f"Error en callback on_suggestion_generated: {e}")
        
        if self.on_alert_created and alert:
            try:
                self.on_alert_created(alert)
            except Exception as e:
                logging.exception(f"Error en callback on_alert_created: {e}")
    
    def _log_detailed_analysis(self, pattern: DetectedPattern, suggestions: List[ImprovementSuggestion], alert: Optional[Alert]):
        """Log detallado del análisis (solo en modo verbose)"""
        logging.debug(f"🔍 Patrón detectado: {pattern.pattern_type} ({pattern.severity.value})")
        logging.debug(f"   Impacto: {pattern.impact_analysis}")
        logging.debug(f"   Sugerencias generadas: {len(suggestions)}")
        if alert:
            logging.debug(f"   Alerta creada: {alert.id}")
    
    def _log_progress(self):
        """Log de progreso del análisis"""
        uptime = datetime.utcnow() - self.stats["start_time"]
        logging.info(f"📊 Progreso: {self.stats['logs_processed']} logs procesados, "
                    f"{self.stats['patterns_detected']} patrones, "
                    f"{self.stats['alerts_created']} alertas (uptime: {uptime})")
    
    def _handle_streaming_error(self, error: Exception):
        """Maneja errores de streaming con reintentos"""
        logging.exception(f"Error de streaming: {error}")
        
        if self.config.retry_attempts > 0:
            logging.info(f"Reintentando en 5 segundos... ({self.config.retry_attempts} intentos restantes)")
            time.sleep(5)
            self.config.retry_attempts -= 1
            self._stream_logs()
        else:
            logging.error("Se agotaron los intentos de reconexión")
            self.stop_streaming()
    
    def _signal_handler(self, signum, frame):
        """Maneja señales de interrupción"""
        logging.info(f"Señal {signum} recibida, deteniendo analizador...")
        self.stop_streaming()
    
    def stop_streaming(self):
        """Detiene el streaming de logs"""
        if not self.running:
            return
        
        self.running = False
        logging.info("🛑 Deteniendo análisis de logs...")
        
        # Cerrar executor
        self.executor.shutdown(wait=True)
        
        # Log de estadísticas finales
        self._log_final_stats()
    
    def pause_streaming(self):
        """Pausa el streaming de logs"""
        self.paused = True
        logging.info("⏸️ Streaming pausado")
    
    def resume_streaming(self):
        """Reanuda el streaming de logs"""
        self.paused = False
        logging.info("▶️ Streaming reanudado")
    
    def _log_final_stats(self):
        """Log de estadísticas finales"""
        if self.stats["start_time"]:
            uptime = datetime.utcnow() - self.stats["start_time"]
            logging.info(f"📈 Estadísticas finales:")
            logging.info(f"   Tiempo total: {uptime}")
            logging.info(f"   Logs procesados: {self.stats['logs_processed']}")
            logging.info(f"   Patrones detectados: {self.stats['patterns_detected']}")
            logging.info(f"   Sugerencias generadas: {self.stats['suggestions_generated']}")
            logging.info(f"   Alertas creadas: {self.stats['alerts_created']}")
            
            if self.stats['logs_processed'] > 0:
                pattern_rate = self.stats['patterns_detected'] / self.stats['logs_processed']
                logging.info(f"   Tasa de patrones: {pattern_rate:.2%}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas actuales del analizador"""
        with self.lock:
            stats_copy = self.stats.copy()
        
        if stats_copy["start_time"]:
            stats_copy["uptime"] = str(datetime.utcnow() - stats_copy["start_time"])
        
        return stats_copy
    
    def get_patterns_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de patrones detectados"""
        patterns = self.pattern_detector.detected_patterns
        
        summary = {
            "total_patterns": len(patterns),
            "by_type": {},
            "by_severity": {},
            "recent_patterns": len(self.pattern_detector.get_recent_patterns())
        }
        
        # Agrupar por tipo
        for pattern in patterns:
            pattern_type = pattern.pattern_type
            summary["by_type"][pattern_type] = summary["by_type"].get(pattern_type, 0) + 1
        
        # Agrupar por severidad
        for pattern in patterns:
            severity = pattern.severity.value
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
        
        return summary
    
    def export_analysis_report(self, filepath: str) -> None:
        """Exporta reporte completo de análisis"""
        report = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "project_id": self.config.project_id,
                "filter": self.config.get_gcp_filter_string()
            },
            "statistics": self.get_stats(),
            "patterns_summary": self.get_patterns_summary(),
            "alerts_summary": self.alert_manager.get_alert_summary(),
            "patterns": [
                {
                    "type": p.pattern_type,
                    "severity": p.severity.value,
                    "message": p.message,
                    "impact": p.impact_analysis,
                    "timestamp": p.timestamp.isoformat(),
                    "confidence": p.confidence
                }
                for p in self.pattern_detector.detected_patterns
            ],
            "suggestions": [
                {
                    "id": s.id,
                    "title": s.title,
                    "type": s.suggestion_type.value,
                    "priority": s.priority.value,
                    "effort": s.estimated_effort,
                    "implementation_steps": s.implementation_steps
                }
                for s in self.suggestion_engine.generated_suggestions
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logging.info(f"📄 Reporte exportado a: {filepath}")
    
    def set_callbacks(
        self,
        on_pattern_detected: Optional[Callable] = None,
        on_suggestion_generated: Optional[Callable] = None,
        on_alert_created: Optional[Callable] = None
    ):
        """Configura callbacks personalizados"""
        self.on_pattern_detected = on_pattern_detected
        self.on_suggestion_generated = on_suggestion_generated
        self.on_alert_created = on_alert_created
    
    def is_running(self) -> bool:
        """Verifica si el analizador está ejecutándose"""
        return self.running
    
    def is_paused(self) -> bool:
        """Verifica si el analizador está pausado"""
        return self.paused 