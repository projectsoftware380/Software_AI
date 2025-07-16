#!/usr/bin/env python3
"""
Script Principal del Analizador de Logs en Tiempo Real
======================================================

Uso:
    python main.py --project-id=my-project --config=config.yaml --verbose
    python main.py --help
"""

import argparse
import os
import sys
import signal
import logging
from pathlib import Path
from typing import Optional

# Agregar el directorio src al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.log_analyzer import (
    RealtimeLogAnalyzer,
    LogAnalyzerConfig,
    create_default_config
)


def parse_arguments():
    """Parsea argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Analizador de Logs en Tiempo Real para Pipeline de Trading Forex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py --project-id=my-trading-project
  python main.py --config=my_config.yaml --verbose
  python main.py --create-config
  python main.py --export-report=report.json
        """
    )
    
    # Argumentos principales
    parser.add_argument(
        "--project-id",
        type=str,
        help="ID del proyecto de GCP (default: desde config o PROJECT_ID env)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="log_analyzer_config.yaml",
        help="Archivo de configuración YAML (default: log_analyzer_config.yaml)"
    )
    
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Crear archivo de configuración por defecto y salir"
    )
    
    parser.add_argument(
        "--export-report",
        type=str,
        help="Exportar reporte de análisis a archivo JSON"
    )
    
    # Opciones de comportamiento
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Modo verbose con logging detallado"
    )
    
    parser.add_argument(
        "--severity-threshold",
        choices=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default="MEDIUM",
        help="Umbral mínimo de severidad para alertas (default: MEDIUM)"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="Archivo de salida para logs de análisis"
    )
    
    parser.add_argument(
        "--console-output",
        action="store_true",
        default=True,
        help="Mostrar output en consola (default: True)"
    )
    
    # Opciones de filtros
    parser.add_argument(
        "--pod-app-label",
        type=str,
        default="trading-pipeline",
        help="Label de la aplicación en Kubernetes (default: trading-pipeline)"
    )
    
    parser.add_argument(
        "--min-severity",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Severidad mínima de logs a procesar (default: WARNING)"
    )
    
    return parser.parse_args()


def setup_environment():
    """Configura variables de entorno y logging"""
    # Configurar logging básico
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Verificar credenciales de GCP
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and not os.getenv("GCP_PROJECT_ID"):
        logging.warning("⚠️  No se encontraron credenciales de GCP configuradas")
        logging.warning("   Asegúrate de tener configurado gcloud auth o GOOGLE_APPLICATION_CREDENTIALS")


def create_config_file(config_path: str):
    """Crea archivo de configuración por defecto"""
    try:
        config = create_default_config(config_path)
        print(f"✅ Archivo de configuración creado: {config_path}")
        print(f"   Proyecto: {config.project_id}")
        print(f"   Región: {config.region}")
        print(f"   Filtro de logs: {config.get_gcp_filter_string()}")
        print("\n📝 Puedes editar el archivo para personalizar la configuración")
        return True
    except Exception as e:
        print(f"❌ Error creando archivo de configuración: {e}")
        return False


def load_config(args) -> Optional[LogAnalyzerConfig]:
    """Carga configuración desde archivo o argumentos"""
    try:
        # Intentar cargar desde archivo
        if os.path.exists(args.config):
            config = LogAnalyzerConfig.from_yaml(args.config)
            print(f"📁 Configuración cargada desde: {args.config}")
        else:
            # Crear configuración por defecto
            config = LogAnalyzerConfig()
            print(f"📁 Usando configuración por defecto (archivo {args.config} no encontrado)")
        
        # Sobrescribir con argumentos de línea de comandos
        if args.project_id:
            config.project_id = args.project_id
        
        if args.verbose:
            config.verbose = True
        
        if args.output_file:
            config.output_file = args.output_file
        
        if args.console_output is not None:
            config.console_output = args.console_output
        
        # Actualizar filtros
        config.log_filters["pod_app_label"] = args.pod_app_label
        config.log_filters["min_severity"] = args.min_severity
        
        # Validar configuración
        if not config.validate():
            print("❌ Configuración inválida")
            return None
        
        return config
        
    except Exception as e:
        print(f"❌ Error cargando configuración: {e}")
        return None


def setup_analyzer_callbacks(analyzer: RealtimeLogAnalyzer):
    """Configura callbacks personalizados para el analizador"""
    
    def on_pattern_detected(pattern):
        """Callback cuando se detecta un patrón"""
        if analyzer.config.verbose:
            logging.debug(f"🔍 Patrón detectado: {pattern.pattern_type} ({pattern.severity.value})")
    
    def on_suggestion_generated(suggestions):
        """Callback cuando se generan sugerencias"""
        if analyzer.config.verbose:
            logging.debug(f"💡 {len(suggestions)} sugerencias generadas")
    
    def on_alert_created(alert):
        """Callback cuando se crea una alerta"""
        if analyzer.config.verbose:
            logging.debug(f"🚨 Alerta creada: {alert.id} ({alert.severity.value})")
    
    analyzer.set_callbacks(
        on_pattern_detected=on_pattern_detected,
        on_suggestion_generated=on_suggestion_generated,
        on_alert_created=on_alert_created
    )


def print_startup_banner(config: LogAnalyzerConfig):
    """Imprime banner de inicio"""
    print("\n" + "="*80)
    print("🚀 ANALIZADOR DE LOGS EN TIEMPO REAL - PIPELINE DE TRADING FOREX")
    print("="*80)
    print(f"📊 Proyecto: {config.project_id}")
    print(f"🌍 Región: {config.region}")
    print(f"🔍 Filtro: {config.get_gcp_filter_string()}")
    print(f"⚙️  Modo: {'Verbose' if config.verbose else 'Normal'}")
    print(f"📁 Output: {config.output_file}")
    print("="*80)
    print("💡 Presiona Ctrl+C para detener el análisis")
    print("="*80 + "\n")


def main():
    """Función principal"""
    # Parsear argumentos
    args = parse_arguments()
    
    # Configurar entorno
    setup_environment()
    
    # Crear configuración si se solicita
    if args.create_config:
        success = create_config_file(args.config)
        sys.exit(0 if success else 1)
    
    # Cargar configuración
    config = load_config(args)
    if not config:
        print("❌ No se pudo cargar la configuración")
        sys.exit(1)
    
    # Configurar severidad de alertas
    from src.log_analyzer.patterns import Severity
    severity_threshold = Severity(args.severity_threshold)
    
    try:
        # Crear analizador
        analyzer = RealtimeLogAnalyzer(config)
        
        # Configurar umbral de severidad
        analyzer.alert_manager.set_severity_threshold(severity_threshold)
        
        # Configurar callbacks
        setup_analyzer_callbacks(analyzer)
        
        # Mostrar banner de inicio
        print_startup_banner(config)
        
        # Exportar reporte si se solicita (sin iniciar streaming)
        if args.export_report:
            print(f"📄 Exportando reporte a: {args.export_report}")
            analyzer.export_analysis_report(args.export_report)
            return
        
        # Iniciar análisis
        analyzer.start_streaming()
        
    except KeyboardInterrupt:
        print("\n🛑 Análisis interrumpido por el usuario")
    except Exception as e:
        print(f"❌ Error en el análisis: {e}")
        logging.error(f"Error detallado: {e}", exc_info=True)
        sys.exit(1)
    finally:
        print("\n👋 Análisis finalizado")


if __name__ == "__main__":
    main() 