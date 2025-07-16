#!/usr/bin/env python3
"""
Script de Prueba: Analizador en Espera Continua
===============================================

Este script ejecuta el analizador con un filtro más amplio para capturar
cualquier log del proyecto y se queda esperando indefinidamente.
"""

import sys
import os
from pathlib import Path

# Agregar el directorio src al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.log_analyzer import LogAnalyzerConfig, RealtimeLogAnalyzer


def create_broad_filter_config():
    """Crea una configuración con filtro lo más amplio posible"""
    config = LogAnalyzerConfig()
    config.log_filters = {
        # No filtrar por resource_type ni pod_app_label
        # Solo filtrar por severidad mínima
        "min_severity": "INFO"
    }
    return config


def main():
    """Función principal"""
    print("🚀 ANALIZADOR EN MODO ESPERA CONTINUA")
    print("=" * 50)
    print("📊 Proyecto: trading-ai-460823")
    print("🔍 Filtro: Cualquier log del proyecto (severidad >= INFO)")
    print("⏳ El analizador se quedará esperando indefinidamente")
    print("💡 Presiona Ctrl+C para detener")
    print("=" * 50)
    
    try:
        print("🔧 Inicializando analizador...")
        
        # Crear configuración con filtro amplio
        config = create_broad_filter_config()
        print("✅ Configuración creada")
        
        # Crear analizador
        print("🔧 Creando analizador...")
        analyzer = RealtimeLogAnalyzer(config)
        print("✅ Analizador creado")
        
        # Configurar callbacks simples
        def on_pattern_detected(pattern):
            print(f"\n🔍 PATRÓN DETECTADO: {pattern.pattern_type} ({pattern.severity.value})")
            print(f"   Mensaje: {pattern.message}")
        
        def on_alert_created(alert):
            print(f"\n🚨 ALERTA: {alert.id} ({alert.severity.value})")
        
        analyzer.set_callbacks(
            on_pattern_detected=on_pattern_detected,
            on_alert_created=on_alert_created
        )
        print("✅ Callbacks configurados")
        
        # Iniciar streaming
        print("🚀 Iniciando streaming...")
        analyzer.start_streaming()
        
    except KeyboardInterrupt:
        print("\n🛑 Analizador detenido por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔍 Información adicional de debug:")
        print(f"   Tipo de error: {type(e).__name__}")
        print(f"   Mensaje: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    main() 