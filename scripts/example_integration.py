#!/usr/bin/env python3
"""
Ejemplo de Integración: Pipeline + Análisis de Logs
==================================================

Este script demuestra cómo integrar el pipeline principal con el
sistema de análisis de logs en una sola ejecución.
"""

import os
import sys
import time
import threading
import subprocess
from pathlib import Path

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.log_analyzer import RealtimeLogAnalyzer, LogAnalyzerConfig
from src.shared import constants


class IntegratedPipelineRunner:
    """Ejecutor integrado de pipeline con análisis de logs"""
    
    def __init__(self, project_id: str, enable_logs: bool = True):
        self.project_id = project_id
        self.enable_logs = enable_logs
        self.log_analyzer = None
        self.log_thread = None
        self.pipeline_process = None
        
    def setup_log_analyzer(self):
        """Configura e inicia el analizador de logs"""
        if not self.enable_logs:
            print("📝 Análisis de logs deshabilitado")
            return
            
        try:
            # Cargar configuración
            config = LogAnalyzerConfig()
            config.project_id = self.project_id
            config.verbose = True
            
            # Crear analizador
            self.log_analyzer = RealtimeLogAnalyzer(config)
            
            # Configurar callbacks
            def on_pattern_detected(pattern):
                print(f"🚨 PATRÓN DETECTADO: {pattern.pattern_type} - {pattern.message}")
                
            def on_alert_created(alert):
                print(f"🚨 ALERTA CRÍTICA: {alert.id} - {alert.message}")
                
            self.log_analyzer.set_callbacks(
                on_pattern_detected=on_pattern_detected,
                on_alert_created=on_alert_created
            )
            
            # Función para ejecutar en hilo separado
            def run_log_analyzer():
                try:
                    print("🚀 Iniciando analizador de logs...")
                    self.log_analyzer.start_streaming()
                except Exception as e:
                    print(f"❌ Error en analizador de logs: {e}")
                    
            # Iniciar en hilo separado
            self.log_thread = threading.Thread(target=run_log_analyzer, daemon=True)
            self.log_thread.start()
            
            print("✅ Analizador de logs iniciado en segundo plano")
            
        except Exception as e:
            print(f"❌ Error configurando analizador de logs: {e}")
            
    def run_pipeline(self, image_uri: str):
        """Ejecuta el pipeline principal"""
        try:
            print(f"🚀 Lanzando pipeline con imagen: {image_uri}")
            
            # Comando para ejecutar el pipeline
            cmd = [
                sys.executable, "-m", "src.pipeline.main",
                "--common-image-uri", image_uri
            ]
            
            # Ejecutar pipeline
            self.pipeline_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitorear salida en tiempo real
            while True:
                output = self.pipeline_process.stdout.readline()
                if output == '' and self.pipeline_process.poll() is not None:
                    break
                if output:
                    print(f"📊 Pipeline: {output.strip()}")
                    
            # Obtener código de salida
            return_code = self.pipeline_process.wait()
            
            if return_code == 0:
                print("✅ Pipeline completado exitosamente")
                return True
            else:
                print(f"❌ Pipeline falló con código: {return_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error ejecutando pipeline: {e}")
            return False
            
    def stop_log_analyzer(self):
        """Detiene el analizador de logs"""
        if self.log_analyzer and self.log_analyzer.is_running():
            print("🛑 Deteniendo analizador de logs...")
            self.log_analyzer.stop_streaming()
            
            if self.log_thread and self.log_thread.is_alive():
                self.log_thread.join(timeout=10)
                
            # Exportar reporte final
            try:
                report_path = f"integrated_analysis_report_{int(time.time())}.json"
                self.log_analyzer.export_analysis_report(report_path)
                print(f"📄 Reporte exportado: {report_path}")
            except Exception as e:
                print(f"⚠️ Error exportando reporte: {e}")
                
    def run_integrated_execution(self, image_uri: str):
        """Ejecuta la integración completa"""
        print("=========================================================================")
        print("🚀 EJECUCIÓN INTEGRADA: Pipeline + Análisis de Logs")
        print("=========================================================================")
        
        try:
            # Paso 1: Iniciar analizador de logs
            if self.enable_logs:
                self.setup_log_analyzer()
                time.sleep(5)  # Esperar inicialización
                
            # Paso 2: Ejecutar pipeline
            success = self.run_pipeline(image_uri)
            
            # Paso 3: Mostrar estadísticas finales
            if self.enable_logs and self.log_analyzer:
                stats = self.log_analyzer.get_stats()
                print("\n📊 Estadísticas finales del analizador:")
                print(f"   - Logs procesados: {stats.get('logs_processed', 0)}")
                print(f"   - Patrones detectados: {stats.get('patterns_detected', 0)}")
                print(f"   - Alertas creadas: {stats.get('alerts_created', 0)}")
                
            # Paso 4: Detener analizador
            if self.enable_logs:
                self.stop_log_analyzer()
                
            return success
            
        except KeyboardInterrupt:
            print("\n🛑 Interrupción detectada, limpiando...")
            if self.enable_logs:
                self.stop_log_analyzer()
            return False
        except Exception as e:
            print(f"❌ Error en ejecución integrada: {e}")
            if self.enable_logs:
                self.stop_log_analyzer()
            return False


def main():
    """Función principal de ejemplo"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ejemplo de integración pipeline + logs")
    parser.add_argument("--project-id", default=constants.PROJECT_ID, help="ID del proyecto GCP")
    parser.add_argument("--image-uri", required=True, help="URI de la imagen Docker")
    parser.add_argument("--no-logs", action="store_true", help="Deshabilitar análisis de logs")
    
    args = parser.parse_args()
    
    # Crear ejecutor integrado
    runner = IntegratedPipelineRunner(
        project_id=args.project_id,
        enable_logs=not args.no_logs
    )
    
    # Ejecutar integración
    success = runner.run_integrated_execution(args.image_uri)
    
    if success:
        print("\n🎉 ¡Ejecución integrada completada exitosamente!")
    else:
        print("\n❌ Ejecución integrada falló")
        sys.exit(1)


if __name__ == "__main__":
    main() 