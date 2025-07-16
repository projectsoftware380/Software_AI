# --------------------------------------------------------------------------
# validate_hybrid_flow.py: Orquesta y valida el flujo híbrido en Python.
# --------------------------------------------------------------------------

import subprocess
import sys
import logging
import shlex
import os

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_step(command, step_name):
    """Ejecuta un paso del flujo de trabajo y maneja los errores."""
    logging.info(f"--- PASO: Ejecutando {step_name}... ---")
    try:
        # Añadir el directorio actual al PYTHONPATH para que Python encuentre los módulos de 'src'
        env = os.environ.copy()
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{os.getcwd()};{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = os.getcwd()

        # Determinar el ejecutable de Python del venv
        python_executable_venv = os.path.join(os.getcwd(), "venv", "Scripts", "python.exe")

        # Para comandos que no son de python, usamos el ejecutable de bash
        if command.startswith('bash'):
            # Usar una cadena raw para la ruta de Windows para evitar problemas con las barras invertidas
            bash_executable = r"C:\Program Files\Git\bin\bash.exe"
            # Dividimos el comando para pasarlo de forma segura
            cmd_parts = [bash_executable] + command.split()[1:]
            subprocess.run(cmd_parts, check=True, text=True, env=env)
        else: # Para comandos de python
            # Para Python, ejecutamos directamente con el intérprete de Python del venv
            cmd_parts = shlex.split(command)
            # Reemplazar 'python' con la ruta completa al ejecutable del venv
            if cmd_parts[0] == "python":
                cmd_parts[0] = python_executable_venv
            subprocess.run(cmd_parts, check=True, text=True, env=env)
            
        logging.info(f"--- PASO: {step_name} completado exitosamente. ---")

    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Fallo en el paso: {step_name}.")
        logging.error(f"Comando: {e.cmd}")
        logging.error(f"Código de salida: {e.returncode}")
        logging.error(f"Salida: {e.stdout}")
        logging.error(f"Error: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        logging.error("Error: El ejecutable de bash no se encontró en 'C:\Program Files\Git\bin\bash.exe'")
        logging.error("Asegúrate de que Git para Windows esté instalado en la ruta por defecto o ajusta la ruta.")
        sys.exit(1)

def main():
    """Función principal que orquesta el flujo híbrido."""
    logging.info("======================================================")
    logging.info("🚀 INICIANDO VALIDACIÓN DEL FLUJO HÍBRIDO COMPLETO")
    logging.info("======================================================")

    # Imprimir información del entorno para depuración
    logging.info(f"Python executable: {sys.executable}")
    logging.info(f"Python path: {sys.path}")

    steps = [
        ("bash ./scripts/run_local_ingestion.sh", "Ingesta de datos local"),
        ("bash ./scripts/run_local_preparation.sh", "Preparación de datos local"),
        ("python ./scripts/run_gcp_training.py", "Lanzamiento de pipeline en GCP"),
        ("bash ./scripts/run_local_backtest.sh", "Backtesting local"),
        ("bash ./scripts/run_local_promotion.sh", "Promoción de modelos local"),
    ]

    for command, name in steps:
        run_step(command, name)

    logging.info("======================================================")
    logging.info("✅ VALIDACIÓN DEL FLUJO HÍBRIDO COMPLETADA EXITOSAMENTE")
    logging.info("======================================================")

if __name__ == "__main__":
    main()
