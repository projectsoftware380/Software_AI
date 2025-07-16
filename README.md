# Software_AI

Este repositorio contiene las fuentes para la tubería de trading algorítmico.

## Estructura del Proyecto

```
Software_AI/
├── src/
│   ├── components/         # Componentes individuales de la pipeline (ej. ingestión, preparación)
│   │   └── dukascopy_ingestion/ # Nuevo componente de ingestión de datos desde Dukascopy
│   ├── config/             # Configuración centralizada de la pipeline
│   ├── deploy/             # Lógica de despliegue y CLI para Vertex AI
│   ├── pipeline/           # Definición de la pipeline de Kubeflow
│   ├── log_analyzer/       # Módulo para análisis de logs en tiempo real
│   └── shared/             # Módulos compartidos (constantes, utilidades GCS, logging)
├── scripts/                # Scripts de automatización (ej. despliegue a GCP)
├── tests/                  # Pruebas unitarias e integración
├── docs/                   # Documentación del proyecto
├── requirements.txt        # Dependencias del proyecto
├── pyproject.toml          # Configuración del paquete Python
└── ... (otros archivos de configuración y logs)
```

## Instalación

1.  Clona el repositorio y navega a su directorio.
2.  Instala las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

## Despliegue de la Pipeline

Para compilar y desplegar la pipeline en Google Cloud Vertex AI, utiliza el script de despliegue:

```bash
bash scripts/deploy_to_gcp.sh
```

Este script se encargará de:
1.  Construir la imagen Docker común para los componentes.
2.  Subir la imagen a Google Artifact Registry.
3.  Compilar la definición de la pipeline en un archivo JSON.
4.  Lanzar la pipeline en Vertex AI.

## Ejecución de Tests

Después de instalar las dependencias, puedes ejecutar las pruebas unitarias con `pytest`:

```bash
pytest
```

Para detalles sobre la corrección crítica de errores en la pipeline v5, consulta [docs/Pipeline_v5_error_fix.md](docs/Pipeline_v5_error_fix.md).