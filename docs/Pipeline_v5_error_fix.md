# Corrección de error crítico en la pipeline v5

Este documento resume el problema detectado en la versión v5 del pipeline de trading y las soluciones aplicadas.

## Descripción del error

Durante la ejecución del componente `train_lstm` se producía un `KeyError: 'sma_len'` al intentar construir los indicadores técnicos. El archivo `params.json` generado en la etapa previa no contenía dicha clave porque el componente `optimize_trading_logic` no fusionaba correctamente los parámetros de arquitectura con los de lógica de trading.

## Soluciones implementadas

- **Propagación correcta de rutas**: en `src/pipeline/main.py` se ajustó la ruta de entrada a `optimize_trading_logic` usando el par del bucle `ParallelFor`.
- **Fusión de parámetros**: en `src/components/optimize_trading_logic/task.py` ahora se combinan los parámetros de arquitectura y los de lógica en `best_final_params`.
- **Entrenamiento robusto**: el componente `train_lstm` sigue sin cambios, pero recibe ahora un `params.json` completo.
- **Ajustes menores**: en `src/components/backtest/task.py` se añade `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"` para evitar problemas de inicialización de GPU.

El archivo `algo_trading_mlops_pipeline_v5_corrected.json` contiene la definición de pipeline actualizada.

## Recomendaciones

1. Revisar el código actualizado para entender la gestión de rutas y parámetros.
2. Añadir validaciones de entradas en cada componente.
3. Fortalecer las pruebas unitarias en `tests/test_components.py`.
4. Mantener la documentación de cada componente en su `component.yaml`.

