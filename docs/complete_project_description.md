# Documentación detallada del proyecto "suicidality"

Última actualización: 2026-06-11

Propósito: en este documento detallo, con todo el rigor posible, cada componente, flujo de datos, artefacto, comando y decisión técnica del proyecto. Está pensado para cualquier lector técnico (incluyéndome a mí mismo en el futuro) que necesite entender a fondo los archivos, los modelos y cómo ejecutar y reproducir los experimentos.

Índice
- Resumen general
- Estructura del repositorio y propósito de cada archivo
- Flujo de datos: ingestión y preprocesamiento
- Fase 2: extracción de features, modelado y pipeline
- Fase 3: embeddings, red latente, NLI y ensamblado
- Modelos y artefactos en `models/`
- Reportes y métricas en `reports/`
- CLI y puntos de entrada (`__main__`, `cli.py`, `compare_phase2_phase3.py`)
- Configuración y parámetros ejecutables
- Dispositivos y ejecuciones GPU/CPU
- Tests y su cobertura
- Buenas prácticas, reproducibilidad y troubleshooting
- Cómo generar el PDF (comando sugerido)


**Resumen general**

En este proyecto implementé un sistema para clasificar texto orientado a la detección de ideación suicida o riesgo (suicidality). Organicé el trabajo en dos fases experimentales principales:

- `fase2`: extracción de features tradicionales y modelos clásicos.
- `fase3`: representación densa de texto a través de embeddings, una red latente y un clasificador NLI; además, incorporo un ensamblador que combina las salidas cuando conviene.

Las funcionalidades que desarrollé incluyen:
- Ingesta y preprocesamiento de datasets en formato CSV.
- Extracción de features léxicas y estadísticas.
- Entrenamiento e inferencia de pipelines basados en scikit-learn.
- Generación de embeddings mediante modelos de lenguaje y uso de una red latente (`latent_nn`) para producir features o predicciones.
- Evaluación y comparación de resultados (métricas, tablas y reportes JSON/CSV).
- Un script CLI (`compare_phase2_phase3`) que orquesta la comparación entre ambas fases y genera los reportes necesarios.


**Estructura del repositorio y propósito de cada archivo/directorio**

Raíz
- `pyproject.toml`: configuración del paquete (busco paquetes en `src`), útil para instalar `suicidality` localmente.
- `requirements.txt`: lista de dependencias que debo instalar en el entorno antes de ejecutar experimentos.
- `README.md`: resumen para usuarios y apuntes rápidos.

Directorios principales
- `src/suicidality/` (paquete principal)
  - `__init__.py`: inicialización del paquete.
  - `__main__.py`: punto de entrada para ejecutar `python -m suicidality` cuando el paquete está disponible en el entorno o `PYTHONPATH` apunta a `src`.
  - `cli.py`: definición de la interfaz de línea de comandos y argumentos expuestos.
  - `compare_phase2_phase3.py`: orquestador que carga datasets, aplica pipelines de fase2 y fase3, calcula métricas y escribe reportes en `reports/`.
  - `config.py`: parámetros por defecto, nombres de columnas esperadas y otros ajustes globales.
  - `evaluate.py`: utilidades para cálculo de métricas (precision, recall, F1, AUC) y generación de resúmenes.
  - `llm_prompt_classifier.py`: helpers para clasificación mediante prompts a LLMs (si se emplean).
  - `protocol_metrics.py`: adaptadores para métricas específicas o protocolos de evaluación que uso en los experimentos.

- `src/suicidality/fase2/`
  - `ingest.py`: lectura y validación de CSVs; normalizo columnas (`id`, `text`, `label`) y trato valores faltantes.
  - `preprocess.py`: limpieza y normalización de texto (lowercasing, Unicode normalization, eliminación de URLs/mentions, minúsculas, etc.).
  - `lexing.py`: extracción léxica basada en conteos, detección de términos del léxico de riesgo, n-grams y features booleanos.
  - `features.py`: ensamblaje de vectores de features (TF-IDF, conteos, estadísticas por documento).
  - `model.py`: wrappers y utilidades para entrenar y predecir con estimadores de scikit-learn.
  - `pipeline.py`: construcción del `sklearn.pipeline.Pipeline` que encadena transformaciones y el estimador final; funciones para guardar/cargar con `joblib`.

