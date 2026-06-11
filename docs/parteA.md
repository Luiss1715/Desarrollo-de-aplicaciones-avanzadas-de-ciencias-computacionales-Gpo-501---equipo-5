# Parte A - Diseno conceptual

## 1. Descripcion del modelo de solucion
La solucion es un pipeline de seis etapas: ingesta, analisis lexico, preprocesamiento NLP, extraccion de caracteristicas, clasificacion y generacion de salida. Cada etapa tiene una interfaz clara y datos de entrada y salida definidos.

## 2. Componentes principales
- M1 Ingesta: valida y normaliza texto de entrada.
- M2 Analisis lexico: lexer con patrones de riesgo.
- M3 Preprocesamiento: limpieza, stopwords, lematizacion opcional.
- M4 Caracteristicas: TF-IDF mas banderas lexicas.
- M5 Clasificacion: modelo de ML para etiqueta binaria.
- M6 Salida: JSON con etiqueta, score, tokens criticos y version.

## 3. Relaciones y dependencias
- M1 -> M2 -> M3 -> M4 -> M5 -> M6 (flujo secuencial).
- M2 -> M4 (las banderas lexicas se incorporan a las caracteristicas).
- Dependencias internas: M4 requiere vectorizador entrenado; M5 requiere modelo entrenado.

## 4. Datos de entrada y salida
Entrada: texto (string), source_id opcional, timestamp opcional.
Salida: risk_label {low, high}, risk_score [0,1], alert_tokens, model_version.
