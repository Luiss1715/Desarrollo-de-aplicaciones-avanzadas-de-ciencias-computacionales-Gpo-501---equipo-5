# Fase 3: NLI, espacio latente y red neuronal

## Enfoque

La Fase 3 incorpora dos métodos que complementan el pipeline clásico de Fase 2. El
primero representa cada publicación en un espacio latente mediante embeddings de
Transformer y entrena una red neuronal sobre esos vectores. El segundo usa
inferencia de lenguaje natural (NLI) para emitir un dictamen zero-shot.

En ambos métodos, la entrada es la concatenación de `title` y `text`. La salida
incluye una etiqueta final `yes/no` y un score continuo entre 0 y 1.

## Espacio latente y MLP

El modelo `sentence-transformers/all-MiniLM-L6-v2` es la opción predeterminada para
convertir cada publicación en un vector semántico. A diferencia de TF-IDF, estos
vectores aproximan relaciones de significado aunque dos textos no compartan las
mismas palabras.

Sobre los embeddings se entrena una red neuronal multicapa con dos capas ocultas,
activaciones ReLU y dropout. Una red de este tipo puede aprender fronteras no
lineales en el espacio latente, lo que permite combinar dimensiones semánticas que
una clasificación lineal no representaría directamente. El entrenamiento usa
`BCEWithLogitsLoss` y un split estratificado de entrenamiento y validación.

## NLI zero-shot

NLI determina si una premisa implica, contradice o es neutral respecto de una
hipótesis. En este proyecto, cada publicación funciona como premisa y el modelo
MNLI evalúa etiquetas relacionadas con la hipótesis principal: "This text
expresses suicidal ideation." El score asignado a la etiqueta de ideación suicida
se compara con un umbral configurable, de 0.5 por defecto, para producir el
dictamen `yes/no`.

El modelo predeterminado es `facebook/bart-large-mnli`. Es posible cambiarlo por
una alternativa MNLI más pequeña mediante `--model-name` o `--nli-model` cuando
los recursos de cómputo sean limitados.

## Comparación con Fase 2

El comando de comparación crea un único split estratificado. Fase 2 y la MLP se
entrenan con la misma partición de entrenamiento, mientras que Fase 2, MLP y NLI
se evalúan sobre la misma partición de prueba. Esto conserva una base común para
comparar los dictámenes.

La métrica principal sigue el protocolo del curso:

`AUC = (1 + TPR - FPR) / 2`

También se registran TP, TN, FP, FN, TPR y FPR. Cuando existen scores continuos,
se agrega ROC-AUC de scikit-learn como referencia, pero no reemplaza la AUC del
protocolo.

La mejora no debe analizarse solo por AUC. También deben revisarse los falsos
negativos, por su relevancia en detección de riesgo; los falsos positivos; la
interpretabilidad de las reglas y predicciones; y el costo computacional de
generar embeddings o ejecutar un modelo MNLI.

## Artefactos

La ejecución genera modelos en `models/`, predicciones y métricas en `reports/`,
y una proyección PCA del espacio latente. UMAP es opcional y solo se genera si el
paquete `umap-learn` está instalado.
