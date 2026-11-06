# INF-8239 · Unidad 02 · Ejercicio 04 · Clasificador visual con CNN y Model Card

Proyecto reproducible del laboratorio **U02.LAB07** (Visión computacional: CNN
reproducible con Fashion-MNIST).

| | |
|---|---|
| Estudiante | **Edwing Perez** |
| Asignatura | INF-8239 Ciencia de Datos II · Unidad 02 |
| Autor académico del material | Edwin Ramón José Nolasco |
| Repositorio | https://github.com/edvvdev/unidad02-02 |
| PDF breve de la entrega | [`docs/EJERCICIO_04_PDF_BRIEF.pdf`](docs/EJERCICIO_04_PDF_BRIEF.pdf) |

---

## 1. Qué hace este proyecto

Compara, **con la misma partición y la misma semilla**, un **baseline denso**
(`Flatten → Dense(64) → Dense(10)`) frente a una **CNN**
(`Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → GlobalAveragePooling → Dropout → Dense(10)`)
sobre **Fashion-MNIST**, y documenta el resultado con métricas globales y por clase,
curvas de entrenamiento y validación, matriz de confusión, análisis de errores por
clase, ejemplos visuales, comparación de costo y tamaño, **Model Card**, pruebas y
PDF breve.

Todo el código reutilizable vive en `src/inf8239_u02_cv/`; el script
`scripts/train_cv.py` y el notebook usan **las mismas funciones**, de modo que ambos
caminos producen exactamente los mismos artefactos en `reports/`.

## 2. Instalación y ejecución

### Ruta CPU (recomendada para todos)

```bash
uv python install 3.12      # solo si uv no tiene ya el interprete
uv sync --extra cpu
uv run python scripts/check_runtime.py
```

`check_runtime.py` debe mostrar la versión de TensorFlow y la lista de GPU
detectadas. Si la lista de GPU es `[]`, **el laboratorio continúa en CPU**: no hace
falta reinstalar CUDA.

### Ruta GPU (solo WSL2/Linux con NVIDIA ya configurada)

```bash
uv sync --extra gpu
```

### Pipeline completo (reproduce toda la entrega)

```bash
bash scripts/run_all.sh      # Linux / macOS
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1   # Windows
```

El pipeline ejecuta, en este orden: entorno → `check_runtime` → entrenamiento →
estudio de presupuesto → notebook ejecutado → PDF breve → pruebas → verificación de
cifras. No necesita ningún paso manual.

### Pasos por separado

```bash
# 1. Entrenamiento comparativo (baseline denso y CNN, misma partición y semilla)
uv run python scripts/train_cv.py --epochs 8

# 2. Estudio complementario de presupuesto de entrenamiento
uv run python scripts/budget_study.py --epochs 25

# 3. Notebook ejecutado de principio a fin (deja las salidas guardadas)
uv run python scripts/make_notebook.py     # solo si se quiere regenerar el esqueleto
uv run python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 notebooks/01_ejercicio04_cnn_model_card.ipynb

# 4. PDF breve (lee todas las cifras de reports/)
REPO_URL=https://github.com/edvvdev/unidad02-02 uv run python scripts/build_ejercicio04_pdf.py

# 5. Pruebas y linter
uv run pytest -q
uv run ruff check .

# 6. Comprobacion de que las cifras publicadas coinciden con reports/
uv run python scripts/verify_cifras.py
```

### Dependencias de trabajo (notebook y PDF)

`jupyter`, `nbformat`, `ipykernel` y `reportlab` **no** están en `pyproject.toml`
porque el enunciado prohíbe modificarlo. Se instalan en el entorno del proyecto
**después** de `uv sync` (que elimina cualquier paquete que no esté en el lock):

```bash
uv pip install --python .venv/bin/python jupyter nbformat ipykernel reportlab
```

`scripts/run_all.sh` y `scripts/run_all.ps1` ya lo hacen en el paso 2. Las versiones
exactas del entorno base están exportadas en `requirements.txt` y
`requirements-dev.txt` (`uv export --no-emit-project`).

### Nube (Colab)

```bash
%pip install -r requirements-colab.txt
```

El dataset es público y pequeño: se descarga al iniciar. El modelo se carga desde
una **ruta relativa** (`models/best_cnn.keras`). Nunca se sube `.venv`.

## 3. Dataset

| | |
|---|---|
| Nombre | Fashion-MNIST (`tf.keras.datasets.fashion_mnist`) |
| Procedencia | Ficha de Keras (licencia MIT), descarga automática a la caché local |
| Formato | 28×28 píxeles, escala de grises, canal único `(N, 28, 28, 1)` |
| Clases | 10: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot |
| Bloque oficial de entrenamiento | 60,000 imágenes |
| Bloque oficial de prueba | 10,000 imágenes |
| Objetivo (target) | Etiqueta de clase entera entre 0 y 9 |

### Partición reproducible

| Partición | Imágenes | Origen |
|---|---|---|
| Entrenamiento | 54,000 | Primeras 54,000 del bloque de 60.000 |
| Validación | 6,000 | Últimas 6,000 del bloque de 60.000 |
| Prueba | 10,000 | Bloque oficial de prueba, intacto durante el entrenamiento |

* La partición es **fija y determinista** (no barajada): dos ejecuciones obtienen
  exactamente el mismo conjunto de validación.
* La semilla es **42** en Python, NumPy y TensorFlow.
* Las 10.000 imágenes de prueba **no se usan nunca** para entrenar ni para elegir
  modelos: no hay fuga de datos.
* El reparto por clase es casi uniforme (desbalance máximo/mínimo 1.0145 en
  entrenamiento y 1.1405 en validación), por lo que el promedio global
  no está inflado por clases mayoritarias.

La auditoría completa (formas, tipos, rangos, reparto) queda en
`reports/dataset_audit.json`.

## 4. Resultados de referencia

Ejecución de referencia: **8 épocas**, partición 54.000 / 6.000 / 10.000,
semilla 42, CPU (sin GPU detectada), TensorFlow 2.21.0.

| Métrica | Baseline denso | CNN | Diferencia (CNN − denso) |
|---|---|---|---|
| F1 macro | **0.8647** | **0.7892** | -0.0755 |
| Accuracy | 0.8663 | 0.7930 | -0.0733 |
| Parámetros entrenables | 50890 | 19466 | -31424 |
| Tamaño del modelo (`.keras`) | 619.12 KB | 262.17 KB | -356.95 KB |
| Tiempo de entrenamiento | 12.2 s | 68.9 s | +56.7 s |
| Inferencia (mediana de 3) | 0.0512 ms/img | 0.0823 ms/img | +0.0311 ms/img |
| Épocas ejecutadas | 8 | 8 | — |

Mediciones individuales de inferencia (ms por imagen):
denso `0.0512, 0.0446, 0.0622` → mediana `0.0512`;
CNN `0.0946, 0.0786, 0.0823` → mediana `0.0823`.

### Lectura de las curvas

* **Baseline denso:** la pérdida de entrenamiento baja de 0.6009 a 0.3132 y la
  de validación de 0.4510 a 0.3538. Ambas curvas descienden de forma
  sostenida (patrón «el modelo está aprendiendo patrones útiles») y no hay sobreajuste.
* **CNN:** la pérdida de entrenamiento pasa de 1.5808 a 0.6953 y la de validación
  de 1.0882 a 0.5872; la accuracy de validación sigue en 0.7995 en la
  última época. Es el patrón de **subajuste** del manual: el modelo todavía no ha
  convergido con el presupuesto de 8 épocas.

### Estudio complementario de presupuesto

Para separar «la arquitectura no sirve» de «no hay tiempo de entrenamiento», el
proyecto repite la comparación con **25 épocas**, misma partición y misma
semilla (curvas por época calculadas sobre validación, nunca sobre prueba):

| Modelo | F1 macro 8 ep. | F1 macro 25 ep. | Tiempo de entrenamiento 25 ep. |
|---|---|---|---|
| Baseline denso | 0.8647 | 0.8761 | 32.0 s |
| CNN | 0.7892 | 0.8439 | 215.3 s |

El resultado es **compatible con una mejora de la arquitectura de la CNN al entrenar
con más cómputo**, pero con **una sola semilla y sin intervalos de confianza** no
permite separar de forma concluyente el efecto del presupuesto del efecto de la
arquitectura: por eso la decisión se toma sobre las medidas de la ejecución de
referencia (8 épocas) y el estudio ampliado se reporta como evidencia
complementaria.

### Errores por clase

Las clases que ambos modelos pierden con más frecuencia son **Shirt**
(recall 0.372 en la CNN) y **Shirt** (recall 0.597 en el
baseline denso): las prendas de torso y el calzado se distinguen por un cuello, una
manga o una suela de pocos píxeles. El detalle por clase está en
`reports/cv_errors_by_class.csv` y `reports/errors_by_class.png`, y 16 ejemplos
elegidos **al azar con semilla fija** en `reports/cnn_errors.png`.

### Decisión técnica

Se entrega el ****baseline denso****. Con el presupuesto que fija el manual el baseline
denso gana en F1 macro, entrena 5.6 veces más rápido e infiere
1.6 veces más rápido; la **única** dimensión en la que gana la CNN es la
huella (19466 parámetros y 262.17 KB frente a 50890 y 619.12),
de modo que la CNN es la opción preferente si la restricción dominante es la memoria
o el tamaño del artefacto desplegado. El razonamiento completo, calculado a partir de
las métricas de la ejecución, está en `reports/decision_tecnica.txt`.

> Esta decisión **no** se ha fijado de antemano: el bloque del notebook que la
> escribe calcula el modelo entregado a partir de las métricas de la ejecución.

## 5. Límites

* Fashion-MNIST es un **benchmark didáctico** de siluetas de moda en 28×28. Un buen
  resultado **no** demuestra preparación para imágenes médicas, OCR, vigilancia ni
  ningún otro dominio: antes de usar el modelo fuera de esta tarea hay que
  reentrenarlo y reevaluarlo con datos del dominio.
* Los datos son de Shopping en línea, en escala de grises; no incluyen color, textura
  real, contexto ni personas, y el modelo hereda esas limitaciones.
* La ejecución de referencia se hizo **en CPU (sin GPU detectada)**. Los tiempos de
  entrenamiento e inferencia son válidos para esta máquina y **no** deben
  generalizarse a un equipo con GPU.
* El presupuesto de 8 épocas del manual deja la CNN en subajuste; con más
  épocas mejora, pero no llega al baseline denso en este dataset.

## 6. Estructura del proyecto

```
.
├── AI_USE_DECLARATION.md          # declaración de uso de IA (prompts, verificaciones, correcciones)
├── CAMBIOS_AL_BASE.md             # cada cambio respecto del proyecto base, su motivo y su verificación
├── MODEL_CARD.md                  # ficha del modelo (uso previsto, datos, métricas, límites, supervisión)
├── README.md                      # este documento
├── requirements.txt               # export de `uv export --no-emit-project` (dependencias base + cpu)
├── requirements-dev.txt           # export de `uv export --no-emit-project --only-dev` (pytest, ruff)
├── requirements-colab.txt         # ruta de Colab (archivo del base, ampliado con comentarios)
├── docs/
│   └── EJERCICIO_04_PDF_BRIEF.pdf # PDF breve de la entrega (6 páginas)
├── models/
│   ├── best_dense.keras           # baseline denso entrenado (modelo entregado)
│   ├── best_cnn.keras             # CNN entrenada (alternativa compacta)
│   └── .gitkeep
├── notebooks/
│   └── 01_ejercicio04_cnn_model_card.ipynb   # notebook EJECUTADO, 0 celdas con error
├── reports/
│   ├── cv_metrics.json            # métricas globales, por clase e historiales
│   ├── cv_per_class.csv           # precisión, recall, F1 y soporte por clase y modelo
│   ├── cv_errors_by_class.csv     # errores, falsos positivos y recall por clase y modelo
│   ├── cnn_error_samples.csv      # ficha de los 16 ejemplos de error elegidos
│   ├── dataset_audit.json         # auditoría del dataset y reparto por clase
│   ├── budget_study.json          # estudio complementario de presupuesto
│   ├── budget_study.png           # curvas por época y efecto del presupuesto
│   ├── decision_tecnica.txt       # decisión técnica calculada sobre las métricas
│   ├── training_curves.png        # curvas de entrenamiento y validación de ambos modelos
│   ├── confusion_cnn.png          # matriz de confusión de la CNN
│   ├── confusion_dense.png        # matriz de confusión del baseline denso
│   ├── errors_by_class.png        # recall y errores por clase
│   └── cnn_errors.png             # 16 ejemplos de error (semilla fija)
├── scripts/
│   ├── check_runtime.py           # versión de TensorFlow y GPU detectada (base)
│   ├── train_cv.py                # entrenamiento comparativo + artefactos
│   ├── budget_study.py            # estudio complementario de presupuesto
│   ├── make_notebook.py           # generador del notebook
│   ├── build_ejercicio04_pdf.py   # generador del PDF breve
│   ├── verify_cifras.py           # comprueba que lo publicado coincide con reports/
│   ├── run_all.sh                 # pipeline completo (Linux/macOS)
│   └── run_all.ps1                # pipeline completo (Windows PowerShell)
├── src/inf8239_u02_cv/
│   ├── data.py                    # normalización, partición, auditoría, nombres de clase
│   ├── models.py                  # build_dense y build_cnn (base, sin cambios)
│   ├── experiment.py              # experimento reproducible: entrenamiento, métricas, artefactos
│   ├── diagnostico.py             # patrón de las curvas (compartido por notebook, PDF y verificador)
│   └── figures.py                 # figuras: curvas, confusión, errores por clase, rejilla de errores
└── tests/                         # pruebas del contrato de datos, modelos, métricas y notebook
```

## 7. Pruebas

```bash
uv run pytest -q        # 29 pruebas en verde
uv run ruff check .     # sin incidencias
```

Las pruebas cubren el contrato que pide el manual: forma y rango de las imágenes,
número de clases, dimensión de la salida del modelo y probabilidades finitas que
suman aproximadamente 1; además, la reproducibilidad de la partición, la coherencia
entre `cv_metrics.json`, `cv_per_class.csv` y `cv_errors_by_class.csv`, la mediana de
la inferencia y el estado del notebook (0 celdas con error, todas ejecutadas).

## 8. Referencias

* Manual de la Unidad 02 de INF-8239, laboratorio `U02.LAB07` (prof. Edwin Ramón José Nolasco).
* Ficha de Keras de `fashion_mnist` (licencia MIT): <https://keras.io/api/datasets/fashion_mnist/>
* Documentación de `tf.keras.datasets.fashion_mnist`:
  <https://www.tensorflow.org/datasets/catalog/fashion_mnist>
