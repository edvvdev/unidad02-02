# Cambios respecto al proyecto base

Base de referencia: `INF8239_U02_CV_Proyecto_Base_uv.zip` (MD5
`648f740fd06d10e51c79d7a87a8254b9`), extraído en `Unidad_2/INF8239_U02_CV/` el
2026-11-05. El primer commit del repositorio (`chore: initialize isolated computer
vision project`) contiene **el base tal cual**, sin ninguna modificación: los cambios
de este documento están en los commits posteriores.

## 0. Lo que NO se tocó

| Archivo | Estado | Motivo |
|---|---|---|
| `pyproject.toml` | **Sin modificar** (mismas dependencias, mismos extras `cpu`/`gpu`, mismo grupo `dev`) | El enunciado lo prohíbe expresamente |
| `uv.lock` | **Sin modificar** (verificado con `git diff uv.lock`, sin diferencias) | El enunciado lo prohíbe expresamente y el manual pide versionarlo |
| `.python-version` | Sin modificar (`3.12`) | — |
| `src/inf8239_u02_cv/models.py` | Sin modificar: `build_dense` y `build_cnn` son exactamente los del base | Los dos modelos que se comparan deben ser los del enunciado |
| `scripts/check_runtime.py` | Sin modificar | El manual indica ese comando literal |
| `tests/test_data.py` | Sin modificar | — |
| `tests/test_model_contract.py` | Sin modificar | — |
| `requirements-colab.txt` | Sin modificar | Ya cubría la ruta de Colab que pide el manual |

Las dependencias que hacen falta para el notebook y el PDF (`jupyter`, `nbformat`,
`ipykernel`, `reportlab`) **no** se añadieron a `pyproject.toml`: se instalan con
`uv pip install --python .venv/bin/python ...` **después** de `uv sync`, porque
`uv sync` elimina del entorno todo lo que no está en el lock y dejaría el entorno
inreproducible. `scripts/run_all.sh` y `scripts/run_all.ps1` ya lo hacen en el paso 2.

## 1. Archivos nuevos

| Archivo | Qué hace |
|---|---|
| `src/inf8239_u02_cv/experiment.py` | Experimento reproducible: carga del dataset, partición, entrenamiento de los dos modelos, métricas globales y por clase, medición repetida de inferencia, análisis de errores por clase y escritura de todos los artefactos. Lo usan **el script y el notebook**, de modo que ambos caminos dan el mismo resultado. |
| `src/inf8239_u02_cv/figures.py` | Todas las figuras (curvas de entrenamiento y validación, matrices de confusión, recall y errores por clase, rejilla de 16 ejemplos de error) y la selección **aleatoria con semilla fija** de los ejemplos de error. |
| `scripts/budget_study.py` | Estudio **complementario** de presupuesto de entrenamiento (25 épocas, misma partición y misma semilla) con F1 macro de validación por época. Responde a la pregunta que dejan abiertas las curvas del experimento principal. |
| `scripts/make_notebook.py` | Genera el esqueleto de `notebooks/01_ejercicio04_cnn_model_card.ipynb` a partir de celdas escritas en el propio script: el notebook es reproducible y no depende de copiar y pegar a mano. |
| `scripts/build_ejercicio04_pdf.py` | Genera `docs/EJERCICIO_04_PDF_BRIEF.pdf` (6 páginas) leyendo **todas** las cifras de `reports/`. El enlace al repositorio se toma de `REPO_URL`. |
| `src/inf8239_u02_cv/diagnostico.py` | Diagnóstico **compartido** del patrón de las curvas (aprendizaje útil / sobreajuste / subajuste). Lo usan el notebook, el generador del PDF y `scripts/verify_cifras.py`, de modo que los tres leen exactamente el mismo criterio y no pueden divergir entre sí. |
| `scripts/verify_cifras.py` | Comprueba automáticamente que las cifras publicadas en el README, en la Model Card y en el PDF coinciden con `reports/`, que el PDF tiene entre 4 y 6 páginas, que el notebook no tiene celdas con error, **que el patrón de las curvas es el que corresponde** (recalculado con `diagnostico.py`) y **que el PDF no publica frases incorrectas o sobreafirmadas**. |
| `scripts/run_all.sh` | Pipeline completo en Linux/macOS: entorno → `check_runtime` → entrenamiento → estudio de presupuesto → notebook → PDF → pruebas → verificación de cifras. |
| `scripts/run_all.ps1` | Equivalente para Windows PowerShell. |
| `notebooks/01_ejercicio04_cnn_model_card.ipynb` | Notebook **ejecutado** (23 celdas de código, 0 errores, 6 figuras): auditoría → partición → entrenamiento → curvas → métricas globales y por clase → matriz de confusión → errores por clase y ejemplos visuales → costo y tamaño → decisión técnica → cierre interpretativo. |
| `tests/test_experiment_contract.py` | Contratos del experimento: partición determinista y sin fuga, auditoría, coherencia entre `cv_metrics.json`, `cv_per_class.csv` y `cv_errors_by_class.csv`, mediana de la inferencia, selección aleatoria reproducible de errores. |
| `tests/test_model_output_contract.py` | Contrato de salida de **los dos** modelos: dimensión de la salida, probabilidades finitas que suman 1 y estructura de las arquitecturas. |
| `tests/test_notebook_contract.py` | El notebook está ejecutado, sin celdas con error y con todas las secciones exigidas. |
| `docs/EJERCICIO_04_PDF_BRIEF.pdf` | PDF breve de la entrega (6 páginas). |
| `requirements.txt`, `requirements-dev.txt` | Exportaciones de `uv export --no-emit-project` (no editadas a mano): `requirements.txt` con el extra `cpu` y `requirements-dev.txt` con el grupo `dev` (pytest y ruff). |
| `AI_USE_DECLARATION.md` | Declaración de uso de IA: herramienta, finalidad, prompts relevantes, verificaciones con casillas, correcciones realizadas y límites. |
| `CAMBIOS_AL_BASE.md` | Este documento. |

## 2. Archivos ampliados

| Archivo | Cambio | Motivo |
|---|---|---|
| `src/inf8239_u02_cv/data.py` | Se conservaron `normalize_images` y `validate_images` **sin cambios de comportamiento** y se añadieron `CLASS_NAMES` (los 10 nombres oficiales de clase), `split_train_validation` (la partición fija del base, documentada), `class_distribution`, `class_shares` y `audit_dataset` (formas, tipos, rangos y reparto por clase antes de entrenar). | El base no podía auditar el dataset antes de entrenar ni nombrar las clases en los informes. |
| `scripts/train_cv.py` | Ahora es una CLI fina sobre `src/inf8239_u02_cv/experiment.py`. Se conservan `--epochs` y las salidas del base (`reports/cv_metrics.json`, `reports/confusion_cnn.png`, `reports/cnn_errors.png`, `models/best_cnn.keras`) y se añaden: métricas por clase en el JSON y en `reports/cv_per_class.csv`, historiales de entrenamiento en el JSON, curvas de ambos modelos en `reports/training_curves.png`, tamaño del modelo en KB, **mediana de tres mediciones** de inferencia, análisis de errores por clase (`reports/cv_errors_by_class.csv` y `reports/errors_by_class.png`), auditoría del dataset (`reports/dataset_audit.json`), matriz del baseline denso (`reports/confusion_dense.png`) y ficha de los ejemplos de error (`reports/cnn_error_samples.csv`). Se añaden los argumentos `--seed`, `--batch-size`, `--repeats`, `--reports-dir`, `--models-dir` y `--quiet`. | El manual exige métricas por clase, curvas, tamaño, tiempo de inferencia repetido y errores por clase; el base solo guardaba tres números y una medición única de inferencia. |
| `.gitignore` | Se añadieron `.ruff_cache/`, `.ipynb_checkpoints/` y `*.egg-info/`, y se **quitaron** las reglas que excluían `reports/*.png`, `reports/*.json` y `models/*.keras`. | La evidencia (figuras, métricas, CSV) y el modelo entrenado tienen que viajar con el repositorio: el propio manual indica `git add src scripts tests reports MODEL_CARD.md uv.lock`. `models/budget_*.keras` (modelos del estudio complementario) sí quedan excluidos. |
| `README.md` | Sustituido por la guía reproducible completa: instalación (CPU y GPU), pipeline, pasos por separado, dependencias de trabajo, Colab, dataset y target, partición, resultados de referencia, lectura de curvas, errores por clase, decisión técnica, límites, estructura, pruebas y referencias. | El base era una guía de 12 líneas y la rúbrica pide README con instalación, ejecución, dataset, target, resultados, límites y estructura. |
| `MODEL_CARD.md` | Completado con las cifras reales: modelo y versión, uso previsto, usos fuera de alcance, dataset y particiones, preprocesamiento, métricas globales y por clase (tabla de las 10 clases para los dos modelos), comparación de costo, limitaciones y riesgos, supervisión y monitoreo y referencias. | El base solo tenía las cabeceras vacías. |
| `requirements-colab.txt` | **Sin modificar** (se conserva la lista del base). | La ruta de Colab del manual ya queda cubierta. |

## 3. Decisiones que cambian el resultado del base

| Decisión | Base | Ahora | Por qué |
|---|---|---|---|
| Archivos guardados | Solo `models/best_cnn.keras` | `models/best_dense.keras` **y** `models/best_cnn.keras` | Para poder comparar el **tamaño del modelo** de los dos, que es una de las dimensiones que pide el manual |
| Ejemplos de error | Los 16 primeros índices mal clasificados | 16 ejemplos **elegidos al azar con semilla fija (42)** | El manual de la unidad pide «errores elegidos al azar con semilla fija»; además es más representativo que tomar los primeros |
| Métricas por clase | No se guardaban | `reports/cv_per_class.csv` + bloque `per_class` y `classification_report` en el JSON | El manual exige F1 macro **y métricas por clase**, y avisa de que el promedio puede ocultar una clase débil |
| Tiempo de inferencia | Una única medición | **Mediana de tres mediciones** (las tres quedan en el JSON) | El manual pide «mida el tiempo de inferencia sobre el mismo lote y repita la medición» |
| Curvas | No se guardaban | `reports/training_curves.png` con los dos modelos | El manual pide interpretar las curvas con los cuatro patrones |
| Análisis de errores por clase | No existía | `reports/cv_errors_by_class.csv` + `reports/errors_by_class.png` | Rúbrica: «errores por clase» y «qué clases se pierden» |
| Modelo entregado | El base solo guardaba la CNN, sin decidir nada | **Baseline denso** (decidido por los datos, no fijado de antemano) | Con 8 épocas el denso alcanza F1 macro 0.8647 frente a 0.7892 de la CNN, entrena 5.0 veces más rápido e infiere 2.4 veces más rápido; la CNN solo gana en huella. La decisión se calcula en el notebook y se escribe en `reports/decision_tecnica.txt` |
|Experimento adicional | Ninguno | Estudio de presupuesto con 25 épocas (`scripts/budget_study.py`) | Las curvas del experimento principal mostraron que la CNN queda en **subajuste**; el estudio separa el efecto del presupuesto de entrenamiento del efecto de la arquitectura. Es **complementario**: la comparación oficial sigue siendo la de 8 épocas y se reporta sin modificar |

## 4. Segunda ronda de correcciones (2026-11-05)

Correcciones sobre la entrega ya publicada, todas verificadas ejecutando el
código del repositorio.

| # | Defecto | Corrección | Evidencia |
|---|---|---|---|
| 1 | El PDF publicaba un **diagnóstico de curvas invertido**: la condición `val_loss[-1] > min(val_loss)` detecta **sobreajuste**, pero se usaba para decidir el subajuste, de modo que el informe afirmaba «la pérdida de validación deja de mejorar, con lo que hay que revisar la regularización» mientras la curva embebida en esa misma página, el pie de figura, el notebook y la Model Card describían lo contrario | El diagnóstico se centraliza en `src/inf8239_u02_cv/diagnostico.py`, que detecta el subajuste como pretende el manual (pérdida de validación todavía alta **y en descenso**, sin estabilizar). Lo usan el notebook, el PDF y `verify_cifras.py` | `reports/cv_metrics.json`: la val_loss de la CNN baja de 1,0882 a 0,5872 y la accuracy de validación sube de 0,6737 a 0,7995; el patrón recalculado es `subajuste` |
| 2 | El PDF no incluía dos de las seis figuras: `reports/confusion_dense.png` (la evidencia que sustenta la decisión central) ni `reports/budget_study.png` | Ambas se añaden: las dos matrices de confusión van en paralelo en la sección de evidencia visual y el estudio de presupuesto ocupa su propia sección con su figura | `pdftotext` del PDF regenerado y revisión de las páginas renderizadas |
| 3 | Sobreafirmación en el PDF §5: «la diferencia no es solo de presupuesto: la arquitectura mínima del base … queda por debajo de una capa densa de 784 entradas en este dataset». Con **una sola semilla y sin intervalos** no queda demostrado | Reescrito en el sentido prudente: «el resultado es **compatible con una mejora de la arquitectura** de la CNN al entrenar con más cómputo, pero con una sola semilla y sin intervalos de confianza **no permite separar de forma concluyente** el efecto del presupuesto del efecto de la arquitectura». El README y la Model Card se alinean con esa frase | `reports/decision_tecnica.txt`, README y `MODEL_CARD.md` |
| 4 | Erratas del README: «deShopping», «a escala de grises», «Ambas Curve descienden» y `reports/budget_study.png` ausente de la estructura | Corregidas y el archivo añadido al árbol de `reports/` | `README.md` |
| 5 | Las leyendas tapaban las barras: en `reports/errors_by_class.png` la leyenda se superponía al extremo de la barra de **Sandal** (recall 0,962) y en `reports/budget_study.png` rozaba la barra de la CNN | Las leyendas se salen del área de las barras (`loc="upper center"` con margen superior en las barras y `loc="lower right"` en el panel de curvas, donde las series suben y la esquina queda libre) | Revisión visual de las dos figuras regeneradas |
| 6 | `scripts/verify_cifras.py` solo validaba cifras **numéricas**, por eso el defecto 1 pudo llegar al PDF | Se amplía: recalcula el patrón de las curvas contra `reports/cv_metrics.json`, comprueba que la frase de subajuste está en el PDF, el README y la Model Card, y rechaza explícitamente la frase invertida y la sobreafirmación. El verificador forma parte de `scripts/run_all.sh` | `uv run python scripts/verify_cifras.py` |
| 7 | `AI_USE_DECLARATION.md` marcaba las casillas de verificación sin el comando ni la salida real | Cada comprobación va con su comando y su salida real | `AI_USE_DECLARATION.md` |

## 5. Elementos no incluidos y por qué

| Elemento | Motivo |
|---|---|
| `.venv/` | Excluido por el `.gitignore`; es el entorno local de trabajo |
| `data/` | El dataset no se copia al repositorio: lo descarga Keras en su caché local (`tf.keras.datasets.fashion_mnist`), es público y pequeño, y el manual lo permite en la nube |
| `models/budget_dense.keras`, `models/budget_cnn.keras` | Pertenecen al estudio complementario de presupuesto, no al modelo entregado; se regeneran con `scripts/budget_study.py` |
| `reports/*.png` de versiones intermedias | Solo se versiona la última generación, que es la que documenta el README, la Model Card y el PDF |
| Cualquier `.ipynb_checkpoints/` | Se añadió al `.gitignore`; es caché del editor de notebooks |
