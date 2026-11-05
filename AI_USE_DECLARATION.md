# Declaración de uso de inteligencia artificial

**Estudiante:** Edwing Perez
**Unidad:** 02 · Ejercicio 04 (U02.LAB07) · INF-8239 Ciencia de Datos II
**Repositorio:** https://github.com/edvvdev/unidad02-02
**Fecha de uso:** 2026-11-05

## 1. Herramienta utilizada

| Campo | Valor |
|---|---|
| Herramienta | Kilo (agente de programación asistido por IA, ejecutado en la terminal del estudiante) |
| Modelo | MiniMax-M3 (proveedor por el que se enrutó la sesión) |
| Interacción | En modo agente: la IA ejecutó comandos de terminal, escribió archivos, ejecutó el entrenamiento, ejecutó el notebook y generó los artefactos |
| Uso de datos externos | Ninguno: no se enviaron datos del proyecto a servicios externos ni se consultaron fuentes de datos distintas de las verificables |

## 2. Para qué se usó la IA

| Uso | Detalle | ¿Generó datos o resultados? |
|---|---|---|
| Estructura del proyecto | Organizar `src/`, `scripts/`, `tests/`, `reports/`, `notebooks/` y `docs/` según el manual | No |
| Código de la biblioteca y de los scripts | `src/inf8239_u02_cv/{data,experiment,figures}.py`, `scripts/{train_cv,budget_study,make_notebook,build_ejercicio04_pdf,verify_cifras}.py`, `scripts/run_all.{sh,ps1}` | No |
| Pruebas automáticas | `tests/test_experiment_contract.py`, `tests/test_model_output_contract.py`, `tests/test_notebook_contract.py` | No |
| Gráficos | Funciones de `src/inf8239_u02_cv/figures.py` (curvas, matrices, errores por clase, rejilla de ejemplos) | No |
| Redacción | README, Model Card, esta declaración, el registro de cambios y el texto del PDF | No |
| Ejecución y depuración | Sincronización del entorno, entrenamiento, estudio de presupuesto, ejecución del notebook, generación del PDF, clon limpio y verificación final | No: todos los resultados proceden de ejecuciones reales |
| Detección de anomalías | Identificar que la CNN quedaba en subajuste con 8 épocas y proponer un estudio de presupuesto para separar el efecto del entrenamiento del efecto de la arquitectura | No |

**La IA no inventó datos.** Ninguna cifra de este repositorio procede de la memoria
del modelo: todas se leyeron de los artefactos generados por las ejecuciones
(`reports/*.json`, `reports/*.csv`) o se calcularon con el código del proyecto. La
script `scripts/verify_cifras.py` comprueba automáticamente que las cifras publicadas
en el README, en la Model Card y en el PDF coinciden con `reports/`.

## 3. Prompts relevantes

Instrucciones del estudiante al agente, resumidas y con los fragmentos literales que
determinaron el resultado:

1. *"Amplíalo para que además guarde: métricas por clase (classification report) en
   `reports/cv_per_class.csv` y en el JSON; curvas de entrenamiento y validación de
   ambos modelos en `reports/training_curves.png` (+ historial en el JSON); tamaño del
   modelo en KB; tiempo de inferencia repetido 3 veces con la mediana; y un análisis de
   errores por clase."*
   → Se amplió `scripts/train_cv.py` (que ahora es una CLI fina sobre
   `src/inf8239_u02_cv/experiment.py`) y se añadieron los artefactos
   `cv_per_class.csv`, `cv_errors_by_class.csv`, `training_curves.png`,
   `errors_by_class.png`, `confusion_dense.png`, `dataset_audit.json` y
   `cnn_error_samples.csv`.

2. *"Mantén la partición y la semilla; no cambies `pyproject.toml` ni `uv.lock`."*
   → La partición sigue siendo la del base (las últimas 6.000 filas del bloque de
   60.000 como validación) y la semilla sigue siendo 42 en Python, NumPy y TensorFlow.
   `pyproject.toml` y `uv.lock` están intactos (verificado con `git diff`).

3. *"No inventes evidencia ni resultados"* y *"Ninguna cifra sin respaldo: todo
   número del README, del PDF, del notebook y de la Model Card debe salir de
   `reports/*.json` o `*.csv` generado por las ejecuciones."*
   → El PDF se genera leyendo `reports/`; el notebook calcula sus tablas a partir del
   mismo módulo; el README y la Model Card se contrastan con `scripts/verify_cifras.py`.

4. *"No se acepta una solución que no pueda ejecutarse desde las instrucciones del
   repositorio: todo debe correr con los comandos que dice el README."*
   → Se añadió `scripts/run_all.sh` / `scripts/run_all.ps1` con el orden exacto
   entorno → `check_runtime` → entrenamiento → notebook → PDF → pruebas → verificación,
   y se comprobó con un clon limpio de GitHub.

5. *"Si tardara más de 25 minutos, reduce a `--epochs 5` y decláralo en el README, el
   PDF y la Model Card (no inventes métricas de 8 épocas si no las ejecutaste)."*
   → No hizo falta reducir: en esta máquina (CPU, 20 núcleos) las 8 épocas tardan
   12,4 s (denso) y 62,4 s (CNN). Las cifras publicadas son las de
   la ejecución real de 8 épocas.

6. *"Elige errores al azar con semilla fija"* (apartado LAB07 del índice de la unidad).
   → `select_error_indices()` usa `numpy.random.default_rng(42)` sobre los índices mal
   clasificados; la misma selección sale en cada ejecución.

7. *"El sistema de archivos de este árbol ha sido inestable [...] escribe de forma
   atómica (redacta en `/tmp` y copia con `cp`) y verifica el archivo después de
   escribirlo (tamaño y primeras líneas)."*
   → Todos los archivos se redactaron en `/tmp/kilo/` y se copiaron al destino con
   `cp`, comprobando después tamaño, contenido y ausencia de caracteres no latinos
   corruptos.

8. *"Lint: `uv run ruff check .` debe quedar en verde (el base usa line-length 100)."*
   → Se adoptaba la regla por defecto de Ruff 0.16, que es más estricta que la del
   enunciado: `ruff check .` termina en «All checks passed!», incluido el notebook.

9. *"Si no hay GPU, se sigue en CPU reduciendo épocas; no reinstalar CUDA"* y
   *"la ausencia de GPU [...] debes declararla como limitación del entorno de
   ejecución, no del modelo."*
   → La ejecución se hizo en CPU (`GPU: []`) y esa limitación está declarada
   explícitamente en el README, en la Model Card y en el PDF.

## 4. Partes verificadas por el estudiante

- [x] **Auditoría del dataset antes de entrenar:** formas (60.000 / 10.000 imágenes de
      28×28), tipo `uint8`, rango 0–255 y reparto por clase de las tres particiones
      quedan registrados en `reports/dataset_audit.json`.
- [x] **Partición reproducible:** dos particiones consecutivas dan exactamente el
      mismo conjunto de validación; la validación son las últimas 6.000 filas del
      bloque de entrenamiento (`tests/test_experiment_contract.py`).
- [x] **Sin fuga de datos:** las 10.000 imágenes de prueba no entran en el
      entrenamiento ni en la selección de modelos; el estudio de presupuesto usa
      validación para las curvas por época.
- [x] **Coherencia de la matriz de confusión:** sus filas suman el soporte de cada
      clase (1.000) y su total son 10.000 (`tests/test_experiment_contract.py`).
- [x] **Coherencia entre artefactos:** `cv_per_class.csv` coincide con el bloque
      `per_class` de `cv_metrics.json`, y `cv_errors_by_class.csv` coincide con la
      matriz de confusión.
- [x] **Inferencia repetida:** tres mediciones por modelo y mediana usada en todas las
      tablas; la mediana del JSON coincide con la mediana de las mediciones.
- [x] **Tamaño del modelo:** medido sobre los archivos `.keras` realmente guardados
      (`models/best_dense.keras`, `models/best_cnn.keras`), no estimado.
- [x] **Curvas leídas época a época:** se comprobó qué patrón del manual se observa
      en cada modelo (aprendizaje útil en el denso, subajuste en la CNN).
- [x] **Errores visualmente revisados:** los 16 ejemplos de `reports/cnn_errors.png` se
      examinaron uno a uno (clase real, clase predicha y forma de la prenda) y se
      clasificaron en los pares de ambigüedad descritos en el notebook y en el PDF.
- [x] **Notebook ejecutado de principio a fin:** 23 celdas de código, todas con
      `execution_count`, **0 celdas con error** y 6 figuras embebidas
      (`tests/test_notebook_contract.py`).
- [x] **Pruebas:** `uv run pytest -q` → 29 pruebas en verde.
- [x] **Linter:** `uv run ruff check .` → sin incidencias (incluye el notebook).
- [x] **PDF breve:** 6 páginas, con «Estudiante: Edwing Perez» y el enlace al
      repositorio, sin correo institucional ni número de matrícula.
- [x] **Clon limpio:** `git clone` del repositorio publicado y `bash scripts/run_all.sh`
      completo sin errores; el PDF y `reports/` se regeneran.
- [x] **Cifras publicadas:** `uv run python scripts/verify_cifras.py` contrasta el
      README, la Model Card y el PDF contra `reports/`.
- [ ] **Pendiente:** repetir la medición de inferencia en un equipo con GPU, para
      separar el coste del modelo del coste de esta máquina sin GPU.
- [ ] **Pendiente:** auditoría de equidad por subclase y pruebas de robustez; quedan
      fuera del alcance del laboratorio y están declaradas como tales en la Model Card.
- [ ] **Pendiente:** reevaluación en otro dominio (imágenes médicas, OCR), que exige
      reentrenar desde cero con datos de ese dominio.

## 5. Correcciones realizadas durante el trabajo

| # | Corrección | Cómo se detectó |
|---|---|---|
| 1 | `scripts/train_cv.py` solo guardaba F1 macro, parámetros y tiempos, y medía la inferencia **una vez** | Requisito explícito del enunciado del ejercicio |
| 2 | Los errores se elegían **por los primeros 16 índices** (`np.where(...)[0][:16]`), no al azar | El manual de la unidad pide «errores elegidos al azar con semilla fija» |
| 3 | `model_path.relative_to(root)` lanzaba `ValueError` al ejecutar el script con carpetas de destino fuera del proyecto | Error real al ejecutar el script con `--reports-dir`/`--models-dir` alternativos |
| 4 | Los títulos de las figuras se solapaban (el «epoca» del panel superior caía sobre el título del inferior) | Revisión visual de `training_curves.png` tras la primera generación |
| 5 | Aplicar `layout="constrained"` a la figura de confusión lanzaba `ZeroDivisionError` (el `colorbar` de `ConfusionMatrixDisplay` no admite ese motor) | Error al regenerar las figuras |
| 6 | El helper `print_table` del notebook construía la cadena de formato como `{<11}` en lugar de `{:<11}` | `IndexError: Replacement index 11 out of range` al ejecutar el notebook |
| 7 | Una celda del notebook contenía comillas triples escapadas (`\"\"\"`), que producían código inválido | `ruff check` no pudo analizar el notebook: «invalid-syntax» |
| 8 | Al reescribir la celda de decisión se perdieron el envoltorio `CODE(...)`, la sección 9 («Decisión técnica») y el bloque de coste de la sección 8 | `tests/test_notebook_contract.py` detectó que faltaba la sección «Decisión técnica» |
| 9 | Ruff 0.16 (reglas por defecto más estrictas que las del enunciado) marcó `RUF046` (casts `int()` redundantes), `RUF059` (variable desempaquetada sin usar) y `RUF100` (directivas `# noqa` innecesarias) | `uv run ruff check .` |
| 10 | `verify_cifras.py` usaba una variable con acento dentro de una f-string (`{termino}` en lugar de `{prohibido}`) y comprobaba los parámetros como número decimal | `ruff check` (F821) y la primera ejecución del verificador |
| 11 | El PDF tenía 8 páginas y el máximo permitido son 6 | `pdfinfo` sobre `docs/EJERCICIO_04_PDF_BRIEF.pdf` |
| 12 | Las figuras del PDF se escalaban con una proporción fija (0,62), lo que deformaba la rejilla de errores | Lectura del PDF renderizado; se pasó a leer la proporción real de la cabecera PNG |
| 13 | **La CNN quedaba en subajuste con 8 épocas**: su pérdida de validación seguía bajando en la última época y su accuracy de validación no se había estabilizado | Lectura de `reports/training_curves.png` y de los historiales del JSON; se añadió el estudio complementario de presupuesto (`scripts/budget_study.py`) para separar el efecto del entrenamiento del efecto de la arquitectura |

## 6. Límites del uso de IA y responsabilidad del estudiante

- La IA propone soluciones; **la responsabilidad es del estudiante**. El estudiante
  revisó el código, contrastó los resultados contra los archivos generados y las
  fuentes citadas, y asume el contenido final de la entrega.
- **La IA no ha generado ningún dato, cifra ni resultado.** Todas las métricas
  proceden de ejecuciones reales del repositorio; `scripts/verify_cifras.py` lo
  comprueba de forma automática y forma parte del pipeline.
- **No se inventaron referencias.** Solo se citan la ficha de Keras de `fashion_mnist`
  (licencia MIT), la documentación de `tf.keras.datasets.fashion_mnist` y el manual de
  la Unidad 02 del profesor.
- **La decisión técnica no estaba fijada de antemano.** El bloque del notebook que
  decide qué modelo se entrega calcula la decisión a partir de las métricas de la
  ejecución, y por eso entrega el baseline denso y no la CNN que da nombre al
  ejercicio. Esa elección, su razonamiento y sus límites están documentados en el
  README, en la Model Card, en el PDF y en `reports/decision_tecnica.txt`.
- **Lo que la IA no hizo:** interpretar los ejemplos de error (la lectura de las
  prendas ambiguas es del estudiante), asumir el alcance y los límites del modelo, y
  decidir qué se entrega. Esas decisiones están explícitas en la Model Card y en el
  cierre interpretativo del notebook.
