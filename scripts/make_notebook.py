"""Genera ``notebooks/01_ejercicio04_cnn_model_card.ipynb`` a partir de celdas.

El notebook se versiona **con sus salidas** (se ejecuta con
``jupyter nbconvert --execute --inplace``). Este generador permite reconstruir
el esqueleto del notebook sin copiar y pegar a mano: las celdas viven aqui y
el contenido ejecutable es siempre el mismo.

Uso:  uv run python scripts/make_notebook.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "01_ejercicio04_cnn_model_card.ipynb"

MD = nbf.v4.new_markdown_cell
CODE = nbf.v4.new_code_cell


def cells() -> list:
    return [
        MD(
            """# Ejercicio 04 · Unidad 02 · LAB07
## Clasificador visual con CNN y Model Card

**Estudiante:** Edwing Perez
**Asignatura:** INF-8239 Ciencia de Datos II · Unidad 02
**Repositorio:** https://github.com/edvvdev/unidad02-02
**Manual de la unidad:** U02.LAB07 · Visión computacional: CNN reproducible con Fashion-MNIST

Este notebook recorre el laboratorio completo: **auditoría del dataset** y su reparto
de clases, **partición reproducible**, **curvas de entrenamiento y validación** de los
dos modelos (baseline denso y CNN), **métricas globales y por clase**, **matriz de
confusión**, **errores por clase y ejemplos visuales**, **tabla de costo y tamaño**
y, por último, la **decisión técnica**.

> Todo el código reutilizable vive en `src/inf8239_u02_cv/`. El notebook usa las
> mismas funciones que `scripts/train_cv.py`, de modo que ambos caminos producen
> los mismos artefactos en `reports/`."""
        ),
        CODE(
            """import json
import os
import platform
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import Image, display

ROOT = Path.cwd()
if not (ROOT / "src").exists():  # por si nbconvert se ejecuta desde otra carpeta
    ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))

from inf8239_u02_cv.data import (
    CLASS_NAMES,
    DEFAULT_SEED,
    DEFAULT_VALIDATION_SIZE,
    NUM_CLASSES,
    audit_dataset,
    class_distribution,
    normalize_images,
    split_train_validation,
)
from inf8239_u02_cv.diagnostico import diagnostico_curvas
from inf8239_u02_cv.experiment import (
    run_budget_study,
    run_experiment,
    set_global_seed,
)
from inf8239_u02_cv.figures import plot_budget_study
from inf8239_u02_cv.models import build_cnn, build_dense

SEED = DEFAULT_SEED
EPOCHS = 8                 # presupuesto que fija el manual (U02.LAB07)
BUDGET_EPOCHS = 25         # estudio complementario de presupuesto
BATCH_SIZE = 128
REPEATS = 3
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
REPORTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def print_table(headers, rows, aligns=None):
    # Imprime una tabla de texto alineada (sin dependencias externas).
    aligns = aligns or ["<"] * len(headers)
    widths = [max(len(str(headers[i])), *(len(str(row[i])) for row in rows))
              for i in range(len(headers))] if rows else [len(str(h)) for h in headers]
    formato = "  ".join("{" + ":" + a + str(w) + "}" for a, w in zip(aligns, widths))
    print(formato.format(*[str(h) for h in headers]))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        print(formato.format(*[str(value) for value in row]))


set_global_seed(SEED)
print("Python:", platform.python_version())
print("TensorFlow:", tf.__version__)
print("GPU detectada:", tf.config.list_physical_devices("GPU"))
print("Semilla:", SEED, "| Epocas del manual:", EPOCHS, "| Validacion:", DEFAULT_VALIDATION_SIZE)
print("Funcion de auditoria disponible:", audit_dataset.__name__)"""
        ),
        MD(
            """## 1. Auditoría del dataset (antes de entrenar)

Fashion-MNIST es un benchmark público de moda en escala de grises con 28×28 píxeles
y 10 clases. TensorFlow lo descarga a su caché local la primera vez.

Se registran **formas, tipos, rangos de valores y reparto por clase** antes de
entrenar, tal como pide el manual."""
        ),
        CODE(
            """(x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print("Entrenamiento oficial:", x_train_all.shape, x_train_all.dtype, "etiquetas:", y_train_all.shape)
print("Prueba oficial:        ", x_test.shape, x_test.dtype, "etiquetas:", y_test.shape)
print("Rango de valores (entrenamiento):", x_train_all.min(), "a", x_train_all.max())
print("Rango de valores (prueba):       ", x_test.min(), "a", x_test.max())
print("Numero de clases:", NUM_CLASSES)
print("Nombres de las clases:", list(CLASS_NAMES))"""
        ),
        CODE(
            """distribucion = class_distribution(y_train_all)
print_table(
    ["clase", "entrenamiento (60.000)", "%", "prueba (10.000)", "%"],
    [
        [
            nombre,
            distribucion[nombre],
            f"{100 * distribucion[nombre] / 60000:.2f}",
            class_distribution(y_test)[nombre],
            f"{100 * class_distribution(y_test)[nombre] / 10000:.2f}",
        ]
        for nombre in CLASS_NAMES
    ],
    ["<", ">", ">", ">", ">"],
)
valores = list(distribucion.values())
print("Desbalance del bloque de entrenamiento (max/min):",
      f"{max(valores) / min(valores):.4f}")"""
        ),
        CODE(
            """figure, axes = plt.subplots(1, 2, figsize=(14, 4.6), layout="constrained")
posiciones = np.arange(NUM_CLASSES)
conteo_prueba = class_distribution(y_test)
axes[0].bar(posiciones, [distribucion[n] for n in CLASS_NAMES], color="#4C72B0")
axes[0].set_title("Reparto por clase · bloque de entrenamiento (60.000)")
axes[0].set_xticks(posiciones)
axes[0].set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
axes[0].set_ylabel("numero de imagenes")
axes[0].grid(axis="y", alpha=0.3)
axes[1].bar(posiciones, [conteo_prueba[n] for n in CLASS_NAMES], color="#DD8452")
axes[1].set_title("Reparto por clase · bloque de prueba (10.000)")
axes[1].set_xticks(posiciones)
axes[1].set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
axes[1].set_ylabel("numero de imagenes")
axes[1].grid(axis="y", alpha=0.3)
figure.suptitle("Auditoria del dataset: distribucion de clases de Fashion-MNIST")
plt.show()"""
        ),
        MD(
            """### 1.1 Normalización y partición reproducible

* Las imágenes se normalizan a `[0, 1]` (float32) y se les añade el canal: `(N, 28, 28, 1)`.
* La partición es **fija y determinista**: de las 60.000 imágenes de entrenamiento,
  las **54.000** primeras forman el conjunto de entrenamiento y las **6.000** últimas
  el de validación. Las **10.000** del bloque oficial de prueba no se usan nunca
  durante el entrenamiento (no hay fuga de datos).
* La semilla es **42** para Python, NumPy y TensorFlow, de modo que el
  inicializador de pesos, el orden de los batches y la elección de ejemplos de
  error son reproducibles."""
        ),
        CODE(
            """x_all = normalize_images(x_train_all)
x_test_norm = normalize_images(x_test)
x_train, y_train, x_valid, y_valid = split_train_validation(x_all, y_train_all, DEFAULT_VALIDATION_SIZE)

print("Forma de las imagenes normalizadas:", x_all.shape, x_all.dtype)
print("Rango:", float(x_all.min()), "a", float(x_all.max()))
print()
print("Entrenamiento:", x_train.shape, "| Validacion:", x_valid.shape, "| Prueba:", x_test_norm.shape)
print("La validacion son las ultimas", DEFAULT_VALIDATION_SIZE,
      "filas del bloque de entrenamiento; la prueba nunca se toca al entrenar.")

# Comprobacion de reproducibilidad: dos particiones consecutivas deben ser identicas.
x_train_b, y_train_b, x_valid_b, y_valid_b = split_train_validation(x_all, y_train_all, DEFAULT_VALIDATION_SIZE)
reproducible = (np.array_equal(x_train, x_train_b) and np.array_equal(y_valid, y_valid_b))
print("Particion reproducible:", reproducible)"""
        ),
        CODE(
            """conteo_validacion = class_distribution(y_valid)
print_table(
    ["clase", "entrenamiento", "validacion", "% validacion", "prueba"],
    [
        [
            nombre,
            distribucion[nombre],
            conteo_validacion[nombre],
            f"{100 * conteo_validacion[nombre] / 6000:.2f}",
            conteo_prueba[nombre],
        ]
        for nombre in CLASS_NAMES
    ],
    ["<", ">", ">", ">", ">"],
)
valores_validacion = list(conteo_validacion.values())
print("Reparto de la validacion (minimo - maximo):",
      min(valores_validacion), "-", max(valores_validacion),
      "| desbalance max/min:", f"{max(valores_validacion) / min(valores_validacion):.4f}")"""
        ),
        MD(
            """## 2. Entrenamiento: baseline denso y CNN con la misma partición

| Modelo | Arquitectura |
|---|---|
| Baseline denso | `Flatten → Dense(64, relu) → Dense(10, softmax)` |
| CNN | `Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → GlobalAveragePooling → Dropout(0.25) → Dense(10, softmax)` |

Los dos modelos se entrenan **con el mismo bloque de 54.000 ejemplos, la misma
validación de 6.000 y la misma semilla**, lo que hace la comparación justa.
Optimizador Adam, pérdida `sparse_categorical_crossentropy`, batch 128 y
`EarlyStopping(patience=2, restore_best_weights=True)`.

La función `run_experiment()` de `src/inf8239_u02_cv/experiment.py` deja las métricas
y las figuras en `reports/`."""
        ),
        CODE(
            """# Se guardan las metricas que hubiera escrito el script para poder compararlas despues.
metricas_previas = None
ruta_metricas = REPORTS / "cv_metrics.json"
if ruta_metricas.exists():
    metricas_previas = json.loads(ruta_metricas.read_text(encoding="utf-8"))
    print("Metricas previas detectadas (escritas por scripts/train_cv.py):")
    print(json.dumps(metricas_previas["summary"], indent=2))
else:
    print("Todavia no hay metricas previas: esta ejecucion del notebook sera la primera.")

print()
print("Arquitecturas que se van a entrenar:")
for nombre, constructor in (("Baseline denso", build_dense), ("CNN", build_cnn)):
    modelo = constructor(tf)
    print(f"  {nombre}: {modelo.count_params()} parametros, "
          f"{len(modelo.layers)} capas, salida {modelo.output_shape}")"""
        ),
        CODE(
            """payload = run_experiment(
    epochs=EPOCHS,
    seed=SEED,
    batch_size=BATCH_SIZE,
    repeats=REPEATS,
    reports_dir=REPORTS,
    models_dir=MODELS,
    verbose=2,
)
print("Artefactos en:", REPORTS)"""
        ),
        MD(
            """### 2.1 Curvas de entrenamiento y validación

Se guardan en `reports/training_curves.png`. Los cuatro patrones que propone el
manual se leen así:

| Patrón observado | Lectura |
|---|---|
| Ambas pérdidas disminuyen | El modelo está aprendiendo patrones útiles en ese intervalo. |
| Pérdida de entrenamiento baja y validación sube | Posible sobreajuste: conservar el mejor punto y revisar regularización. |
| Ambas pérdidas permanecen altas | Posible subajuste, configuración inadecuada o problema de datos. |
| Accuracy alta pero una clase con recall bajo | El promedio oculta una debilidad específica: revisar matriz y ejemplos. |"""
        ),
        CODE(
            """display(Image(filename=str(REPORTS / "training_curves.png"), width=950))"""
        ),
        CODE(
            """for nombre, titulo in (("dense", "Baseline denso"), ("cnn", "CNN")):
    history = payload[nombre]["history"]
    print(f"--- {titulo} ---")
    print_table(
        ["epoca", "perdida ent.", "perdida val.", "acc. ent.", "acc. val."],
        [
            [
                epoca,
                f"{loss:.4f}",
                f"{val_loss:.4f}",
                f"{acc:.4f}",
                f"{val_acc:.4f}",
            ]
            for epoca, (loss, val_loss, acc, val_acc) in enumerate(
                zip(history["loss"], history["val_loss"],
                    history["accuracy"], history["val_accuracy"]), start=1)
        ],
        [">", ">", ">", ">", ">"],
    )
    print()

# Diagnostico del patron observado (los cuatro patrones del manual).
# Se llama a la MISMA funcion que usan el generador del PDF y
# scripts/verify_cifras.py: src/inf8239_u02_cv/diagnostico.py
for nombre, titulo in (("dense", "Baseline denso"), ("cnn", "CNN")):
    diagnostico = diagnostico_curvas(payload[nombre]["history"])
    print(f"{titulo}: {diagnostico['patron']}")
    print(f"    {diagnostico['frase']}")
    print(f"    caida de la perdida de entrenamiento = {diagnostico['caida_entrenamiento']:.4f} | "
          f"caida de la de validacion = {diagnostico['caida_validacion']:.4f} | "
          f"accuracy de validacion final = {diagnostico['accuracy_validacion_final']:.4f} | "
          f"epocas ejecutadas = {diagnostico['epocas']} de {EPOCHS}")"""
        ),
        MD(
            """## 3. Métricas globales: denso frente a CNN

La comparación usa **la misma partición** y la misma semilla, y el tiempo de
inferencia es la **mediana de tres mediciones** sobre las 10.000 imágenes de prueba."""
        ),
        CODE(
            """filas_global = [
    [
        "Baseline denso",
        f"{payload['dense']['f1_macro']:.4f}",
        f"{payload['dense']['accuracy']:.4f}",
        payload["dense"]["parameters"],
        f"{payload['dense']['model_size_kb']:.2f}",
        f"{payload['dense']['train_seconds']:.1f}",
        f"{payload['dense']['inference_ms_per_image']:.4f}",
        payload["dense"]["epochs_completed"],
    ],
    [
        "CNN",
        f"{payload['cnn']['f1_macro']:.4f}",
        f"{payload['cnn']['accuracy']:.4f}",
        payload["cnn"]["parameters"],
        f"{payload['cnn']['model_size_kb']:.2f}",
        f"{payload['cnn']['train_seconds']:.1f}",
        f"{payload['cnn']['inference_ms_per_image']:.4f}",
        payload["cnn"]["epochs_completed"],
    ],
]
encabezados = ["modelo", "F1 macro", "accuracy", "parametros", "tamano (KB)",
               "entrenamiento (s)", "inferencia mediana (ms/img)", "epocas"]
print_table(encabezados, filas_global, ["<", ">", ">", ">", ">", ">", ">", ">"])
print()
print("Mediciones de inferencia (ms por imagen) y mediana usada:")
print("  denso:", payload["dense"]["inference_ms_per_image_runs"],
      "-> mediana", payload["dense"]["inference_ms_per_image"])
print("  CNN  :", payload["cnn"]["inference_ms_per_image_runs"],
      "-> mediana", payload["cnn"]["inference_ms_per_image"])
print()
print("Hardware y entorno de ejecucion:")
for clave, valor in payload["meta"]["hardware"].items():
    print(f"  {clave}: {valor}")"""
        ),
        MD(
            """## 4. Métricas por clase

El promedio global puede esconder una debilidad. La tabla siguiente viene de
`reports/cv_per_class.csv` (classification report por clase)."""
        ),
        CODE(
            """import csv

with (REPORTS / "cv_per_class.csv").open(encoding="utf-8") as archivo:
    filas_pc = list(csv.DictReader(archivo))

for modelo in ("dense", "cnn"):
    print(f"--- {modelo} ---")
    print_table(
        ["clase", "precision", "recall", "F1", "soporte"],
        [
            [
                fila["class_name"],
                fila["precision"],
                fila["recall"],
                fila["f1"],
                fila["support"],
            ]
            for fila in filas_pc if fila["model"] == modelo
        ],
        ["<", ">", ">", ">", ">"],
    )
    print()"""
        ),
        CODE(
            """print("Classification report de la CNN (salida de sklearn):")
print(payload["cnn"]["classification_report"])
print("Classification report del baseline denso:")
print(payload["dense"]["classification_report"])"""
        ),
        MD(
            """## 5. Matriz de confusión

`reports/confusion_cnn.png` (y `reports/confusion_dense.png` para el baseline).
Las filas son la clase real y las columnas la clase predicha."""
        ),
        CODE(
            """display(Image(filename=str(REPORTS / "confusion_cnn.png"), width=820))
confusion = np.array(payload["cnn"]["confusion_matrix"])
print("Suma de la diagonal (aciertos de la CNN):", int(np.trace(confusion)))
print("Suma de la matriz (ejemplos de prueba):", int(confusion.sum()))
print("Los errores por clase se derivan de esta matriz en el apartado 6.")"""
        ),
        MD(
            """## 6. Errores por clase y ejemplos visuales

* `reports/cv_errors_by_class.csv`: errores, falsos positivos y recall por clase.
* `reports/errors_by_class.png`: recall y número de errores por clase.
* `reports/cnn_errors.png`: 16 ejemplos mal clasificados **elegidos al azar con
  semilla fija** (42), con su clase real y su predicción.
* `reports/cnn_error_samples.csv`: ficha de esos 16 ejemplos con la confianza."""
        ),
        CODE(
            """with (REPORTS / "cv_errors_by_class.csv").open(encoding="utf-8") as archivo:
    filas_err = list(csv.DictReader(archivo))

for modelo in ("dense", "cnn"):
    print(f"--- {modelo} ---")
    print_table(
        ["clase", "soporte", "aciertos", "errores", "tasa error", "recall", "falsos pos."],
        [
            [
                fila["class_name"],
                fila["support"],
                fila["correct"],
                fila["errors"],
                fila["error_rate"],
                fila["recall"],
                fila["false_positives"],
            ]
            for fila in filas_err if fila["model"] == modelo
        ],
        ["<", ">", ">", ">", ">", ">", ">"],
    )
    print()"""
        ),
        CODE(
            """display(Image(filename=str(REPORTS / "errors_by_class.png"), width=950))"""
        ),
        CODE(
            """for modelo in ("dense", "cnn"):
    propias = [fila for fila in filas_err if fila["model"] == modelo]
    propias_ordenadas = sorted(propias, key=lambda fila: float(fila["recall"]))
    print(f"Clases que el modelo {modelo} pierde (ordenadas por recall mas bajo):")
    for fila in propias_ordenadas:
        print(f"  {fila['class_name']:<12} recall={float(fila['recall']):.3f} | "
              f"errores={fila['errors']:>4} de {fila['support']} | "
              f"falsos positivos={fila['false_positives']:>4}")
    print(f"  Total de errores sobre 10.000 ejemplos: "
          f"{sum(int(f['errors']) for f in propias)}")
    print()"""
        ),
        CODE(
            """display(Image(filename=str(REPORTS / "cnn_errors.png"), width=820))

with (REPORTS / "cnn_error_samples.csv").open(encoding="utf-8") as archivo:
    print_table(
        ["indice", "clase real", "clase predicha", "confianza", "prob. real"],
        [
            [
                fila["test_index"],
                fila["true_class"],
                fila["predicted_class"],
                fila["confidence"],
                fila["true_probability"],
            ]
            for fila in csv.DictReader(archivo)
        ],
        [">", "<", "<", ">", ">"],
    )"""
        ),
        MD(
            """### 6.1 Lectura de los errores

Los pares que se repiten son los esperados en un benchmark de moda en 28×28:

* **Shirt ↔ T-shirt/top ↔ Pullover ↔ Coat**: las prendas de torso se distinguen por
  un cuello y una manga de pocos píxeles; cuando el cuello está abierto o la manga
  no se distingue, la imagen es genuinamente ambigua.
* **Sandal ↔ Sneaker ↔ Ankle boot**: el calzado se separa por la suela y el tobillo,
  detalles que ocupan muy pocos píxeles en la parte inferior de la imagen.
* **T-shirt/top ↔ Dress**: la diferencia es la longitud y la forma; cuando la prenda
  ocupa todo el marco, ambas clases se solapan.

En ninguna de estas parejas hay un "fallo humano" evidente: son ambigüedades de la
misma imagen. Esto **no demuestra preparación para imágenes médicas, OCR ni
vigilancia**: Fashion-MNIST es un benchmark didáctico de siluetas de moda."""
        ),
        MD(
            """## 7. Estudio de presupuesto de entrenamiento (complemento)

El experimento principal usa el presupuesto que fija el manual (8 épocas). Las
curvas dejan una pregunta abierta: **¿la diferencia entre los dos modelos es una
limitación de la arquitectura o del número de épocas?**

Este apartado repite el experimento con **25 épocas** para los dos modelos, con la
**misma partición y la misma semilla**. Las curvas por época se calculan sobre el
conjunto de **validación**, nunca sobre la prueba, para no filtrar información.
Es un experimento **complementario**: la comparación oficial sigue siendo la de 8
épocas, y su resultado se reporta sin modificar."""
        ),
        CODE(
            """ruta_estudio = REPORTS / "budget_study.json"
estudio = None
if ruta_estudio.exists():
    previo = json.loads(ruta_estudio.read_text(encoding="utf-8"))
    if previo["meta"]["epochs"] == BUDGET_EPOCHS and previo["meta"]["seed"] == SEED:
        estudio = previo
        print("Se reutiliza reports/budget_study.json, generado por scripts/budget_study.py")
        print("con la misma particion y la misma semilla.")
if estudio is None:
    print("No hay estudio de presupuesto previo: se ejecuta ahora en el notebook.")
    estudio = run_budget_study(
        epochs=BUDGET_EPOCHS,
        seed=SEED,
        batch_size=BATCH_SIZE,
        reports_dir=REPORTS,
        models_dir=MODELS,
        verbose=2,
    )

linea_base = (
    {nombre: metricas_previas[nombre]["f1_macro"] for nombre in ("dense", "cnn")}
    if metricas_previas is not None
    else {nombre: payload[nombre]["f1_macro"] for nombre in ("dense", "cnn")}
)
plot_budget_study(estudio, linea_base, REPORTS / "budget_study.png", prescribed_epochs=EPOCHS)
display(Image(filename=str(REPORTS / "budget_study.png"), width=950))"""
        ),
        CODE(
            """print_table(
    ["modelo", "F1 macro 8 ep.", "F1 macro 25 ep.", "parametros", "tamano (KB)",
     "entrenamiento 8 ep. (s)", "entrenamiento 25 ep. (s)"],
    [
        [
            nombre,
            f"{payload[nombre]['f1_macro']:.4f}",
            f"{estudio['models'][nombre]['f1_macro_test']:.4f}",
            payload[nombre]["parameters"],
            f"{payload[nombre]['model_size_kb']:.2f}",
            f"{payload[nombre]['train_seconds']:.1f}",
            f"{estudio['models'][nombre]['train_seconds']:.1f}",
        ]
        for nombre in ("dense", "cnn")
    ],
    ["<", ">", ">", ">", ">", ">", ">"],
)"""
        ),
        MD(
            """## 8. Costo, tamaño y lectura Green AI

El manual es explícito: *más accuracy no es automáticamente mejor*. Se comparan
desempeño, parámetros, tamaño en disco, tiempo de entrenamiento, tiempo de
inferencia y hardware."""
        ),
        CODE(
            """dens = payload["dense"]
cnn = payload["cnn"]
print("Comparacion de costo con el presupuesto del manual (8 epocas):")
print(f"  F1 macro:            denso {dens['f1_macro']:.4f} | CNN {cnn['f1_macro']:.4f} "
      f"({cnn['f1_macro'] - dens['f1_macro']:+.4f})")
print(f"  Parametros:          denso {dens['parameters']} | CNN {cnn['parameters']} "
      f"({cnn['parameters'] - dens['parameters']:+d})")
print(f"  Tamano del archivo:  denso {dens['model_size_kb']:.2f} KB | CNN {cnn['model_size_kb']:.2f} KB "
      f"({cnn['model_size_kb'] - dens['model_size_kb']:+.2f} KB)")
print(f"  Entrenamiento:        denso {dens['train_seconds']:.1f} s | CNN {cnn['train_seconds']:.1f} s "
      f"({cnn['train_seconds'] - dens['train_seconds']:+.1f} s)")
print(f"  Inferencia mediana:  denso {dens['inference_ms_per_image']:.4f} ms/img | "
      f"CNN {cnn['inference_ms_per_image']:.4f} ms/img "
      f"({cnn['inference_ms_per_image'] - dens['inference_ms_per_image']:+.4f} ms/img)")
print()
print("Costo con el presupuesto ampliado (25 epocas, estudio complementario):")
for nombre in ("dense", "cnn"):
    modelo = estudio["models"][nombre]
    print(f"  {nombre}: F1 macro {modelo['f1_macro_test']:.4f} | "
          f"{modelo['train_seconds']:.1f} s de entrenamiento | "
          f"{modelo['parameters']} parametros | {modelo['model_size_kb']:.2f} KB")"""
        ),
        MD(
            """## 9. Decisión técnica

La decisión se calcula **con las cifras de este cuaderno**, no con la expectativa.
Criterio aplicado:

1. **Con el presupuesto del manual (8 épocas)** se compara F1 macro, parámetros,
   tamaño, tiempo de entrenamiento e inferencia.
2. Si la diferencia se explica por **subajuste** (las curvas siguen bajando y la
   validación todavía mejora al final), se evalúa el presupuesto ampliado antes de
   concluir que la arquitectura es peor.
3. Se declara explícitamente el modelo entregado **y para qué régimen** es la
   elección correcta.

El bloque siguiente aplica el criterio y escribe el resultado en
`reports/decision_tecnica.txt`."""
        ),
        CODE(
            """dens = payload["dense"]
cnn = payload["cnn"]
f1_denso_8 = dens["f1_macro"]
f1_cnn_8 = cnn["f1_macro"]
f1_denso_25 = estudio["models"]["dense"]["f1_macro_test"]
f1_cnn_25 = estudio["models"]["cnn"]["f1_macro_test"]

factor_entrenamiento = cnn["train_seconds"] / dens["train_seconds"]
factor_inferencia = cnn["inference_ms_per_image"] / dens["inference_ms_per_image"]

if f1_cnn_8 > f1_denso_8:
    entregado = "CNN"
    justificacion = (
        f"Con el presupuesto del manual la CNN alcanza un F1 macro de {f1_cnn_8:.4f} frente a "
        f"{f1_denso_8:.4f} del baseline denso y ademas ocupa menos disco, de modo que la mejora "
        f"predictiva no cuesta mas recurso."
    )
else:
    entregado = "Baseline denso"
    justificacion = (
        f"Con {EPOCHS} epocas el baseline denso alcanza un F1 macro de {f1_denso_8:.4f} frente a "
        f"{f1_cnn_8:.4f} de la CNN, y ademas entrena {factor_entrenamiento:.1f} veces mas rapido y "
        f"infiere {factor_inferencia:.1f} veces mas rapido. La CNN no aporta mejora predictiva con "
        f"este presupuesto, de modo que segun el principio Green AI del manual (mas accuracy no es "
        f"automaticamente mejor, y hay que justificar el coste) no se justifica su coste de "
        f"entrenamiento. Con {BUDGET_EPOCHS} epocas la CNN sube a {f1_cnn_25:.4f} y el denso a "
        f"{f1_denso_25:.4f}: el resultado es compatible con una mejora de la arquitectura de la CNN "
        f"al entrenar con mas computo, pero con una sola semilla y sin intervalos de confianza no "
        f"permite separar de forma concluyente el efecto del presupuesto del efecto de la arquitectura."
    )

alternativo = (
    f"La CNN se conserva como alternativa documentada: usa {cnn['parameters']} parametros frente a "
    f"{dens['parameters']} del denso y ocupa {cnn['model_size_kb']:.2f} KB frente a "
    f"{dens['model_size_kb']:.2f} KB, es decir, la unica dimension en la que gana es la huella "
    f"(memoria y tamano desplegado). Es el modelo que habria que elegir si la restriccion dominante "
    f"fuera el tamano del artefacto o la memoria del dispositivo, no el tiempo de CPU."
)

decision = (
    "Decision tecnica (calculada sobre las metricas de esta ejecucion)\\n"
    "--------------------------------------------------------------\\n"
    f"  Modelo entregado: {entregado}\\n"
    f"  Presupuesto del manual ({EPOCHS} epocas):\\n"
    f"    F1 macro denso = {f1_denso_8:.4f} | F1 macro CNN = {f1_cnn_8:.4f} "
    f"({f1_cnn_8 - f1_denso_8:+.4f})\\n"
    f"    Entrenamiento denso = {dens['train_seconds']:.1f} s | CNN = {cnn['train_seconds']:.1f} s "
    f"({factor_entrenamiento:.1f}x)\\n"
    f"    Parametros denso = {dens['parameters']} | CNN = {cnn['parameters']}\\n"
    f"    Tamano denso = {dens['model_size_kb']:.2f} KB | CNN = {cnn['model_size_kb']:.2f} KB\\n"
    f"    Inferencia mediana denso = {dens['inference_ms_per_image']:.4f} ms/img | "
    f"CNN = {cnn['inference_ms_per_image']:.4f} ms/img ({factor_inferencia:.1f}x)\\n"
    f"  Presupuesto ampliado ({BUDGET_EPOCHS} epocas, experimento complementario):\\n"
    f"    F1 macro denso = {f1_denso_25:.4f} | F1 macro CNN = {f1_cnn_25:.4f} "
    f"({f1_cnn_25 - f1_denso_25:+.4f})\\n"
    f"  Justificacion: {justificacion}\\n"
    f"  Alternativa documentada: {alternativo}\\n"
    "  Limitacion del benchmark: Fashion-MNIST es didactico; el resultado no se extrapola\\n"
    "  a imagenes medicas, OCR ni vigilancia.\\n"
    "  Limitacion del entorno: la ejecucion fue en CPU (sin GPU detectada), por lo que los\\n"
    "  tiempos de entrenamiento e inferencia son validos para esta maquina y no para un equipo\\n"
    "  con GPU.\\n"
)
print(decision)
(REPORTS / "decision_tecnica.txt").write_text(decision, encoding="utf-8")
print("Decision escrita en reports/decision_tecnica.txt")"""
        ),
        MD(
            """## 10. Reproducibilidad: el notebook y el script deben coincidir

`scripts/train_cv.py` y este notebook llaman a la misma función
`run_experiment()`. Se comprueba que los artefactos escritos por el notebook
coinciden con los que había escrito el script (mismo F1 macro, mismos parámetros,
mismo tamaño)."""
        ),
        CODE(
            """if metricas_previas is not None:
    print_table(
        ["modelo", "F1 macro script", "F1 macro notebook", "diferencia"],
        [
            [
                nombre,
                f"{metricas_previas[nombre]['f1_macro']:.6f}",
                f"{payload[nombre]['f1_macro']:.6f}",
                f"{abs(metricas_previas[nombre]['f1_macro'] - payload[nombre]['f1_macro']):.6f}",
            ]
            for nombre in ("dense", "cnn")
        ],
        ["<", ">", ">", ">"],
    )
else:
    print("No habia metricas previas del script para comparar: esta fue la primera ejecucion.")"""
        ),
        MD(
            """## 11. Cierre interpretativo (manual LAB07)

* **Resultado principal:** dos modelos comparados con la misma partición y semilla.
* **Evidencia predictiva:** F1 macro, accuracy, métricas por clase, matriz de confusión.
* **Clase más difícil:** la que aparece con menor recall en la tabla por clase.
* **Costo comparado:** parámetros, tamaño en disco, tiempo de entrenamiento e inferencia mediana.
* **¿La mejora justifica el costo?:** se responde con las cifras de los apartados 8 y 9.
* **Limitación del benchmark:** Fashion-MNIST es un benchmark didáctico de siluetas de moda.
* **Decisión antes de usar otro dominio:** reentrenar y reevaluar con datos del dominio,
  nunca reutilizar estos pesos.

## 12. Referencias

* Manual de la Unidad 02 de INF-8239, laboratorio `U02.LAB07` (prof. Edwin Ramón José Nolasco).
* Ficha de Keras de `fashion_mnist` (licencia MIT): <https://keras.io/api/datasets/fashion_mnist/>
* Documentación de `tf.keras.datasets.fashion_mnist`:
  <https://www.tensorflow.org/datasets/catalog/fashion_mnist>"""
        ),
    ]


def main() -> None:
    notebook = nbf.v4.new_notebook(cells=cells())
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Notebook escrito en {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
