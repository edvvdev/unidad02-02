# Model Card · Fashion-MNIST CNN

**Estudiante:** Edwing Perez · **INF-8239 Ciencia de Datos II · Unidad 02 · U02.LAB07**
**Repositorio:** https://github.com/edvvdev/unidad02-02
**Estado:** resultados de una ejecución real (semilla 42, partición 54.000 / 6.000 / 10.000).
Todas las cifras proceden de `reports/cv_metrics.json`, `reports/cv_per_class.csv`,
`reports/cv_errors_by_class.csv` y `reports/budget_study.json`.

## Modelo y versión

| | Baseline denso (modelo entregado) | CNN (alternativa compacta) |
|---|---|---|
| Archivo | `models/best_dense.keras` | `models/best_cnn.keras` |
| Arquitectura | `Flatten → Dense(64, relu) → Dense(10, softmax)` | `Conv2D(32,3,same,relu) → MaxPool → Conv2D(64,3,same,relu) → MaxPool → GlobalAveragePooling2D → Dropout(0.25) → Dense(10, softmax)` |
| Entrada | `(N, 28, 28, 1)` float32 en `[0, 1]` | idem |
| Salida | 10 probabilidades softmax que suman 1 | idem |
| Parámetros entrenables | 50890 | 19466 |
| Tamaño del archivo | 619.12 KB | 262.17 KB |
| Optimizador | Adam (lr por defecto 0,001), sin decaimiento | idem |
| Pérdida | `sparse_categorical_crossentropy` | idem |
| Regularización | `EarlyStopping(patience=2, restore_best_weights=True)` | idem |
| Épocas | 8 de 8 solicitadas | 8 de 8 solicitadas |
| Semilla | 42 (Python, NumPy y TensorFlow) | 42 |

**Modelo entregado: **baseline denso**.** La CNN se conserva como alternativa documentada
por su huella (19466 parámetros y 262.17 KB frente a
50890 y 619.12 KB del denso): es la opción preferente si la restricción dominante es la
memoria o el tamaño del artefacto desplegado. El razonamiento completo está en
`reports/decision_tecnica.txt` y en el apartado 8 del notebook.

## Uso previsto

* Demostración académica de un clasificador de imágenes en Fashion-MNIST: **10 clases
  de prendas de moda en escala de grises, 28×28**.
* Reproducir el laboratorio U02.LAB07: auditoría del dataset, partición reproducible,
  comparación justa baseline/CNN, métricas por clase, análisis de errores, comparación
  de costo y decisión técnica.
* Uso como **modelo de referencia** para estudiar subajuste, la relación entre
  presupuesto de entrenamiento y desempeño, y el coste (Green AI) de una arquitectura
  convolucional frente a una capa densa.
* Uso educativo para escribir pruebas de contrato sobre la forma de la salida, la
  suma de probabilidades y la coherencia de las métricas por clase.

## Usos fuera de alcance

* **No** es un modelo de visión médica, de OCR, de vigilancia ni de reconocimiento
  biométrico. Fashion-MNIST es un benchmark **didáctico** de siluetas de moda; un
  buen resultado aquí **no** demuestra preparación para ningún otro dominio.
* **No** debe usarse con imágenes de personas, ropa real, catálogos de producto,
  vídeo, imágenes en color o cualquier distribución distinta de Fashion-MNIST.
* **No** debe aplicarse a decisiones que afecten a personas (precio, crédito,
  contratación, accesos) sin un ciclo de validación propio en el dominio y con
  supervisión humana.
* **No** es un detector: no localiza prendas ni mide tallas: solo clasifica la clase
  de la prenda dibujada en la imagen.
* **No** se ha evaluado con datos sesgados, adversarios o venenosos, ataques de
  evasión ni datos fuera de distribución: su comportamiento en esas condiciones es
  **desconocido**.

## Dataset y particiones

| | |
|---|---|
| Dataset | Fashion-MNIST (`tf.keras.datasets.fashion_mnist`) |
| Procedencia y licencia | Ficha de Keras para `fashion_mnist`, **licencia MIT** |
| Descarga | Automática a la caché local de Keras en la primera ejecución |
| Clases | 10: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot |
| Formato original | 28×28, escala de grises, `uint8` |
| Bloque oficial de entrenamiento | 60,000 imágenes |
| Bloque oficial de prueba | 10.000 imágenes |

| Partición | Imágenes | Reparto por clase | Desbalance máx/mín |
|---|---|---|---|
| Entrenamiento | 54.000 | 5367–5445 por clase | 1,0145 |
| Validación | 6.000 | 555–633 por clase | 1,1405 |
| Prueba | 10.000 | 1000 por clase (exactamente balanceada) | 1.0000 |

* La partición es **fija y determinista**: las últimas 6.000 imágenes del
  bloque oficial de 60.000 son validación. **No** se baraja en el particionado.
* La semilla **42** se aplica a Python, NumPy y TensorFlow (`TF_DETERMINISTIC_OPS=1`),
  de modo que los pesos iniciales, el orden de los batches y la elección de los
  ejemplos de error son reproducibles.
* Las 10.000 imágenes de prueba **no se usan** para entrenar ni para seleccionar
  modelos: no hay fuga de datos. El estudio de presupuesto usa **validación** para
  las curvas por época.
* La auditoría completa (formas, tipos, rangos, reparto) está en
  `reports/dataset_audit.json`.

## Preprocesamiento

1. `uint8` → `float32` y división por 255: rango `[0, 1]` (verificado en la auditoría:
   mínimo 0,0 y máximo 1,0 en las tres particiones).
2. Añadido del canal: `(N, 28, 28)` → `(N, 28, 28, 1)`.
3. **Ninguna** normalización adicional (sin estandarización por canal, sin aumento de
   datos, sin recorte): el preprocesamiento es idéntico para entrenamiento,
   validación, prueba e inferencia.
4. Salida: 10 probabilidades softmax; en la ejecución de referencia todas las
   probabilidades son finitas y suman 1 (comprobado en `tests/`).

## Métricas globales y por clase

Conjunto de prueba: 10.000 imágenes. F1 macro y accuracy:

| Métrica | Baseline denso | CNN | Diferencia (CNN − denso) |
|---|---|---|---|
| F1 macro | **0.8647** | **0.7892** | -0.0755 |
| Accuracy | 0.8663 | 0.7930 | -0.0733 |
| Recall medio (10 clases) | 0.8663 | 0.7930 | -0.0733 |

### Métricas por clase (conjunto de prueba)

Soport de 10.000 ejemplos por clase en las diez clases.

| Clase | Precisión denso | Recall denso | F1 denso | Precisión CNN | Recall CNN | F1 CNN | Errores denso | Errores CNN |
|---|---|---|---|---|---|---|---|---|
| T-shirt/top | 0.791 | 0.848 | 0.8185 | 0.6446 | 0.807 | 0.7167 | 152 | 193 |
| Trouser | 0.9818 | 0.969 | 0.9753 | 0.9883 | 0.932 | 0.9593 | 31 | 68 |
| Pullover | 0.8362 | 0.684 | 0.7525 | 0.7064 | 0.64 | 0.6716 | 316 | 360 |
| Dress | 0.8369 | 0.903 | 0.8687 | 0.7542 | 0.844 | 0.7966 | 97 | 156 |
| Coat | 0.7108 | 0.87 | 0.7824 | 0.66 | 0.664 | 0.662 | 130 | 336 |
| Sandal | 0.9349 | 0.962 | 0.9483 | 0.9506 | 0.885 | 0.9166 | 38 | 115 |
| Shirt | 0.7193 | 0.597 | 0.6525 | 0.4973 | 0.372 | 0.4256 | 403 | 628 |
| Sneaker | 0.9289 | 0.94 | 0.9344 | 0.8547 | 0.953 | 0.9012 | 60 | 47 |
| Bag | 0.9724 | 0.953 | 0.9626 | 0.9168 | 0.937 | 0.9268 | 47 | 63 |
| Ankle boot | 0.967 | 0.937 | 0.9518 | 0.9353 | 0.896 | 0.9152 | 63 | 104 |

**Lectura.** El promedio global oculta la debilidad: la clase **Shirt** es la que
más se pierde en los dos modelos (recall 0.597 en el denso y 0.372 en
la CNN), seguida de **Pullover** y **Coat**. Son prendas de torso que se distinguen por
un cuello y una manga de pocos píxeles; cuando el cuello está abierto o la manga no se
distingue, la imagen es genuinamente ambigua. El detalle completo está en
`reports/cv_per_class.csv` y `reports/cv_errors_by_class.csv`.

## Comparación de costo

* Hardware: Linux-6.12.113+deb13-amd64-x86_64-with-glibc2.41, procesador x86_64, 20 núcleos lógicos,
  **GPU detectada: ninguna** (`tf.config.list_physical_devices("GPU")` → `[]`).
* Entorno: Python 3.12.15, TensorFlow 2.21.0, NumPy 2.5.3, uv con
  `uv sync --extra cpu`; el paquete del proyecto se instala en modo editable.
* Tiempos: **mediana de tres mediciones** de inferencia sobre las 10.000 imágenes
  de prueba.

| | Baseline denso | CNN |
|---|---|---|
| Parámetros entrenables | 50890 | 19466 |
| Tamaño del modelo (`.keras`) | 619.12 KB | 262.17 KB |
| Tiempo de entrenamiento (8 épocas) | 12.2 s | 68.9 s |
| Tiempo de inferencia (mediana de 3) | 0.0512 ms/imagen | 0.0823 ms/imagen |
| Mediciones de inferencia | 0.0512, 0.0446, 0.0622 ms/imagen | 0.0946, 0.0786, 0.0823 ms/imagen |

* **Estudio complementario de presupuesto** (misma partición y misma semilla,
  25 épocas): el denso alcanza F1 macro 0.8761 en
  32.0 s y la CNN 0.8439 en 215.3 s. Es un
  experimento **complementario**: la comparación oficial del laboratorio sigue siendo
  la de 8 épocas y su resultado se reporta sin modificar.
* **Alcance de la comparación.** El resultado es **compatible con una mejora de la
  arquitectura de la CNN al entrenar con más cómputo**, pero con **una sola semilla y
  sin intervalos de confianza** no permite separar de forma concluyente el efecto del
  presupuesto del efecto de la arquitectura. La decisión de entrega se apoya en las
  medidas de la ejecución de referencia (8 épocas), donde la comparación sí es
  concluyente.
* **Conclusión Green AI.** La CNN es más compacta (62 % menos parámetros y 58 % menos
  disco) pero, con este dataset, este presupuesto y esta arquitectura, **no aporta
  ninguna mejora predictiva**: el baseline denso gana en F1 macro
  (0.8647 frente a 0.7892), entrena 5.6 veces más rápido e
  infiere 1.6 veces más rápido. Una mejora que no existe no justifica el
  coste adicional, de modo que **se entrega el baseline denso** y la CNN queda
  documentada para entornos con restricción de memoria.

## Limitaciones y riesgos

* **Limitación del benchmark (principal).** Fashion-MNIST es un conjunto didáctico de
  siluetas de moda en 28×28 y en escala de grises. No contiene color, textura real,
  contexto, personas ni condiciones de iluminación variadas. Los resultados **no** se
  extrapolan a imágenes médicas, OCR, vigilancia, e-commerce real ni a ningún otro
  dominio: reentrenar y reevaluar.
* **Limitación del entorno de ejecución.** Toda la evidencia se generó **en CPU, sin
  GPU detectada**. Los tiempos de entrenamiento e inferencia son válidos para esta
  máquina y **no** son extrapolables a un equipo con GPU; en ese caso habría que
  repetir la medición de inferencia.
* **Subajuste de la CNN.** Con 8 épocas la CNN todavía no ha convergido
  (la pérdida de validación baja en la última época y la accuracy de validación queda
  en 0.7995). Sus métricas de este informe describen ese estado, no el
  rendimiento máximo alcanzable con más cómputo o una arquitectura mayor.
* **Arquitectura mínima.** El modelo convolucional del proyecto base tiene solo dos
  bloques de convolución y resume toda la imagen con `GlobalAveragePooling2D` (64
  características antes de la capa de salida). No es representativo del estado del
  arte en clasificación de imágenes.
* **Sobreajuste tardío del baseline.** En el presupuesto ampliado de 25
  épocas la pérdida de validación del denso deja de mejorar en las últimas épocas:
  con más entrenamiento habría que usar el punto guardado por `EarlyStopping` y
  regularizar.
* **Sin evaluación de equidad ni de robustez.** No se han medido sesgos por subclase
  (género, tipo de prenda), ni ataques adversarios, ni datos corruptos o fuera de
  distribución.
* **Sin trazabilidad de datos.** El dataset no tiene identificadores de origen: no se
  puede detectar ni auditar duplicados o fugas de información entre las particiones.
* **Reproducibilidad dependiente del entorno.** Con la misma semilla y la misma
  máquina los resultados se reproducen; con otra versión de TensorFlow, otro hardware
  o sin `TF_DETERMINISTIC_OPS` pueden variar ligeramente.

## Supervisión y monitoreo

* **Uso previsto:** exclusivamente académico y de demostración, con supervisión de la
  persona docente.
* **Supervisión humana obligatoria** en cualquier uso posterior: la predicción es una
  estimación, no una verdad, y en este dominio los errores no son triviales
  (por ejemplo, confundir una camisa con un jersey).
* **Revisión periódica:** si el modelo se actualiza (más datos, más épocas, otra
  arquitectura), hay que repetir la auditoría, la comparación con la misma partición,
  el análisis de errores por clase y actualizar esta ficha con las cifras nuevas.
* **Criterios de parada:** si el recall de la clase más débil (Shirt) baja por debajo
  de su valor de referencia (0.372 en la CNN, 0.597 en el denso),
  o si la inferencia mediana supera el 1 ms por imagen en el dispositivo de destino,
  el modelo no debe desplegarse sin revisar.
* **Trazabilidad:** cada entrega queda identificada por el par (semilla 42, partición
  54.000 / 6.000 / 10.000) y por el contenido de `reports/`, que es la fuente única de
  las cifras de esta ficha.
* **No hay uso en producción:** este modelo no está desplegado en ningún servicio y no
  debe integrarse en uno sin el ciclo de validación completo.

## Referencias

* Manual de la Unidad 02 de INF-8239, laboratorio `U02.LAB07` (prof. Edwin Ramón José Nolasco).
* Ficha de Keras de `fashion_mnist` (licencia MIT): <https://keras.io/api/datasets/fashion_mnist/>
* Documentación de `tf.keras.datasets.fashion_mnist`:
  <https://www.tensorflow.org/datasets/catalog/fashion_mnist>
