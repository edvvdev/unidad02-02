from __future__ import annotations

import numpy as np

# Nombres oficiales de las 10 clases de Fashion-MNIST.
# Fuente: ficha de Keras para fashion_mnist (licencia MIT).
CLASS_NAMES: tuple[str, ...] = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)
NUM_CLASSES = len(CLASS_NAMES)
DEFAULT_SEED = 42
DEFAULT_VALIDATION_SIZE = 6000
IMAGE_SHAPE = (28, 28, 1)


def normalize_images(images: np.ndarray) -> np.ndarray:
    values = images.astype("float32") / 255.0
    return values[..., None]


def validate_images(images: np.ndarray, labels: np.ndarray) -> None:
    if images.ndim != 4 or images.shape[1:] != (28, 28, 1):
        raise ValueError(f"Forma inesperada: {images.shape}")
    if len(images) != len(labels):
        raise ValueError("Imágenes y etiquetas no coinciden")
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Las imágenes deben estar normalizadas entre 0 y 1")


def split_train_validation(
    images: np.ndarray,
    labels: np.ndarray,
    validation_size: int = DEFAULT_VALIDATION_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide el bloque de entrenamiento en entrenamiento y validacion.

    La particion es **fija y determinista**: las ultimas ``validation_size``
    filas del bloque oficial de 60.000 imagenes de TensorFlow, tal como en el
    proyecto base del profesor. No se baraja, de modo que dos ejecuciones
    obtienen exactamente el mismo conjunto de validacion.
    """
    if validation_size <= 0 or validation_size >= len(images):
        raise ValueError(f"validation_size invalido: {validation_size}")
    if len(images) != len(labels):
        raise ValueError("Imagenes y etiquetas no coinciden")
    train_images = images[:-validation_size]
    train_labels = labels[:-validation_size]
    valid_images = images[-validation_size:]
    valid_labels = labels[-validation_size:]
    return train_images, train_labels, valid_images, valid_labels


def class_distribution(labels: np.ndarray) -> dict[str, int]:
    """Devuelve el numero de ejemplos de cada clase."""
    counts = np.bincount(np.asarray(labels), minlength=NUM_CLASSES).tolist()
    return {name: counts[index] for index, name in enumerate(CLASS_NAMES)}


def class_shares(labels: np.ndarray) -> dict[str, float]:
    """Devuelve el porcentaje de ejemplos de cada clase."""
    counts = np.bincount(np.asarray(labels), minlength=NUM_CLASSES).tolist()
    total = sum(counts)
    return {name: round(100.0 * counts[index] / total, 3)
            for index, name in enumerate(CLASS_NAMES)}


def audit_dataset(splits: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Audita las particiones antes de entrenar (formas, rangos y reparto).

    ``splits`` relaciona el nombre de la particion con el par
    ``(imagenes, etiquetas)``. El resultado es un diccionario serializable que
    el script y el notebook guardan en ``reports/dataset_audit.json``.
    """
    audit: dict = {
        "num_classes": NUM_CLASSES,
        "class_names": list(CLASS_NAMES),
        "image_shape": list(IMAGE_SHAPE),
        "splits": {},
    }
    for name, (images, labels) in splits.items():
        counts = np.bincount(np.asarray(labels), minlength=NUM_CLASSES).tolist()
        total = sum(counts)
        audit["splits"][name] = {
            "n_samples": len(labels),
            "n_images": len(images),
            "image_shape": list(images.shape[1:]),
            "dtype": str(images.dtype),
            "value_min": float(images.min()),
            "value_max": float(images.max()),
            "value_mean": round(float(images.mean()), 6),
            "class_counts": {CLASS_NAMES[i]: counts[i] for i in range(NUM_CLASSES)},
            "class_share_pct": {
                CLASS_NAMES[i]: round(100.0 * counts[i] / total, 3)
                for i in range(NUM_CLASSES)
            },
            "class_count_min": min(counts),
            "class_count_max": max(counts),
            "imbalance_ratio_max_over_min": round(max(counts) / min(counts), 4),
        }
    return audit
