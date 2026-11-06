"""Experimento reproducible del LAB07: baseline denso frente a CNN.

El modulo concentra la auditoria del dataset, la particion, el entrenamiento de
los dos modelos, la evaluacion (global y por clase) y el analisis de errores.
Tanto ``scripts/train_cv.py`` como el notebook usan estas funciones, de modo que
ambos caminos producen exactamente los mismos artefactos en ``reports/``.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .data import (
    CLASS_NAMES,
    DEFAULT_SEED,
    DEFAULT_VALIDATION_SIZE,
    NUM_CLASSES,
    audit_dataset,
    normalize_images,
    split_train_validation,
    validate_images,
)
from .figures import (
    plot_confusion_matrix,
    plot_error_grid,
    plot_errors_by_class,
    plot_training_curves,
    select_error_indices,
)
from .models import build_cnn, build_dense

MODEL_BUILDERS = {"dense": build_dense, "cnn": build_cnn}
MODEL_TITLES = {"dense": "Baseline denso", "cnn": "CNN"}


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Fija las tres semillas (Python, NumPy y TensorFlow) para reproducibilidad."""
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_fashion_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carga Fashion-MNIST de la cache de Keras y normaliza a [0, 1] con canal."""
    (x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    return normalize_images(x_train_all), y_train_all, normalize_images(x_test), y_test


def compile_model(model):
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    epochs: int,
    batch_size: int,
    model_path: Path,
    verbose: int = 2,
):
    """Entrena un modelo y devuelve el modelo, su historial y el tiempo empleado."""
    factory = MODEL_BUILDERS[name]
    model = compile_model(factory(tf))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
    ]
    start = perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
    train_seconds = perf_counter() - start
    return model, history, train_seconds


def measure_inference(model, x_test: np.ndarray, repeats: int = 3):
    """Mide el tiempo de inferencia varias veces y devuelve la mediana."""
    timings: list[float] = []
    probabilities = None
    for _ in range(repeats):
        start = perf_counter()
        probabilities = model.predict(x_test, verbose=0)
        timings.append(1000.0 * (perf_counter() - start) / len(x_test))
    return float(np.median(timings)), [round(value, 4) for value in timings], probabilities


def per_class_rows(y_true: np.ndarray, y_pred: np.ndarray, model: str) -> list[dict]:
    """Precision, recall, F1 y soporte por clase (classification report)."""
    precision, recall, fscore, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    rows = []
    for index, name in enumerate(CLASS_NAMES):
        rows.append(
            {
                "model": model,
                "class_id": index,
                "class_name": name,
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1": round(float(fscore[index]), 4),
                "support": int(support[index]),
            }
        )
    return rows


def error_rows(y_true: np.ndarray, y_pred: np.ndarray, model: str) -> list[dict]:
    """Errores por clase: clase real mal predicha (FN) y falsos positivos (FP)."""
    rows = []
    for index, name in enumerate(CLASS_NAMES):
        real = np.asarray(y_true) == index
        support = int(real.sum())
        errors = int((real & (np.asarray(y_pred) != index)).sum())
        false_positives = int((~real & (np.asarray(y_pred) == index)).sum())
        rows.append(
            {
                "model": model,
                "class_id": index,
                "class_name": name,
                "support": support,
                "correct": support - errors,
                "errors": errors,
                "error_rate": round(errors / support, 4) if support else 0.0,
                "recall": round((support - errors) / support, 4) if support else 0.0,
                "false_positives": false_positives,
                "predicted_as_this_class": int((np.asarray(y_pred) == index).sum()),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _relative_path(path: Path, root: Path) -> str:
    """Ruta del modelo relativa al proyecto (o absoluta si esta fuera)."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def model_size_kb(path: Path) -> float:
    """Tamano del archivo .keras guardado, en KB (1 KB = 1024 bytes)."""
    return round(path.stat().st_size / 1024.0, 2)


def hardware_profile() -> dict:
    """Perfil del equipo de ejecucion (para la Model Card y el PDF)."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
        "gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
    }


class ValidationF1(tf.keras.callbacks.Callback):
    """Registra el F1 macro de validacion al final de cada epoca.

    Se evalua sobre el conjunto de validacion (nunca sobre la prueba), de modo
    que la curva de presupuesto no filtra informacion del conjunto de prueba.
    """

    def __init__(self, x_valid: np.ndarray, y_valid: np.ndarray) -> None:
        super().__init__()
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.f1_macro: list[float] = []

    def on_epoch_end(self, epoch, logs=None) -> None:
        predictions = self.model.predict(self.x_valid, verbose=0).argmax(axis=1)
        self.f1_macro.append(float(f1_score(self.y_valid, predictions, average="macro")))


def run_budget_study(
    epochs: int = 25,
    seed: int = DEFAULT_SEED,
    batch_size: int = 128,
    reports_dir: Path | None = None,
    models_dir: Path | None = None,
    verbose: int = 2,
) -> dict:
    """Estudio de presupuesto: como evoluciona cada modelo con mas epocas.

    Responde a la pregunta que dejan abiertas las curvas del experimento
    principal: si la diferencia entre el baseline denso y la CNN es una
    limitacion de la arquitectura o simplemente del numero de epocas. Se
    mantienen la particion y la semilla del experimento principal.
    """
    set_global_seed(seed)
    root = Path(__file__).resolve().parents[2]
    reports_dir = Path(reports_dir) if reports_dir else root / "reports"
    models_dir = Path(models_dir) if models_dir else root / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    x_train_all, y_train_all, x_test, y_test = load_fashion_mnist()
    x_train, y_train, x_valid, y_valid = split_train_validation(x_train_all, y_train_all)
    validate_images(x_train, y_train)
    validate_images(x_valid, y_valid)
    validate_images(x_test, y_test)

    study: dict = {
        "meta": {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "train_size": len(x_train),
            "valid_size": len(x_valid),
            "test_size": len(x_test),
            "hardware": hardware_profile(),
        },
        "models": {},
    }

    for name in ("dense", "cnn"):
        model_path = models_dir / f"budget_{name}.keras"
        factory = MODEL_BUILDERS[name]
        model = compile_model(factory(tf))
        tracker = ValidationF1(x_valid, y_valid)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
            tracker,
        ]
        start = perf_counter()
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )
        train_seconds = perf_counter() - start
        inference_median, inference_runs, probabilities = measure_inference(model, x_test, repeats=3)
        y_pred = probabilities.argmax(axis=1)
        study["models"][name] = {
            "f1_macro_test": float(f1_score(y_test, y_pred, average="macro")),
            "accuracy_test": float((y_pred == y_test).mean()),
            "parameters": int(model.count_params()),
            "model_size_kb": model_size_kb(model_path),
            "train_seconds": round(train_seconds, 3),
            "inference_ms_per_image": round(inference_median, 4),
            "inference_ms_per_image_runs": inference_runs,
            "epochs_completed": len(history.history["loss"]),
            "validation_f1_macro_per_epoch": [round(value, 4) for value in tracker.f1_macro],
            "history": history.history,
        }
        print(f"[presupuesto] {name}: F1 macro (prueba) = {study['models'][name]['f1_macro_test']:.4f}")

    (reports_dir / "budget_study.json").write_text(
        json.dumps(study, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return study


def run_experiment(
    epochs: int = 8,
    seed: int = DEFAULT_SEED,
    batch_size: int = 128,
    validation_size: int = DEFAULT_VALIDATION_SIZE,
    repeats: int = 3,
    reports_dir: Path | None = None,
    models_dir: Path | None = None,
    verbose: int = 2,
) -> dict:
    """Ejecuta el experimento completo y deja los artefactos en ``reports/``."""
    set_global_seed(seed)
    root = Path(__file__).resolve().parents[2]
    reports_dir = Path(reports_dir) if reports_dir else root / "reports"
    models_dir = Path(models_dir) if models_dir else root / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    x_train_all, y_train_all, x_test, y_test = load_fashion_mnist()
    x_train, y_train, x_valid, y_valid = split_train_validation(
        x_train_all, y_train_all, validation_size=validation_size
    )
    validate_images(x_train, y_train)
    validate_images(x_valid, y_valid)
    validate_images(x_test, y_test)

    audit = audit_dataset(
        {
            "entrenamiento": (x_train, y_train),
            "validacion": (x_valid, y_valid),
            "prueba": (x_test, y_test),
            "entrenamiento_completo": (x_train_all, y_train_all),
        }
    )

    metrics: dict = {}
    histories: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}
    probability_maps: dict[str, np.ndarray] = {}
    per_class: list[dict] = []
    errors_table: list[dict] = []

    for name in ("dense", "cnn"):
        model_path = models_dir / f"best_{name}.keras"
        model, history, train_seconds = train_model(
            name,
            x_train,
            y_train,
            x_valid,
            y_valid,
            epochs=epochs,
            batch_size=batch_size,
            model_path=model_path,
            verbose=verbose,
        )
        histories[name] = history.history
        inference_median, inference_runs, probabilities = measure_inference(model, x_test, repeats)
        y_pred = probabilities.argmax(axis=1)
        predictions[name] = y_pred
        probability_maps[name] = probabilities

        report = classification_report(
            y_test, y_pred, labels=list(range(NUM_CLASSES)), target_names=list(CLASS_NAMES), digits=4
        )
        per_class.extend(per_class_rows(y_test, y_pred, name))
        errors_table.extend(error_rows(y_test, y_pred, name))

        metrics[name] = {
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "parameters": int(model.count_params()),
            "train_seconds": round(train_seconds, 3),
            "inference_ms_per_image": round(inference_median, 4),
            "inference_ms_per_image_runs": inference_runs,
            "inference_repeats": repeats,
            "accuracy": float((y_pred == y_test).mean()),
            "epochs_requested": epochs,
            "epochs_completed": len(history.history["loss"]),
            "model_path": _relative_path(model_path, root),
            "model_size_kb": model_size_kb(model_path),
            "per_class": {row["class_name"]: {
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "support": row["support"],
            } for row in per_class_rows(y_test, y_pred, name)},
            "classification_report": report,
            "confusion_matrix": confusion_matrix(
                y_test, y_pred, labels=list(range(NUM_CLASSES))
            ).tolist(),
            "history": history.history,
        }
        print(f"\n=== {MODEL_TITLES[name]} ===")
        print(json.dumps({key: metrics[name][key] for key in (
            "f1_macro", "accuracy", "parameters", "train_seconds",
            "inference_ms_per_image", "epochs_completed", "model_size_kb")}, indent=2))

    # Curvas de entrenamiento y validacion de ambos modelos.
    plot_training_curves(histories, reports_dir / "training_curves.png")

    # Matrices de confusion (se conserva la del base y se anade la del denso).
    plot_confusion_matrix(
        y_test, predictions["cnn"], reports_dir / "confusion_cnn.png",
        "Matriz de confusión - CNN (conjunto de prueba)",
    )
    plot_confusion_matrix(
        y_test, predictions["dense"], reports_dir / "confusion_dense.png",
        "Matriz de confusión - baseline denso (conjunto de prueba)",
    )

    # Analisis de errores por clase.
    plot_errors_by_class(errors_table, reports_dir / "errors_by_class.png")

    # Ejemplos de error de la CNN elegidos al azar con semilla fija.
    wrong = np.flatnonzero(y_test != predictions["cnn"])
    selected = select_error_indices(y_test, predictions["cnn"], count=16, seed=seed)
    plot_error_grid(
        x_test, y_test, predictions["cnn"], selected, reports_dir / "cnn_errors.png"
    )
    sample_rows = []
    for index in selected:
        confidence = float(probability_maps["cnn"][index].max())
        sample_rows.append(
            {
                "test_index": int(index),
                "true_class": CLASS_NAMES[int(y_test[index])],
                "predicted_class": CLASS_NAMES[int(predictions["cnn"][index])],
                "confidence": round(confidence, 4),
                "true_probability": round(
                    float(probability_maps["cnn"][index][int(y_test[index])]), 4
                ),
            }
        )
    _write_csv(reports_dir / "cnn_error_samples.csv", sample_rows)

    _write_csv(reports_dir / "cv_per_class.csv", per_class)
    _write_csv(reports_dir / "cv_errors_by_class.csv", errors_table)

    summary = {
        "dense": {
            key: metrics["dense"][key]
            for key in ("f1_macro", "accuracy", "parameters", "model_size_kb",
                        "train_seconds", "inference_ms_per_image", "epochs_completed")
        },
        "cnn": {
            key: metrics["cnn"][key]
            for key in ("f1_macro", "accuracy", "parameters", "model_size_kb",
                        "train_seconds", "inference_ms_per_image", "epochs_completed")
        },
    }
    payload = {
        "meta": {
            "seed": seed,
            "epochs_requested": epochs,
            "batch_size": batch_size,
            "validation_size": validation_size,
            "train_size": len(x_train),
            "valid_size": len(x_valid),
            "test_size": len(x_test),
            "early_stopping": {"patience": 2, "restore_best_weights": True},
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy",
            "hardware": hardware_profile(),
        },
        "dataset": audit,
        "dense": metrics["dense"],
        "cnn": metrics["cnn"],
        "summary": summary,
    }
    (reports_dir / "cv_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (reports_dir / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nErrores totales de la CNN sobre la prueba: {len(wrong)} de {len(y_test)}")
    print(f"Artefactos escritos en {reports_dir}")
    return payload
