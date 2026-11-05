from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from .data import CLASS_NAMES

MODEL_TITLES = {"dense": "Baseline denso", "cnn": "CNN"}
ERROR_SEED = 42


def _save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return path


def _figure(rows: int, columns: int, figsize: tuple[float, float]):
    """Crea una figura con distribucion constrained: evita solapes de titulos."""
    return plt.subplots(rows, columns, figsize=figsize, layout="constrained")


def plot_training_curves(histories: dict[str, dict], path: Path) -> Path:
    """Curvas de perdida y accuracy de entrenamiento y validacion de ambos modelos."""
    figure, axes = _figure(2, 2, (12, 8))
    for column, name in enumerate(("dense", "cnn")):
        history = histories[name]
        epochs = list(range(1, len(history["loss"]) + 1))
        axes[0, column].plot(epochs, history["loss"], marker="o", label="entrenamiento")
        axes[0, column].plot(epochs, history["val_loss"], marker="s", label="validacion")
        axes[0, column].set_title(f"{MODEL_TITLES[name]} - perdida")
        axes[0, column].set_xlabel("epoca")
        axes[0, column].set_ylabel("perdida (cross-entropy)")
        axes[0, column].legend()
        axes[0, column].grid(alpha=0.3)

        axes[1, column].plot(epochs, history["accuracy"], marker="o", label="entrenamiento")
        axes[1, column].plot(epochs, history["val_accuracy"], marker="s", label="validacion")
        axes[1, column].set_title(f"{MODEL_TITLES[name]} - accuracy")
        axes[1, column].set_xlabel("epoca")
        axes[1, column].set_ylabel("accuracy")
        axes[1, column].legend()
        axes[1, column].grid(alpha=0.3)

    figure.suptitle("Curvas de entrenamiento y validacion (misma particion, semilla 42)")
    return _save(figure, path)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str) -> Path:
    """Matriz de confusion con los nombres reales de las clases."""
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=CLASS_NAMES,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d",
    )
    figure = display.figure_
    figure.set_size_inches(10, 9)
    for label in figure.axes[0].get_xticklabels():
        label.set_fontsize(8)
    for label in figure.axes[0].get_yticklabels():
        label.set_fontsize(8)
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return path


def select_error_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    count: int = 16,
    seed: int = ERROR_SEED,
) -> np.ndarray:
    """Elige errores **al azar con semilla fija** (reproducible entre ejecuciones)."""
    wrong = np.flatnonzero(np.asarray(y_true) != np.asarray(y_pred))
    if len(wrong) == 0:
        return wrong
    generator = np.random.default_rng(seed)
    size = min(count, len(wrong))
    return np.sort(generator.choice(wrong, size=size, replace=False))


def plot_error_grid(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    indices: np.ndarray,
    path: Path,
    title: str = "Errores de la CNN (16 ejemplos al azar con semilla 42)",
) -> Path:
    """Rejilla con los ejemplos mal clasificados: clase real frente a prediccion."""
    figure, axes = _figure(4, 4, (9, 9))
    for axis, index in zip(axes.ravel(), indices):
        axis.imshow(np.asarray(images[index]).squeeze(), cmap="gray")
        axis.set_title(
            f"Real: {CLASS_NAMES[int(y_true[index])]}\nPred: {CLASS_NAMES[int(y_pred[index])]}",
            fontsize=8,
        )
        axis.axis("off")
    for axis in axes.ravel()[len(indices):]:
        axis.axis("off")
    figure.suptitle(title)
    return _save(figure, path)


def plot_errors_by_class(rows: list[dict], path: Path) -> Path:
    """Recall por clase y errores por clase de cada modelo."""
    names = [row["class_name"] for row in rows if row["model"] == "dense"]
    dense = [row for row in rows if row["model"] == "dense"]
    cnn = [row for row in rows if row["model"] == "cnn"]
    position = np.arange(len(names))
    width = 0.38

    figure, axes = _figure(1, 2, (15, 5.5))

    axes[0].bar(position - width / 2, [row["recall"] for row in dense], width, label="Baseline denso")
    axes[0].bar(position + width / 2, [row["recall"] for row in cnn], width, label="CNN")
    axes[0].set_title("Recall por clase (conjunto de prueba)")
    axes[0].set_ylabel("recall")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(position)
    axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(position - width / 2, [row["errors"] for row in dense], width, label="Baseline denso")
    axes[1].bar(position + width / 2, [row["errors"] for row in cnn], width, label="CNN")
    axes[1].set_title("Errores por clase (clase real mal predicha)")
    axes[1].set_ylabel("numero de errores")
    axes[1].set_xticks(position)
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    figure.suptitle("Analisis de errores por clase")
    return _save(figure, path)


def plot_budget_study(study: dict, baseline_f1: dict, path: Path, prescribed_epochs: int = 8) -> Path:
    """Curva de F1 macro de validacion por epoca y comparacion de presupuestos."""
    figure, axes = _figure(1, 2, (14, 5.4))

    for name, color in (("dense", "#4C72B0"), ("cnn", "#DD8452")):
        values = study["models"][name]["validation_f1_macro_per_epoch"]
        axes[0].plot(range(1, len(values) + 1), values, marker="o", markersize=3,
                     color=color, label=MODEL_TITLES[name])
    axes[0].axvline(prescribed_epochs, color="black", linestyle="--", linewidth=1)
    axes[0].annotate(f"presupuesto del manual\n({prescribed_epochs} epocas)",
                     xy=(prescribed_epochs, 0.02), xytext=(prescribed_epochs + 1.5, 0.05),
                     fontsize=8)
    axes[0].set_title("F1 macro de validacion por epoca")
    axes[0].set_xlabel("epoca")
    axes[0].set_ylabel("F1 macro (validacion)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    names = ["dense", "cnn"]
    positions = np.arange(len(names))
    width = 0.36
    axes[1].bar(positions - width / 2, [baseline_f1[name] for name in names], width,
                 label=f"{prescribed_epochs} epocas (manual)")
    axes[1].bar(positions + width / 2,
                [study["models"][name]["f1_macro_test"] for name in names], width,
                label=f"{study['meta']['epochs']} epocas (estudio de presupuesto)")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([MODEL_TITLES[name] for name in names], fontsize=9)
    axes[1].set_ylabel("F1 macro (conjunto de prueba)")
    axes[1].set_title("Efecto del presupuesto de entrenamiento")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    figure.suptitle("Estudio de presupuesto de entrenamiento")
    return _save(figure, path)
