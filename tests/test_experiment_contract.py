"""Contratos del experimento LAB07: particion, metricas por clase y artefactos."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from inf8239_u02_cv.data import (
    CLASS_NAMES,
    DEFAULT_SEED,
    DEFAULT_VALIDATION_SIZE,
    NUM_CLASSES,
    audit_dataset,
    class_distribution,
    normalize_images,
    split_train_validation,
    validate_images,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _fake_block(n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(DEFAULT_SEED)
    images = generator.integers(0, 256, size=(n, 28, 28), dtype=np.uint8)
    labels = np.arange(n) % NUM_CLASSES
    return images, labels


def test_split_is_deterministic_and_respects_sizes():
    images, labels = _fake_block(1000)
    x_train, y_train, x_valid, y_valid = split_train_validation(images, labels, 200)
    assert x_train.shape == (800, 28, 28)
    assert x_valid.shape == (200, 28, 28)
    assert len(y_train) == 800 and len(y_valid) == 200
    otra = split_train_validation(images, labels, 200)
    assert np.array_equal(x_train, otra[0]) and np.array_equal(y_valid, otra[3])


def test_split_uses_the_tail_of_the_training_block():
    images, labels = _fake_block(1000)
    x_train, _, x_valid, _ = split_train_validation(images, labels, 200)
    assert np.array_equal(x_valid, images[-200:])
    assert not np.shares_memory(x_train, x_valid)


def test_split_rejects_invalid_validation_size():
    images, labels = _fake_block(50)
    with pytest.raises(ValueError, match="validation_size"):
        split_train_validation(images, labels, 0)
    with pytest.raises(ValueError, match="validation_size"):
        split_train_validation(images, labels, 50)


def test_default_validation_size_matches_the_laboratory():
    assert DEFAULT_VALIDATION_SIZE == 6000
    assert DEFAULT_SEED == 42


def test_class_names_cover_the_ten_classes():
    assert NUM_CLASSES == 10
    assert len(CLASS_NAMES) == 10
    assert len(set(CLASS_NAMES)) == 10


def test_class_distribution_counts_every_example():
    labels = np.arange(50) % NUM_CLASSES
    counts = class_distribution(labels)
    assert sum(counts.values()) == 50
    assert set(counts) == set(CLASS_NAMES)


def test_normalize_keeps_range_and_adds_channel():
    images, _ = _fake_block(20)
    normalized = normalize_images(images)
    assert normalized.shape == (20, 28, 28, 1)
    assert normalized.dtype == np.float32
    assert 0.0 <= float(normalized.min()) and float(normalized.max()) <= 1.0
    validate_images(normalized, np.zeros(20))


def test_audit_dataset_reports_shapes_and_distribution():
    images, labels = _fake_block(100)
    normalized = normalize_images(images)
    audit = audit_dataset({"prueba": (normalized, labels)})
    assert audit["num_classes"] == NUM_CLASSES
    assert audit["splits"]["prueba"]["n_samples"] == 100
    assert audit["splits"]["prueba"]["image_shape"] == [28, 28, 1]
    assert sum(audit["splits"]["prueba"]["class_counts"].values()) == 100
    assert audit["splits"]["prueba"]["imbalance_ratio_max_over_min"] >= 1.0


def test_per_class_rows_and_error_rows_are_consistent():
    from inf8239_u02_cv.experiment import error_rows, per_class_rows

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 0, 2])
    per_class = per_class_rows(y_true, y_pred, "cnn")
    errores = error_rows(y_true, y_pred, "cnn")
    assert len(per_class) == NUM_CLASSES and len(errores) == NUM_CLASSES
    for fila in per_class:
        assert fila["support"] == int((y_true == fila["class_id"]).sum())
    for fila in errores:
        support = int((y_true == fila["class_id"]).sum())
        assert fila["support"] == support
        assert fila["errors"] + fila["correct"] == support
        if support:
            assert abs(fila["recall"] - fila["correct"] / support) < 1e-3
            assert abs(fila["error_rate"] + fila["recall"] - 1.0) < 1e-3
        else:
            assert fila["recall"] == 0.0 and fila["error_rate"] == 0.0


def test_error_selection_is_random_but_reproducible():
    from inf8239_u02_cv.figures import select_error_indices

    y_true = np.array([0] * 20 + [1] * 20)
    y_pred = np.array([1] * 20 + [1] * 20)
    indices = select_error_indices(y_true, y_pred, count=8, seed=42)
    assert len(indices) == 8
    assert np.all(y_true[indices] != y_pred[indices])
    assert np.array_equal(indices, select_error_indices(y_true, y_pred, count=8, seed=42))
    assert not np.array_equal(indices, select_error_indices(y_true, y_pred, count=8, seed=7))


def test_measure_inference_returns_the_median_of_the_runs():
    from inf8239_u02_cv.experiment import measure_inference

    class FakeModel:
        def predict(self, x, verbose=0):
            return np.zeros((len(x), NUM_CLASSES), dtype="float32")

    mediana, runs, probabilities = measure_inference(FakeModel(), np.zeros((10, 28, 28, 1)), repeats=3)
    assert len(runs) == 3
    assert mediana == pytest.approx(sorted(runs)[1], abs=1e-3)
    assert probabilities.shape == (10, NUM_CLASSES)


def test_model_size_kb_is_measured_from_the_file(tmp_path):
    from inf8239_u02_cv.experiment import model_size_kb

    archivo = tmp_path / "modelo.keras"
    archivo.write_bytes(b"0" * 2048)
    assert model_size_kb(archivo) == 2.0


@pytest.mark.skipif(
    not (REPORTS / "cv_metrics.json").exists(),
    reason="Requiere una ejecucion previa de scripts/train_cv.py",
)
def test_reports_are_internally_consistent():
    metricas = json.loads((REPORTS / "cv_metrics.json").read_text(encoding="utf-8"))
    for nombre in ("dense", "cnn"):
        bloque = metricas[nombre]
        assert 0.0 <= bloque["f1_macro"] <= 1.0
        assert 0.0 <= bloque["accuracy"] <= 1.0
        assert bloque["parameters"] > 0
        assert bloque["model_size_kb"] > 0
        assert len(bloque["inference_ms_per_image_runs"]) == bloque["inference_repeats"]
        assert bloque["inference_ms_per_image"] == pytest.approx(
            sorted(bloque["inference_ms_per_image_runs"])[len(bloque["inference_ms_per_image_runs"]) // 2],
            abs=1e-3,
        )
        matriz = np.array(bloque["confusion_matrix"])
        assert matriz.shape == (NUM_CLASSES, NUM_CLASSES)
        assert matriz.sum() == metricas["meta"]["test_size"]
        assert len(bloque["per_class"]) == NUM_CLASSES
        assert len(bloque["history"]["loss"]) == bloque["epochs_completed"]


@pytest.mark.skipif(
    not (REPORTS / "cv_per_class.csv").exists(),
    reason="Requiere una ejecucion previa de scripts/train_cv.py",
)
def test_per_class_csv_matches_the_json():
    import csv

    metricas = json.loads((REPORTS / "cv_metrics.json").read_text(encoding="utf-8"))
    with (REPORTS / "cv_per_class.csv").open(encoding="utf-8") as archivo:
        filas = list(csv.DictReader(archivo))
    assert len(filas) == 2 * NUM_CLASSES
    for fila in filas:
        esperado = metricas[fila["model"]]["per_class"][fila["class_name"]]
        assert float(fila["recall"]) == pytest.approx(esperado["recall"], abs=1e-3)
        assert float(fila["f1"]) == pytest.approx(esperado["f1"], abs=1e-3)
        assert int(fila["support"]) == esperado["support"]


@pytest.mark.skipif(
    not (REPORTS / "cv_errors_by_class.csv").exists(),
    reason="Requiere una ejecucion previa de scripts/train_cv.py",
)
def test_errors_by_class_csv_is_consistent_with_the_confusion_matrix():
    import csv

    metricas = json.loads((REPORTS / "cv_metrics.json").read_text(encoding="utf-8"))
    with (REPORTS / "cv_errors_by_class.csv").open(encoding="utf-8") as archivo:
        filas = list(csv.DictReader(archivo))
    for fila in filas:
        indice = int(fila["class_id"])
        matriz = np.array(metricas[fila["model"]]["confusion_matrix"])
        errores = int(matriz[indice].sum() - matriz[indice, indice])
        assert int(fila["errors"]) == errores
        assert int(fila["correct"]) == int(matriz[indice, indice])
        assert int(fila["support"]) == int(matriz[indice].sum())
