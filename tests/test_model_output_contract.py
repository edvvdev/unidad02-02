"""Contrato de salida de los dos modelos: forma, clases y probabilidades."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from inf8239_u02_cv.data import NUM_CLASSES
from inf8239_u02_cv.models import build_cnn, build_dense


@pytest.mark.parametrize("constructor", [build_dense, build_cnn])
def test_output_dimension_matches_the_number_of_classes(constructor):
    model = constructor(tf)
    probabilidades = model(np.zeros((3, 28, 28, 1), dtype="float32")).numpy()
    assert probabilidades.shape == (3, NUM_CLASSES)


@pytest.mark.parametrize("constructor", [build_dense, build_cnn])
def test_probabilities_are_finite_and_sum_to_one(constructor):
    model = constructor(tf)
    entrada = np.full((2, 28, 28, 1), 0.5, dtype="float32")
    probabilidades = model(entrada).numpy()
    assert np.isfinite(probabilidades).all()
    assert (probabilidades >= 0).all() and (probabilidades <= 1).all()
    np.testing.assert_allclose(probabilidades.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("constructor", [build_dense, build_cnn])
def test_models_accept_the_normalized_input_shape(constructor):
    model = constructor(tf)
    assert model.input_shape == (None, 28, 28, 1)
    assert model.output_shape == (None, NUM_CLASSES)


def test_cnn_is_smaller_than_the_dense_baseline_in_parameters():
    denso = build_dense(tf)
    cnn = build_cnn(tf)
    assert cnn.count_params() < denso.count_params()


def test_dense_baseline_flattens_and_cnn_uses_convolutions():
    denso = build_dense(tf)
    cnn = build_cnn(tf)
    assert any(capa.__class__.__name__ == "Flatten" for capa in denso.layers)
    assert any(capa.__class__.__name__ == "Conv2D" for capa in cnn.layers)
    assert not any(capa.__class__.__name__ == "Conv2D" for capa in denso.layers)
