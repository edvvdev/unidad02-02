"""El notebook de la entrega debe estar ejecutado y sin celdas con error."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_ejercicio04_cnn_model_card.ipynb"


@pytest.mark.skipif(not NOTEBOOK.exists(), reason="El notebook aun no se ha generado")
def test_notebook_has_no_error_outputs():
    datos = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    errores = []
    for indice, celda in enumerate(datos.get("cells", [])):
        for salida in celda.get("outputs", []):
            if salida.get("output_type") == "error":
                errores.append((indice, salida.get("ename"), salida.get("evalue")))
    assert not errores, f"Celdas con error en el notebook: {errores}"


@pytest.mark.skipif(not NOTEBOOK.exists(), reason="El notebook aun no se ha generado")
def test_every_code_cell_has_been_executed():
    datos = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    codigo = [celda for celda in datos.get("cells", []) if celda.get("cell_type") == "code"]
    assert codigo, "El notebook no tiene celdas de codigo"
    sin_ejecutar = [celda.get("execution_count") for celda in codigo if not celda.get("execution_count")]
    assert not sin_ejecutar, f"Celdas de codigo sin ejecutar: {sin_ejecutar}"


@pytest.mark.skipif(not NOTEBOOK.exists(), reason="El notebook aun no se ha generado")
def test_notebook_keeps_the_expected_sections():
    datos = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    texto = "\n".join(
        "".join(celda.get("source", [])) for celda in datos.get("cells", []) if celda.get("cell_type") == "markdown"
    )
    for seccion in (
        "Auditoría del dataset",
        "partición reproducible",
        "Curvas de entrenamiento y validación",
        "Métricas por clase",
        "Matriz de confusión",
        "Errores por clase",
        "Decisión técnica",
    ):
        assert seccion in texto, f"Falta la seccion: {seccion}"
