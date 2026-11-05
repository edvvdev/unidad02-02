"""Comprueba que todas las cifras publicadas coinciden con los artefactos.

El requisito de la entrega es que ninguna cifra del README, de la Model Card o
del PDF breve exista sin respaldo. Este script lee `reports/` y verifica que los
documentos publicados contienen esos mismos valores.

Criterio:
* Las metricas reproducibles (F1 macro, accuracy, parametros) deben aparecer
  **exactamente** con el mismo formato en los documentos.
* El tamano del archivo y los tiempos dependen del entorno de ejecucion: se
  aceptan como correctos si estan dentro de un margen pequeno.
* El PDF debe tener entre 4 y 6 paginas, identificar al estudiante y el
  repositorio, y no debe contener correo institucional ni matricula.

Uso:  uv run python scripts/verify_cifras.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
PDF = DOCS / "EJERCICIO_04_PDF_BRIEF.pdf"
README = ROOT / "README.md"
MODEL_CARD = ROOT / "MODEL_CARD.md"
NOTebook = ROOT / "notebooks" / "01_ejercicio04_cnn_model_card.ipynb"

STUDENT = "Edwing Perez"
DEFAULT_REPO = "https://github.com/edvvdev/unidad02-02"
SIZE_TOLERANCE_KB = 2.0
TIME_TOLERANCE = 0.25


def fail(mensajes: list[str], texto: str) -> None:
    if mensajes:
        print("FALLO:")
        for mensaje in mensajes:
            print(f"  - {mensaje}")
        sys.exit(1)
    print(f"OK: {texto}")


def numeros_en_texto(texto: str) -> set[float]:
    return {float(valor.replace(",", ".")) for valor in re.findall(r"\d+[.,]\d+", texto)}


def comprobar_documento(nombre: str, texto: str, metricas: dict, tolerancias: dict) -> list[str]:
    problemas: list[str] = []
    numeros = numeros_en_texto(texto)
    for modelo in ("dense", "cnn"):
        bloque = metricas[modelo]
        for clave, decimales in (("f1_macro", 4), ("accuracy", 4)):
            valor = round(bloque[clave], decimales)
            if valor not in numeros:
                problemas.append(
                    f"{nombre}: no aparece {clave} de {modelo} = {valor:.{decimales}f}"
                )
        if not re.search(rf"\b{bloque['parameters']}\b", texto):
            problemas.append(f"{nombre}: no aparecen los parametros de {modelo} ({bloque['parameters']})")
    for modelo in ("dense", "cnn"):
        bloque = metricas[modelo]
        for clave, tolerancia, unidad in tolerancias:
            valor = bloque[clave]
            if not any(abs(valor - n) <= tolerancia for n in numeros):
                problemas.append(
                    f"{nombre}: no aparece {clave} de {modelo} = {valor} {unidad} "
                    f"(margen {tolerancia})"
                )
    return problemas


def main() -> None:
    problemas: list[str] = []

    metricas_path = REPORTS / "cv_metrics.json"
    if not metricas_path.exists():
        print("FALLO: falta reports/cv_metrics.json; ejecuta scripts/train_cv.py primero.")
        sys.exit(1)
    metricas = json.loads(metricas_path.read_text(encoding="utf-8"))

    tolerancias = (
        ("model_size_kb", SIZE_TOLERANCE_KB, "KB"),
        ("train_seconds", metricas["dense"]["train_seconds"] * TIME_TOLERANCE + 1.0, "s"),
        ("inference_ms_per_image", metricas["dense"]["inference_ms_per_image"] * TIME_TOLERANCE + 0.01, "ms/img"),
    )

    for documento in (README, MODEL_CARD):
        if not documento.exists():
            problemas.append(f"falta {documento.name}")
            continue
        problemas += comprobar_documento(
            documento.name, documento.read_text(encoding="utf-8"), metricas, tolerancias
        )

    if not PDF.exists():
        problemas.append("falta docs/EJERCICIO_04_PDF_BRIEF.pdf")
    else:
        texto_pdf = subprocess.run(
            ["pdftotext", "-layout", str(PDF), "-"],
            capture_output=True, text=True, check=False,
        ).stdout
        if STUDENT not in texto_pdf:
            problemas.append("el PDF no menciona al estudiante")
        if DEFAULT_REPO not in texto_pdf:
            problemas.append("el PDF no contiene el enlace al repositorio")
        for prohibido in ("Correo institucional", "matrícula", "matricula"):
            if prohibido.lower() in texto_pdf.lower():
                problemas.append(f"el PDF contiene un termino prohibido: {prohibido}")
        paginas = subprocess.run(
            ["pdfinfo", str(PDF)], capture_output=True, text=True, check=False
        ).stdout
        total = re.search(r"Pages:\s+(\d+)", paginas)
        if total is None or not 4 <= int(total.group(1)) <= 6:
            problemas.append("el PDF debe tener entre 4 y 6 paginas")
        problemas += comprobar_documento("PDF", texto_pdf, metricas, tolerancias)

    if not NOTebook.exists():
        problemas.append("falta el notebook")
    else:
        datos = json.loads(NOTebook.read_text(encoding="utf-8"))
        errores = [
            salida
            for celda in datos.get("cells", [])
            for salida in celda.get("outputs", [])
            if salida.get("output_type") == "error"
        ]
        if errores:
            problemas.append(f"el notebook tiene {len(errores)} celdas con error")
        sin_ejecutar = [
            celda
            for celda in datos.get("cells", [])
            if celda.get("cell_type") == "code" and not celda.get("execution_count")
        ]
        if sin_ejecutar:
            problemas.append(f"el notebook tiene {len(sin_ejecutar)} celdas de codigo sin ejecutar")

    fail(problemas, "todas las cifras publicadas coinciden con reports/")


if __name__ == "__main__":
    main()
