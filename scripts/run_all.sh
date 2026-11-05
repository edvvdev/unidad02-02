#!/usr/bin/env bash
# Pipeline completo del Ejercicio 04 (U02.LAB07).
# Uso:  bash scripts/run_all.sh
#
# Es reproducible desde un clon limpio: no necesita pasos manuales.
# Orden: entorno -> check_runtime -> entrenamiento -> notebook -> PDF -> pruebas.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EPOCHS="${EPOCHS:-8}"
BUDGET_EPOCHS="${BUDGET_EPOCHS:-25}"
export REPO_URL="${REPO_URL:-https://github.com/edvvdev/unidad02-02}"

echo "==> 1/9 Entorno (uv sync --extra cpu)"
uv sync --extra cpu

echo "==> 2/9 Dependencias de trabajo (jupyter, nbformat, ipykernel, reportlab)"
# Son dependencias de trabajo: no estan en pyproject.toml porque el enunciado
# prohibe modificarlo. `uv sync` elimina lo que no esta en el lock, por eso se
# instalan DESPUES de la sincronizacion y antes de usar el notebook y el PDF.
uv pip install --python .venv/bin/python jupyter nbformat ipykernel reportlab

echo "==> 3/9 Comprobacion del runtime (scripts/check_runtime.py)"
uv run python scripts/check_runtime.py

echo "==> 4/9 Entrenamiento comparativo: baseline denso y CNN (--epochs ${EPOCHS})"
uv run python scripts/train_cv.py --epochs "${EPOCHS}"

echo "==> 5/9 Estudio de presupuesto (complemento, --epochs ${BUDGET_EPOCHS})"
uv run python scripts/budget_study.py --epochs "${BUDGET_EPOCHS}"

echo "==> 6/9 Ejecucion del notebook con salidas guardadas"
uv run python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 notebooks/01_ejercicio04_cnn_model_card.ipynb

echo "==> 7/9 PDF breve del ejercicio (docs/EJERCICIO_04_PDF_BRIEF.pdf)"
uv run python scripts/build_ejercicio04_pdf.py

echo "==> 8/9 Pruebas del proyecto"
uv run pytest -q

echo "==> 9/9 Verificacion de las cifras publicadas contra reports/"
uv run python scripts/verify_cifras.py

echo
echo "Pipeline completo. Artefactos en reports/, models/ y docs/EJERCICIO_04_PDF_BRIEF.pdf"
