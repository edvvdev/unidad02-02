# Pipeline completo del Ejercicio 04 (U02.LAB07) en Windows PowerShell.
# Uso:  powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
#
# Es reproducible desde un clon limpio: no necesita pasos manuales.
# Orden: entorno -> check_runtime -> entrenamiento -> notebook -> PDF -> pruebas.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Epochs = if ($env:EPOCHS) { $env:EPOCHS } else { "8" }
$BudgetEpochs = if ($env:BUDGET_EPOCHS) { $env:BUDGET_EPOCHS } else { "25" }
if (-not $env:REPO_URL) { $env:REPO_URL = "https://github.com/edvvdev/unidad02-02" }

Write-Host "==> 1/9 Entorno (uv sync --extra cpu)"
uv sync --extra cpu

Write-Host "==> 2/9 Dependencias de trabajo (jupyter, nbformat, ipykernel, reportlab)"
# Son dependencias de trabajo: no estan en pyproject.toml porque el enunciado
# prohibe modificarlo. `uv sync` elimina lo que no esta en el lock, por eso se
# instalan DESPUES de la sincronizacion y antes de usar el notebook y el PDF.
uv pip install --python .venv/bin/python jupyter nbformat ipykernel reportlab

Write-Host "==> 3/9 Comprobacion del runtime (scripts/check_runtime.py)"
uv run python scripts/check_runtime.py

Write-Host "==> 4/9 Entrenamiento comparativo: baseline denso y CNN (--epochs $Epochs)"
uv run python scripts/train_cv.py --epochs $Epochs

Write-Host "==> 5/9 Estudio de presupuesto (complemento, --epochs $BudgetEpochs)"
uv run python scripts/budget_study.py --epochs $BudgetEpochs

Write-Host "==> 6/9 Ejecucion del notebook con salidas guardadas"
uv run python -m jupyter nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=3600 notebooks/01_ejercicio04_cnn_model_card.ipynb

Write-Host "==> 7/9 PDF breve del ejercicio (docs/EJERCICIO_04_PDF_BRIEF.pdf)"
uv run python scripts/build_ejercicio04_pdf.py

Write-Host "==> 8/9 Pruebas del proyecto"
uv run pytest -q

Write-Host "==> 9/9 Verificacion de las cifras publicadas contra reports/"
uv run python scripts/verify_cifras.py

Write-Host "Pipeline completo. Artefactos en reports/, models/ y docs/EJERCICIO_04_PDF_BRIEF.pdf"
