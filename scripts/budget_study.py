"""Estudio de presupuesto de entrenamiento (complemento del LAB07).

El experimento principal compara los dos modelos con el presupuesto que fija el
manual (8 epocas). Este script responde a la pregunta que dejan abiertas las
curvas: si la diferencia entre el baseline denso y la CNN es una limitacion de la
arquitectura o del numero de epocas. Mantiene la misma particion y la misma
semilla, y solo usa el conjunto de validacion para las curvas por epoca.

Uso:  uv run python scripts/budget_study.py --epochs 25

Salidas:
    reports/budget_study.json   F1 macro por epoca y metricas finales
    reports/budget_study.png    curva de presupuesto y comparacion
    models/budget_dense.keras   baseline denso con el presupuesto ampliado
    models/budget_cnn.keras     CNN con el presupuesto ampliado
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inf8239_u02_cv.experiment import run_budget_study
from inf8239_u02_cv.figures import plot_budget_study

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25, help="epocas maxima por modelo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    study = run_budget_study(
        epochs=args.epochs,
        seed=args.seed,
        batch_size=args.batch_size,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        verbose=0 if args.quiet else 2,
    )

    baseline = {}
    metrics_path = Path(args.reports_dir) / "cv_metrics.json"
    if metrics_path.exists():
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        baseline = {name: previous[name]["f1_macro"] for name in ("dense", "cnn")}

    if baseline:
        plot_budget_study(study, baseline, Path(args.reports_dir) / "budget_study.png")
        print("Figura escrita en reports/budget_study.png")

    print("\nEstudio de presupuesto:")
    print(json.dumps(
        {
            name: {
                "f1_macro_test": study["models"][name]["f1_macro_test"],
                "accuracy_test": study["models"][name]["accuracy_test"],
                "parameters": study["models"][name]["parameters"],
                "model_size_kb": study["models"][name]["model_size_kb"],
                "train_seconds": study["models"][name]["train_seconds"],
                "epochs_completed": study["models"][name]["epochs_completed"],
            }
            for name in ("dense", "cnn")
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
