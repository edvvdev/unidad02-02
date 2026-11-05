"""Entrena el baseline denso y la CNN con la misma particion y semilla.

Uso (manual LAB07):
    uv run python scripts/train_cv.py --epochs 8

Salidas:
    reports/cv_metrics.json        metricas globales, por clase e historiales
    reports/cv_per_class.csv        precision, recall, F1 y soporte por clase
    reports/cv_errors_by_class.csv  errores, falsos positivos y recall por clase
    reports/dataset_audit.json      auditoria del dataset y reparto por clase
    reports/training_curves.png     curvas de entrenamiento y validacion
    reports/confusion_cnn.png       matriz de confusion de la CNN
    reports/confusion_dense.png     matriz de confusion del baseline denso
    reports/cnn_errors.png          ejemplos de error de la CNN (semilla fija)
    reports/errors_by_class.png     recall y errores por clase
    reports/cnn_error_samples.csv   ficha de los ejemplos de error elegidos
    models/best_dense.keras         baseline denso entrenado
    models/best_cnn.keras           CNN entrenada
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inf8239_u02_cv.experiment import run_experiment

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8, help="epocas maxima por modelo")
    parser.add_argument("--seed", type=int, default=42, help="semilla de Python, NumPy y TensorFlow")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3, help="repeticiones de la medicion de inferencia")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--quiet", action="store_true", help="silencia el log de epochs de Keras")
    args = parser.parse_args()

    payload = run_experiment(
        epochs=args.epochs,
        seed=args.seed,
        batch_size=args.batch_size,
        repeats=args.repeats,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        verbose=0 if args.quiet else 2,
    )

    print("\nResumen comparativo (misma particion, misma semilla):")
    print(json.dumps(payload["summary"], indent=2))
    print("\nClassification report de la CNN:")
    print(payload["cnn"]["classification_report"])


if __name__ == "__main__":
    main()
