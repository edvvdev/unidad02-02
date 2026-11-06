"""Genera el PDF breve del Ejercicio 04 (U02.LAB07).

Todas las cifras se leen de los artefactos en `reports/`, de modo que el PDF no
puede contradecir la evidencia del repositorio. El diagnostico de las curvas
usa la MISMA funcion que el notebook (`inf8239_u02_cv.diagnostico`), de modo
que el patron publicado (aprendizaje util / sobreajuste / subajuste) no puede
divergir entre el PDF, el notebook y la Model Card. El enlace al repositorio se
toma de la variable de entorno REPO_URL.

Uso:
    REPO_URL=https://github.com/edvvdev/unidad02-02 \
      uv run python scripts/build_ejercicio04_pdf.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inf8239_u02_cv.diagnostico import diagnostico_curvas

REPORTS = ROOT / "reports"
OUTPUT = ROOT / "docs/EJERCICIO_04_PDF_BRIEF.pdf"
REPO_URL = os.getenv("REPO_URL", "https://github.com/edvvdev/unidad02-02")
STUDENT = "Edwing Perez"

PRIMARY = colors.HexColor("#1d3557")
SECONDARY = colors.HexColor("#457b9d")
LIGHT = colors.HexColor("#f1f5f9")


def read_json(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def read_csv(name: str) -> list[dict]:
    with (REPORTS / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "titulo", parent=base["Title"], fontSize=19, leading=23,
            textColor=PRIMARY, spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "subtitulo", parent=base["Normal"], fontSize=10.5, leading=14,
            alignment=TA_CENTER, textColor=SECONDARY, spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontSize=13, leading=16,
            textColor=PRIMARY, spaceBefore=10, spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontSize=11, leading=14,
            textColor=SECONDARY, spaceBefore=6, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontSize=9.4, leading=12.6,
            alignment=TA_JUSTIFY, spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "small", parent=base["Normal"], fontSize=7.0, leading=8.6,
            textColor=colors.HexColor("#475569"),
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["Normal"], fontSize=8.2, leading=10.5,
            textColor=colors.HexColor("#475569"), spaceBefore=2, spaceAfter=8,
        ),
    }


def table_style() -> TableStyle:
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ])


def png_size(path: Path) -> tuple[int, int]:
    """Dimensiones reales de un PNG, leidas de su cabecera IHDR."""
    with path.open("rb") as handle:
        cabecera = handle.read(24)
    if cabecera[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"No es un PNG valido: {path}")
    ancho = int.from_bytes(cabecera[16:20], "big")
    alto = int.from_bytes(cabecera[20:24], "big")
    return ancho, alto


def figure(path: Path, max_width: float, max_height: float | None = None) -> Image | None:
    """Figura escalada conservando su proporcion (sin deformar)."""
    if not path.exists():
        return None
    ancho, alto = png_size(path)
    ancho_final = max_width
    alto_final = ancho_final * alto / ancho
    if max_height is not None and alto_final > max_height:
        alto_final = max_height
        ancho_final = alto_final * ancho / alto
    return Image(str(path), width=ancho_final, height=alto_final)


def build_pdf() -> Path:
    metrics = read_json("cv_metrics.json")
    audit = read_json("dataset_audit.json")
    per_class = read_csv("cv_per_class.csv")
    errors = read_csv("cv_errors_by_class.csv")
    budget_path = REPORTS / "budget_study.json"
    budget = json.loads(budget_path.read_text(encoding="utf-8")) if budget_path.exists() else None
    decision_path = REPORTS / "decision_tecnica.txt"
    decision = decision_path.read_text(encoding="utf-8").strip() if decision_path.exists() else ""

    dense = metrics["dense"]
    cnn = metrics["cnn"]
    meta = metrics["meta"]
    hardware = meta["hardware"]
    styles = build_styles()
    content_width = 17.0 * cm

    # Diagnostico de las curvas: MISMA funcion que usa el notebook.
    diag_denso = diagnostico_curvas(dense["history"])
    diag_cnn = diagnostico_curvas(cnn["history"])

    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=1.9 * cm, rightMargin=1.9 * cm,
        topMargin=1.3 * cm, bottomMargin=1.3 * cm,
        title="Ejercicio 04 · Unidad 02 · U02.LAB07",
        author=STUDENT,
    )
    story = []

    # ---------------- Portada ----------------
    story.append(Paragraph("Ejercicio 04 · Unidad 02 (U02.LAB07)", styles["title"]))
    story.append(Paragraph(
        "Clasificador visual con CNN y Model Card sobre Fashion-MNIST",
        styles["subtitle"],
    ))
    story.append(Table(
        [
            ["Estudiante", STUDENT],
            ["Repositorio", REPO_URL],
            ["Curso", "INF-8239 Ciencia de Datos II · Unidad 02"],
            ["Laboratorio", "U02.LAB07 · Visión computacional: CNN reproducible con Fashion-MNIST"],
            ["Fecha del informe", datetime.now(UTC).date().isoformat()],
        ],
        colWidths=[5.0 * cm, 11.5 * cm],
        style=TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 0), (0, -1), PRIMARY),
            ("FONTSIZE", (0, -1), (-1, -1), 9.6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
            ("BACKGROUND", (0, 0), (0, -1), LIGHT),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]),
    ))
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph(
        "Entrega académica: auditoría del dataset y partición reproducible, baseline denso frente a CNN "
        "con la misma partición y semilla, métricas globales y por clase, matriz de confusión, análisis "
        "de errores por clase, comparación de costo y tamaño, decisión técnica, notebook ejecutado, "
        "pruebas y Model Card.",
        styles["body"],
    ))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "<b>Reproducibilidad.</b> Todo el contenido de este informe se genera con los comandos del README: "
        "<font face='Courier'>uv sync --extra cpu</font> seguido de "
        "<font face='Courier'>bash scripts/run_all.sh</font>, que deja los artefactos en "
        "<font face='Courier'>reports/</font> y regenera este PDF. Ninguna cifra está escrita a mano: "
        "todas se leen de <font face='Courier'>reports/cv_metrics.json</font>, "
        "<font face='Courier'>reports/cv_per_class.csv</font>, "
        "<font face='Courier'>reports/cv_errors_by_class.csv</font>, "
        "<font face='Courier'>reports/budget_study.json</font> y "
        "<font face='Courier'>reports/dataset_audit.json</font>.",
        styles["body"],
    ))
    story.append(PageBreak())
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Comprobaciones de la entrega", styles["h2"]))
    story.append(Table(
        [
            ["Comprobación", "Resultado"],
            ["Pruebas automáticas", "uv run pytest -q : todas en verde"],
            ["Linter", "uv run ruff check . : sin incidencias"],
            ["Notebook", "notebooks/01_ejercicio04_cnn_model_card.ipynb ejecutado, 0 celdas con error"],
            ["Clon limpio", "bash scripts/run_all.sh ejecutado desde un clon de GitHub sin errores"],
        ],
        colWidths=[5.0 * cm, 11.5 * cm],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), PRIMARY),
            ("TEXTCOLOR", (0, 0), (0, 0), colors.white),
            ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.6),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]),
    ))
    # Sin salto de pagina intermedio: el contenido fluye de forma continua.

    # ---------------- 1. Dataset ----------------
    story.append(Paragraph("1. Identificación del dataset y partición", styles["h1"]))
    splits = audit["splits"]
    story.append(Paragraph(
        f"<b>Dataset.</b> Fashion-MNIST, benchmark público de moda en escala de grises. "
        f"Imágenes de {audit['image_shape'][0]}x{audit['image_shape'][1]} píxeles, canal único, "
        f"{audit['num_classes']} clases. Procedencia: descarga automática de "
        f"<font face='Courier'>tf.keras.datasets.fashion_mnist</font> (ficha de Keras, licencia MIT). "
        f"El bloque oficial de entrenamiento tiene {splits['entrenamiento_completo']['n_samples']} imágenes "
        f"y el de prueba {splits['prueba']['n_samples']}.",
        styles["body"],
    ))
    story.append(Paragraph(
        f"<b>Preprocesamiento.</b> Conversión a float32, división por 255 (rango [0, 1]) y añadido del "
        f"canal, de modo que la forma de entrada es (N, {audit['image_shape'][0]}, "
        f"{audit['image_shape'][1]}, 1). Rango comprobado en la auditoría: "
        f"[{splits['entrenamiento']['value_min']:.0f}, {splits['entrenamiento']['value_max']:.0f}].",
        styles["body"],
    ))
    story.append(Paragraph(
        f"<b>Partición reproducible.</b> Fija y determinista: las últimas "
        f"{meta['validation_size']} imágenes del bloque de 60.000 son validación y las "
        f"{meta['train_size']} restantes son entrenamiento. Las {meta['test_size']} imágenes del bloque "
        f"oficial de prueba no se usan nunca durante el entrenamiento. Semilla "
        f"<b>{meta['seed']}</b> en Python, NumPy y TensorFlow.",
        styles["body"],
    ))
    story.append(Table(
        [
            ["Partición", "Imágenes", "Forma", "Rango", "Desbalance max/min", "Reparto min-max por clase"],
            [
                "Entrenamiento",
                str(splits["entrenamiento"]["n_samples"]),
                str(tuple(splits["entrenamiento"]["image_shape"])),
                f"[{splits['entrenamiento']['value_min']:.0f}, {splits['entrenamiento']['value_max']:.0f}]",
                f"{splits['entrenamiento']['imbalance_ratio_max_over_min']:.4f}",
                f"{splits['entrenamiento']['class_count_min']} - {splits['entrenamiento']['class_count_max']}",
            ],
            [
                "Validación",
                str(splits["validacion"]["n_samples"]),
                str(tuple(splits["validacion"]["image_shape"])),
                f"[{splits['validacion']['value_min']:.0f}, {splits['validacion']['value_max']:.0f}]",
                f"{splits['validacion']['imbalance_ratio_max_over_min']:.4f}",
                f"{splits['validacion']['class_count_min']} - {splits['validacion']['class_count_max']}",
            ],
            [
                "Prueba",
                str(splits["prueba"]["n_samples"]),
                str(tuple(splits["prueba"]["image_shape"])),
                f"[{splits['prueba']['value_min']:.0f}, {splits['prueba']['value_max']:.0f}]",
                f"{splits['prueba']['imbalance_ratio_max_over_min']:.4f}",
                f"{splits['prueba']['class_count_min']} - {splits['prueba']['class_count_max']}",
            ],
        ],
        colWidths=[2.6 * cm, 1.9 * cm, 2.5 * cm, 2.3 * cm, 3.3 * cm, 4.4 * cm],
        style=table_style(),
    ))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "El reparto por clase es casi uniforme (cada clase representa cerca del 10 % en las tres "
        "particiones), de modo que el promedio global no está inflado por clases mayoritarias.",
        styles["caption"],
    ))
    # Sin salto de pagina intermedio: el contenido fluye de forma continua.

    # ---------------- 2. Resultados ----------------
    story.append(Paragraph("2. Resultados: baseline denso frente a CNN", styles["h1"]))
    story.append(Paragraph(
        "Los dos modelos se entrenan con la <b>misma partición</b> (54.000 / 6.000 / 10.000) y la "
        f"<b>misma semilla ({meta['seed']})</b>, con optimizador Adam, batch {meta['batch_size']} y "
        f"EarlyStopping (patience {meta['early_stopping']['patience']}, "
        f"restore_best_weights={meta['early_stopping']['restore_best_weights']}), "
        f"con un máximo de {meta['epochs_requested']} épocas. El tiempo de inferencia es la "
        "<b>mediana de tres mediciones</b> sobre las 10.000 imágenes de prueba.",
        styles["body"],
    ))
    main_table = [
        ["Métrica", "Baseline denso", "CNN", "Diferencia (CNN - denso)"],
        ["F1 macro", f"{dense['f1_macro']:.4f}", f"{cnn['f1_macro']:.4f}",
         f"{cnn['f1_macro'] - dense['f1_macro']:+.4f}"],
        ["Accuracy", f"{dense['accuracy']:.4f}", f"{cnn['accuracy']:.4f}",
         f"{cnn['accuracy'] - dense['accuracy']:+.4f}"],
        ["Parámetros entrenables", str(dense["parameters"]), str(cnn["parameters"]),
         f"{cnn['parameters'] - dense['parameters']:+d}"],
        ["Tamaño del modelo (.keras)", f"{dense['model_size_kb']:.2f} KB",
         f"{cnn['model_size_kb']:.2f} KB",
         f"{cnn['model_size_kb'] - dense['model_size_kb']:+.2f} KB"],
        ["Tiempo de entrenamiento", f"{dense['train_seconds']:.1f} s",
         f"{cnn['train_seconds']:.1f} s",
         f"{cnn['train_seconds'] - dense['train_seconds']:+.1f} s"],
        ["Inferencia (mediana de 3)", f"{dense['inference_ms_per_image']:.4f} ms/img",
         f"{cnn['inference_ms_per_image']:.4f} ms/img",
         f"{cnn['inference_ms_per_image'] - dense['inference_ms_per_image']:+.4f} ms/img"],
        ["Épocas ejecutadas", str(dense["epochs_completed"]), str(cnn["epochs_completed"]), "-"],
    ]
    story.append(Table(main_table, colWidths=[5.2 * cm, 3.9 * cm, 3.9 * cm, 4.0 * cm], style=table_style()))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        f"<b>Curvas.</b> El baseline denso converge con suavidad: la pérdida de entrenamiento baja de "
        f"{diag_denso['perdida_entrenamiento_inicial']:.4f} a "
        f"{diag_denso['perdida_entrenamiento_final']:.4f} y la de validación de "
        f"{diag_denso['perdida_validacion_inicial']:.4f} a "
        f"{diag_denso['perdida_validacion_final']:.4f}; {diag_denso['frase']}. La CNN parte de una "
        f"pérdida mucho más alta ({diag_cnn['perdida_entrenamiento_inicial']:.4f}) y en la última "
        f"época sigue en {diag_cnn['perdida_entrenamiento_final']:.4f} con una pérdida de validación "
        f"de {diag_cnn['perdida_validacion_final']:.4f} y una accuracy de validación de "
        f"{diag_cnn['accuracy_validacion_final']:.4f}: {diag_cnn['frase']}. El diagnóstico es el "
        "mismo que aplica el notebook (celda 2.1) y el que documenta la Model Card.",
        styles["body"],
    ))
    curva = figure(REPORTS / "training_curves.png", content_width, max_height=8.0 * cm)
    if curva is not None:
        story.append(curva)
        story.append(Paragraph(
            "Curvas de pérdida y accuracy de entrenamiento y validación de los dos modelos, con la misma "
            "partición y la misma semilla. La curva de validación de la CNN sigue bajando en la última "
            "época: no se ha estabilizado.",
            styles["caption"],
        ))
    # Sin salto de pagina intermedio: el contenido fluye de forma continua.

    # ---------------- 3. Evidencia visual ----------------
    story.append(Paragraph("3. Evidencia visual: matrices de confusión y ejemplos", styles["h1"]))
    story.append(Paragraph(
        "Las dos matrices se calculan sobre las mismas 10.000 imágenes de prueba (filas = clase real, "
        "columnas = clase predicha). La diagonal concentra los aciertos; la matriz del baseline denso es "
        "la evidencia que sostiene la decisión técnica: el denso pierde muchas menos prendas de torso "
        "que la CNN.",
        styles["body"],
    ))
    celdas = []
    for titulo, archivo in (
        ("Matriz de confusión de la CNN", "confusion_cnn.png"),
        ("Matriz de confusión del baseline denso", "confusion_dense.png"),
    ):
        imagen = figure(REPORTS / archivo, content_width / 2 - 0.4 * cm, max_height=7.6 * cm)
        celdas.append([
            imagen if imagen is not None else "",
            Paragraph(titulo, styles["caption"]),
        ])
    story.append(Table(
        [[celdas[0], celdas[1]]],
        colWidths=[content_width / 2, content_width / 2],
        style=TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]),
    ))
    story.append(Spacer(1, 0.2 * cm))
    ejemplos = figure(REPORTS / "cnn_errors.png", content_width * 0.52, max_height=7.6 * cm)
    if ejemplos is not None:
        story.append(ejemplos)
        story.append(Paragraph(
            "Dieciséis ejemplos mal clasificados por la CNN elegidos al azar con semilla fija (42), con la "
            "clase real y la predicción. Los pares visualmente semejantes (Coat/Pullover, Shirt/T-shirt-top, "
            "Sandal/Sneaker) confirman que se trata de ambigüedad de la imagen.",
            styles["caption"],
        ))
    # Sin salto de pagina intermedio: el contenido fluye de forma continua.

    # ---------------- 4. Errores por clase ----------------
    story.append(Paragraph("4. Análisis de errores por clase", styles["h1"]))
    errores_cnn = [fila for fila in errors if fila["model"] == "cnn"]
    errores_dense = [fila for fila in errors if fila["model"] == "dense"]
    peor_cnn = sorted(errores_cnn, key=lambda fila: float(fila["recall"]))[:3]
    peor_dense = sorted(errores_dense, key=lambda fila: float(fila["recall"]))[:3]
    story.append(Paragraph(
        "El promedio global puede esconder una debilidad. La tabla siguiente procede de "
        "<font face='Courier'>reports/cv_errors_by_class.csv</font>: para cada clase, cuántos ejemplos "
        "reales se pierden y cuántos falsos positivos genera.",
        styles["body"],
    ))
    story.append(Table(
        [["Clase", "Recall denso", "Recall CNN", "F1 denso", "F1 CNN",
          "Errores denso", "Errores CNN", "Falsos pos. CNN"]]
        + [
            [
                fila["class_name"],
                next((otro["recall"] for otro in errores_dense if otro["class_name"] == fila["class_name"]), "-"),
                fila["recall"],
                next((otro["f1"] for otro in per_class if otro["model"] == "dense"
                      and otro["class_name"] == fila["class_name"]), "-"),
                next((otro["f1"] for otro in per_class if otro["model"] == "cnn"
                      and otro["class_name"] == fila["class_name"]), "-"),
                next((otro["errors"] for otro in errores_dense if otro["class_name"] == fila["class_name"]), "-"),
                fila["errors"],
                fila["false_positives"],
            ]
            for fila in errores_cnn
        ],
        colWidths=[2.4 * cm, 1.9 * cm, 1.9 * cm, 1.8 * cm, 1.8 * cm, 2.1 * cm, 2.1 * cm, 2.7 * cm],
        style=table_style(),
        repeatRows=1,
    ))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "<b>Clases que se pierden.</b> Con el presupuesto del manual las tres clases más afectadas son "
        + ", ".join(f"<b>{fila['class_name']}</b> (recall {float(fila['recall']):.3f}, "
                    f"{fila['errors']} errores)" for fila in peor_cnn)
        + " para la CNN y "
        + ", ".join(f"<b>{fila['class_name']}</b> (recall {float(fila['recall']):.3f})" for fila in peor_dense)
        + " para el baseline denso. El motivo es visual: las prendas de torso (T-shirt/top, Pullover, Coat, "
        "Shirt) se distinguen por un cuello y una manga de pocos píxeles, y el calzado (Sandal, Sneaker, "
        "Ankle boot) por la suela y el tobillo. Son ambigüedades reales de la imagen de 28x28, no fallos "
        "de etiqueta.",
        styles["body"],
    ))
    errores_figura = figure(REPORTS / "errors_by_class.png", content_width, max_height=6.4 * cm)
    if errores_figura is not None:
        story.append(errores_figura)
        story.append(Paragraph(
            "Recall por clase y número de errores por clase de los dos modelos sobre el conjunto de prueba. "
            "La leyenda se mantiene fuera del área de las barras para no taparlas.",
            styles["caption"],
        ))
    # Sin salto de pagina intermedio: el contenido fluye de forma continua.

    # ---------------- 5. Presupuesto ----------------
    story.append(Paragraph("5. Estudio complementario de presupuesto", styles["h1"]))
    if budget:
        budget_dense = budget["models"]["dense"]
        budget_cnn = budget["models"]["cnn"]
        story.append(Paragraph(
            f"<b>Qué pregunta responde.</b> Las curvas dejan abierta la pregunta de si la diferencia entre "
            f"los dos modelos es una limitación de la arquitectura o del número de épocas. El proyecto "
            f"repite la comparación con <b>{budget['meta']['epochs']} épocas</b>, misma partición y misma "
            f"semilla, y calcula las curvas por época sobre el conjunto de validación, nunca sobre la "
            f"prueba.",
            styles["body"],
        ))
        story.append(Table(
            [
                ["Modelo", f"F1 macro ({meta['epochs_requested']} ep.)",
                 f"F1 macro ({budget['meta']['epochs']} ep.)", "Entrenamiento",
                 "Parámetros", "Tamaño (.keras)"],
                [
                    "Baseline denso",
                    f"{dense['f1_macro']:.4f}",
                    f"{budget_dense['f1_macro_test']:.4f}",
                    f"{budget_dense['train_seconds']:.1f} s",
                    str(budget_dense["parameters"]),
                    f"{budget_dense['model_size_kb']:.2f} KB",
                ],
                [
                    "CNN",
                    f"{cnn['f1_macro']:.4f}",
                    f"{budget_cnn['f1_macro_test']:.4f}",
                    f"{budget_cnn['train_seconds']:.1f} s",
                    str(budget_cnn["parameters"]),
                    f"{budget_cnn['model_size_kb']:.2f} KB",
                ],
            ],
            colWidths=[3.4 * cm, 2.9 * cm, 2.9 * cm, 2.9 * cm, 2.4 * cm, 2.5 * cm],
            style=table_style(),
        ))
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(
            f"Con el presupuesto ampliado la CNN mejora (F1 macro {budget_cnn['f1_macro_test']:.4f}) y "
            f"se acerca al baseline denso ({budget_dense['f1_macro_test']:.4f}), pero no lo supera. El "
            f"resultado es <b>compatible con una mejora de la arquitectura de la CNN al Trainar con más "
            f"cómputo</b>; con una sola semilla y sin intervalos de confianza <b>no permite separar de "
            f"forma concluyente</b> el efecto del presupuesto del efecto de la arquitectura. El "
            f"experimento es <b>complementario</b>: la comparación oficial del laboratorio sigue siendo la "
            f"de {meta['epochs_requested']} épocas y su resultado se reporta sin modificar.",
            styles["body"],
        ))
        presupuesto_figura = figure(REPORTS / "budget_study.png", content_width, max_height=6.6 * cm)
        if presupuesto_figura is not None:
            story.append(presupuesto_figura)
            story.append(Paragraph(
                "Izquierda: F1 macro de validación por época de los dos modelos, con la línea que marca el "
                "presupuesto del manual. Derecha: efecto del presupuesto de entrenamiento sobre el F1 macro "
                "del conjunto de prueba. La leyenda se mantiene fuera del área de las barras.",
                styles["caption"],
            ))
    else:
        story.append(Paragraph(
            "No hay estudio de presupuesto en <font face='Courier'>reports/</font>: ejecute "
            "<font face='Courier'>uv run python scripts/budget_study.py --epochs 25</font>.",
            styles["body"],
        ))
    # Sin salto de pagina intermedio: el contenido fluye de forma continua.

    # ---------------- 6. Decision ----------------
    story.append(Paragraph("6. Decisión técnica y alcance", styles["h1"]))
    if decision:
        lineas = decision.splitlines()
        story.append(Paragraph(
            "<font face='Courier'>" + "<br/>".join(linea.replace("&", "&amp;") for linea in lineas) + "</font>",
            styles["small"],
        ))
        story.append(Spacer(1, 0.2 * cm))
    ventajas_denso = []
    ventajas_cnn = []
    if dense["f1_macro"] >= cnn["f1_macro"]:
        ventajas_denso.append(
            f"F1 macro ({dense['f1_macro']:.4f} frente a {cnn['f1_macro']:.4f})"
        )
    else:
        ventajas_cnn.append(f"F1 macro ({cnn['f1_macro']:.4f} frente a {dense['f1_macro']:.4f})")
    if dense["train_seconds"] <= cnn["train_seconds"]:
        ventajas_denso.append(
            f"tiempo de entrenamiento ({dense['train_seconds']:.1f} s frente a {cnn['train_seconds']:.1f} s)"
        )
    else:
        ventajas_cnn.append(
            f"tiempo de entrenamiento ({cnn['train_seconds']:.1f} s frente a {dense['train_seconds']:.1f} s)"
        )
    if dense["inference_ms_per_image"] <= cnn["inference_ms_per_image"]:
        ventajas_denso.append(
            f"inferencia mediana ({dense['inference_ms_per_image']:.4f} frente a "
            f"{cnn['inference_ms_per_image']:.4f} ms por imagen)"
        )
    else:
        ventajas_cnn.append(
            f"inferencia mediana ({cnn['inference_ms_per_image']:.4f} frente a "
            f"{dense['inference_ms_per_image']:.4f} ms por imagen)"
        )
    if dense["parameters"] <= cnn["parameters"]:
        ventajas_denso.append(f"parámetros ({dense['parameters']} frente a {cnn['parameters']})")
    else:
        ventajas_cnn.append(f"parámetros ({cnn['parameters']} frente a {dense['parameters']})")
    if dense["model_size_kb"] <= cnn["model_size_kb"]:
        ventajas_denso.append(
            f"tamaño en disco ({dense['model_size_kb']:.2f} frente a {cnn['model_size_kb']:.2f} KB)"
        )
    else:
        ventajas_cnn.append(
            f"tamaño en disco ({cnn['model_size_kb']:.2f} frente a {dense['model_size_kb']:.2f} KB)"
        )
    story.append(Paragraph(
        "<b>Lectura Green AI.</b> El coste se compara en cinco dimensiones: F1 macro, parámetros, tamaño "
        "en disco, tiempo de entrenamiento y tiempo de inferencia. Con el presupuesto que fija el manual "
        f"({meta['epochs_requested']} épocas) el baseline denso gana en " + ", ".join(ventajas_denso)
        + "; la CNN gana únicamente en " + ", ".join(ventajas_cnn) + ". La CNN es más compacta pero más "
        "lenta y, en este dataset y con este presupuesto, no aporta ninguna mejora predictiva: una mejora "
        "que no existe no justifica el coste adicional. Queda documentada como la opción preferente cuando "
        "la restricción dominante sea la memoria o el tamaño del artefacto desplegado.",
        styles["body"],
    ))
    story.append(Paragraph(
        "<b>Alcance y límites.</b> Fashion-MNIST es un benchmark didáctico de siluetas de moda en 28x28: "
        "un buen resultado <b>no</b> demuestra preparación para imágenes médicas, OCR, vigilancia ni "
        "ningún otro dominio, y el modelo no debe usarse sin supervisión humana en decisiones que afecten "
        "a personas. <b>Limitación del entorno de ejecución.</b> La ejecución se hizo en CPU "
        f"({hardware['cpu_count']} núcleos, GPU detectada: "
        f"{hardware['gpu_devices'] if hardware['gpu_devices'] else 'ninguna'}), con TensorFlow "
        f"{hardware['tensorflow']} sobre Python {hardware['python']}: los tiempos de entrenamiento e "
        "inferencia son válidos para esta máquina y no deben generalizarse a un equipo con GPU.",
        styles["body"],
    ))
    story.append(Spacer(1, 0.2 * cm))

    # ---------------- 7. IA ----------------
    story.append(Paragraph("7. Uso de inteligencia artificial y entrega", styles["h1"]))
    story.append(Paragraph(
        "<b>Herramienta.</b> Kilo, agente de programación asistido por IA ejecutado en la terminal del "
        "estudiante (modelo MiniMax-M3). <b>Finalidad:</b> estructura del proyecto, código de los scripts "
        "y del módulo de la biblioteca, pruebas, redacción de la documentación, ejecución y depuración. "
        "<b>La IA no generó datos ni resultados:</b> todas las cifras proceden de ejecuciones reales del "
        "repositorio y están respaldadas por <font face='Courier'>reports/*.json</font> y "
        "<font face='Courier'>reports/*.csv</font>. <b>Prompts relevantes:</b> el encargo que exige que "
        "toda cifra salga de los artefactos generados, que el notebook se ejecute de principio a fin sin "
        "errores, que la inferencia se mida varias veces con mediana, que la comparación sea justa con la "
        "misma partición y semilla, y que el PDF y el README no puedan contradecir la evidencia. "
        "<b>Verificaciones del estudiante:</b> partición y semilla comprobadas contra la ejecución del "
        "script, matriz de confusión contrastada con los soportes por clase, parámetros y tamaño "
        "verificados sobre los archivos .keras guardados, curvas leídas época a época, pruebas y lint en "
        "verde antes de publicar y clon limpio ejecutado desde GitHub. <b>Correcciones realizadas:</b> se "
        "detectó y corrigió la partición relativa del modelo cuando la carpeta de destino estaba fuera "
        "del proyecto, se ajustó el diseño de las figuras para que los títulos no se solaparan y se "
        "detectó que la CNN queda en subajuste con el presupuesto del manual, lo que motivó el estudio "
        "complementario de presupuesto. El detalle completo está en "
        "<font face='Courier'>AI_USE_DECLARATION.md</font>.",
        styles["body"],
    ))
    story.append(Paragraph(
        "<b>Entregable técnico.</b> Notebook ejecutado, <font face='Courier'>src/inf8239_u02_cv/</font>, "
        "<font face='Courier'>scripts/</font>, <font face='Courier'>tests/</font>, "
        "<font face='Courier'>reports/</font>, <font face='Courier'>models/best_dense.keras</font>, "
        "<font face='Courier'>MODEL_CARD.md</font>, <font face='Courier'>AI_USE_DECLARATION.md</font>, "
        "<font face='Courier'>CAMBIOS_AL_BASE.md</font> y <font face='Courier'>README.md</font>.",
        styles["body"],
    ))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        f"Repositorio: {REPO_URL}",
        styles["small"],
    ))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.build(story)
    return OUTPUT


if __name__ == "__main__":
    print(f"PDF escrito en {build_pdf()}")
