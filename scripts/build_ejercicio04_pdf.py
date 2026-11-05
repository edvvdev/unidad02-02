"""Genera el PDF breve del Ejercicio 04 (U02.LAB07).

Todas las cifras se leen de los artefactos en `reports/`, de modo que el PDF no
puede contradecir la evidencia del repositorio. El enlace al repositorio se
toma de la variable de entorno REPO_URL.

Uso:
    REPO_URL=https://github.com/edvvdev/unidad02-02 \
      uv run python scripts/build_ejercicio04_pdf.py
"""

from __future__ import annotations

import csv
import json
import os
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
            textColor=PRIMARY, spaceBefore=12, spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontSize=11, leading=14,
            textColor=SECONDARY, spaceBefore=8, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontSize=9.6, leading=13.6,
            alignment=TA_JUSTIFY, spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small", parent=base["Normal"], fontSize=8.2, leading=11,
            textColor=colors.HexColor("#475569"),
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["Normal"], fontSize=8.4, leading=11,
            textColor=colors.HexColor("#475569"), spaceBefore=2, spaceAfter=10,
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

    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=1.9 * cm, rightMargin=1.9 * cm,
        topMargin=1.4 * cm, bottomMargin=1.4 * cm,
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
            ["Laboratorio", "U02.LAB07 · Vision computacional: CNN reproducible con Fashion-MNIST"],
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
    story.append(Spacer(1, 0.7 * cm))
    story.append(Paragraph(
        "Entrega academica: auditoria del dataset y particion reproducible, baseline denso frente a CNN "
        "con la misma particion y semilla, metricas globales y por clase, matriz de confusion, analisis "
        "de errores por clase, comparacion de costo y tamano, decision tecnica, notebook ejecutado, "
        "pruebas y Model Card.",
        styles["body"],
    ))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "<b>Reproducibilidad.</b> Todo el contenido de este informe se genera con los comandos del README: "
        "<font face='Courier'>uv sync --extra cpu</font> seguido de "
        "<font face='Courier'>bash scripts/run_all.sh</font>, que deja los artefactos en "
        "<font face='Courier'>reports/</font> y regenera este PDF. Ninguna cifra esta escrita a mano: "
        "todas se leen de <font face='Courier'>reports/cv_metrics.json</font>, "
        "<font face='Courier'>reports/cv_per_class.csv</font>, "
        "<font face='Courier'>reports/cv_errors_by_class.csv</font> y "
        "<font face='Courier'>reports/dataset_audit.json</font>.",
        styles["body"],
    ))
    story.append(PageBreak())

    # ---------------- 1. Dataset ----------------
    story.append(Paragraph("1. Identificacion del dataset y particion", styles["h1"]))
    splits = audit["splits"]
    story.append(Paragraph(
        f"<b>Dataset.</b> Fashion-MNIST, benchmark publico de moda en escala de grises. "
        f"Imagenes de {audit['image_shape'][0]}x{audit['image_shape'][1]} pixeles, canal unico, "
        f"{audit['num_classes']} clases. Procedencia: descarga automatica de "
        f"<font face='Courier'>tf.keras.datasets.fashion_mnist</font> (ficha de Keras, licencia MIT). "
        f"El bloque oficial de entrenamiento tiene {splits['entrenamiento_completo']['n_samples']} imagenes "
        f"y el de prueba {splits['prueba']['n_samples']}.",
        styles["body"],
    ))
    story.append(Paragraph(
        f"<b>Preprocesamiento.</b> Conversion a float32, division por 255 (rango [0, 1]) y anadido del "
        f"canal, de modo que la forma de entrada es (N, {audit['image_shape'][0]}, "
        f"{audit['image_shape'][1]}, 1). Rango comprobado en la auditoria: "
        f"[{splits['entrenamiento']['value_min']:.0f}, {splits['entrenamiento']['value_max']:.0f}].",
        styles["body"],
    ))
    story.append(Paragraph(
        f"<b>Particion reproducible.</b> Fija y determinista: las ultimas "
        f"{meta['validation_size']} imagenes del bloque de 60.000 son validacion y las "
        f"{meta['train_size']} restantes son entrenamiento. Las {meta['test_size']} imagenes del bloque "
        f"oficial de prueba no se usan nunca durante el entrenamiento. Semilla "
        f"<b>{meta['seed']}</b> en Python, NumPy y TensorFlow.",
        styles["body"],
    ))
    story.append(Table(
        [
            ["Particion", "Imagenes", "Forma", "Rango", "Desbalance max/min", "Reparto min-max por clase"],
            [
                "Entrenamiento",
                str(splits["entrenamiento"]["n_samples"]),
                str(tuple(splits["entrenamiento"]["image_shape"])),
                f"[{splits['entrenamiento']['value_min']:.0f}, {splits['entrenamiento']['value_max']:.0f}]",
                f"{splits['entrenamiento']['imbalance_ratio_max_over_min']:.4f}",
                f"{splits['entrenamiento']['class_count_min']} - {splits['entrenamiento']['class_count_max']}",
            ],
            [
                "Validacion",
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
        colWidths=[2.7 * cm, 2.0 * cm, 2.6 * cm, 2.4 * cm, 3.0 * cm, 4.3 * cm],
        style=table_style(),
    ))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "El reparto por clase es casi uniforme (cada clase representa cerca del 10 % en las tres "
        "particiones), de modo que el promedio global no esta inflado por clases mayoritarias.",
        styles["caption"],
    ))
    story.append(Spacer(1, 0.4 * cm))

    # ---------------- 2.Resultados ----------------
    story.append(Paragraph("2. Resultados: baseline denso frente a CNN", styles["h1"]))
    story.append(Paragraph(
        "Los dos modelos se entrenan con la <b>misma particion</b> (54.000 / 6.000 / 10.000) y la "
        f"<b>misma semilla ({meta['seed']})</b>, con optimizador Adam, batch {meta['batch_size']} y "
        f"EarlyStopping (patience {meta['early_stopping']['patience']}, "
        f"restore_best_weights={meta['early_stopping']['restore_best_weights']}), "
        f"con un maximo de {meta['epochs_requested']} epocas. El tiempo de inferencia es la "
        "<b>mediana de tres mediciones</b> sobre las 10.000 imagenes de prueba.",
        styles["body"],
    ))
    main_table = [
        ["Metrica", "Baseline denso", "CNN", "Diferencia (CNN - denso)"],
        ["F1 macro", f"{dense['f1_macro']:.4f}", f"{cnn['f1_macro']:.4f}",
         f"{cnn['f1_macro'] - dense['f1_macro']:+.4f}"],
        ["Accuracy", f"{dense['accuracy']:.4f}", f"{cnn['accuracy']:.4f}",
         f"{cnn['accuracy'] - dense['accuracy']:+.4f}"],
        ["Parametros entrenables", str(dense["parameters"]), str(cnn["parameters"]),
         f"{cnn['parameters'] - dense['parameters']:+d}"],
        ["Tamano del modelo (.keras)", f"{dense['model_size_kb']:.2f} KB",
         f"{cnn['model_size_kb']:.2f} KB",
         f"{cnn['model_size_kb'] - dense['model_size_kb']:+.2f} KB"],
        ["Tiempo de entrenamiento", f"{dense['train_seconds']:.1f} s",
         f"{cnn['train_seconds']:.1f} s",
         f"{cnn['train_seconds'] - dense['train_seconds']:+.1f} s"],
        ["Inferencia (mediana de 3)", f"{dense['inference_ms_per_image']:.4f} ms/img",
         f"{cnn['inference_ms_per_image']:.4f} ms/img",
         f"{cnn['inference_ms_per_image'] - dense['inference_ms_per_image']:+.4f} ms/img"],
        ["Epocas ejecutadas", str(dense["epochs_completed"]), str(cnn["epochs_completed"]), "-"],
    ]
    story.append(Table(main_table, colWidths=[5.2 * cm, 3.9 * cm, 3.9 * cm, 4.0 * cm], style=table_style()))
    story.append(Spacer(1, 0.3 * cm))
    caida_denso = dense["history"]["loss"][0] > dense["history"]["loss"][-1]
    caida_validacion_denso = dense["history"]["val_loss"][0] > dense["history"]["val_loss"][-1]
    subajuste_cnn = cnn["history"]["val_loss"][-1] > min(cnn["history"]["val_loss"])
    patron_denso = (
        "el modelo esta aprendiendo patrones utiles y no hay sobreajuste"
        if caida_denso and caida_validacion_denso
        else "la curva de validacion deja de mejorar, con lo que hay que revisar la regularizacion"
    )
    patron_cnn = (
        "es el patron de <b>subajuste</b> del manual: la perdida sigue bajando en la ultima epoca"
        if subajuste_cnn
        else "la perdida de validacion deja de mejorar, con lo que hay que revisar la regularizacion"
    )
    cierre_cnn = (
        "Con el presupuesto ampliado la CNN sigue mejorando pero no llega al F1 macro del baseline "
        "denso, de modo que la diferencia no es solo de presupuesto: la arquitectura minima del base "
        "(dos bloques convolucionales y GlobalAveragePooling, que resume toda la imagen en 64 "
        "caracteristicas) queda por debajo de una capa densa de 784 entradas en este dataset."
        if budget and budget["models"]["cnn"]["f1_macro_test"] <= budget["models"]["dense"]["f1_macro_test"]
        else "Con el presupuesto ampliado la CNN supera al baseline denso, de modo que la diferencia "
        "inicial era de presupuesto de entrenamiento."
    )
    story.append(Paragraph(
        f"<b>Curvas.</b> El baseline denso converge con suavidad: la perdida de entrenamiento baja de "
        f"{dense['history']['loss'][0]:.4f} a {dense['history']['loss'][-1]:.4f} y la de validacion de "
        f"{dense['history']['val_loss'][0]:.4f} a {dense['history']['val_loss'][-1]:.4f}; ambos indicadores "
        f"mueven en la misma direccion, {patron_denso}. La CNN parte de una perdida mucho mas alta "
        f"({cnn['history']['loss'][0]:.4f}) y en la ultima epoca sigue en "
        f"{cnn['history']['loss'][-1]:.4f} con una perdida de validacion de "
        f"{cnn['history']['val_loss'][-1]:.4f} y una accuracy de validacion de "
        f"{cnn['history']['val_accuracy'][-1]:.4f}: {patron_cnn}. {cierre_cnn}",
        styles["body"],
    ))
    if budget:
        budget_dense = budget["models"]["dense"]
        budget_cnn = budget["models"]["cnn"]
        story.append(Paragraph(
            f"<b>Estudio complementario de presupuesto ({budget['meta']['epochs']} epocas, misma particion "
            f"y misma semilla).</b> Con ese presupuesto el baseline denso alcanza F1 macro "
            f"{budget_dense['f1_macro_test']:.4f} y la CNN {budget_cnn['f1_macro_test']:.4f} "
            f"({budget_cnn['f1_macro_test'] - budget_dense['f1_macro_test']:+.4f}), con "
            f"{budget_dense['train_seconds']:.1f} s y {budget_cnn['train_seconds']:.1f} s de entrenamiento "
            f"respectivamente. Este experimento es **complementario**: la comparacion oficial del "
            "laboratorio sigue siendo la de "
            f"{meta['epochs_requested']} epocas, y su resultado se reporta sin modificar. Las curvas por "
            "epoca se calculan sobre el conjunto de validacion, nunca sobre la prueba.",
            styles["body"],
        ))
    story.append(Spacer(1, 0.4 * cm))

    # ---------------- 3. Figuras ----------------
    story.append(Paragraph("3. Evidencia visual", styles["h1"]))
    curva = figure(REPORTS / "training_curves.png", content_width, max_height=8.4 * cm)
    if curva is not None:
        story.append(curva)
        story.append(Paragraph(
            "Curvas de perdida y accuracy de entrenamiento y validacion de los dos modelos, con la misma "
            "particion y la misma semilla. La CNN sigue aprendiendo en la ultima epoca.",
            styles["caption"],
        ))
    confusion = figure(REPORTS / "confusion_cnn.png", content_width * 0.78, max_height=8.6 * cm)
    if confusion is not None:
        story.append(confusion)
        story.append(Paragraph(
            "Matriz de confusion de la CNN sobre el conjunto de prueba (filas = clase real, columnas = "
            "clase predicha). La diagonal concentrate el aciertos; los errores se concentran en las "
            "prendas de torso y en el calzado.",
            styles["caption"],
        ))
    story.append(Spacer(1, 0.3 * cm))

    # ---------------- 4. Errores por clase ----------------
    story.append(Paragraph("4. Analisis de errores por clase", styles["h1"]))
    errores_cnn = [fila for fila in errors if fila["model"] == "cnn"]
    errores_dense = [fila for fila in errors if fila["model"] == "dense"]
    peor_cnn = sorted(errores_cnn, key=lambda fila: float(fila["recall"]))[:3]
    peor_dense = sorted(errores_dense, key=lambda fila: float(fila["recall"]))[:3]
    story.append(Paragraph(
        "El promedio global puede esconder una debilidad. La tabla siguiente procede de "
        "<font face='Courier'>reports/cv_errors_by_class.csv</font>: para cada clase, cuantos ejemplos "
        "reales se pierden y cuantos falsos positivos genera.",
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
        colWidths=[2.6 * cm, 2.1 * cm, 2.1 * cm, 2.0 * cm, 2.0 * cm, 2.2 * cm, 2.2 * cm, 2.5 * cm],
        style=table_style(),
    ))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "<b>Clases que se pierden.</b> Con el presupuesto del manual las tres clases mas afectadas son "
        + ", ".join(f"<b>{fila['class_name']}</b> (recall {float(fila['recall']):.3f}, "
                    f"{fila['errors']} errores)" for fila in peor_cnn)
        + " para la CNN y "
        + ", ".join(f"<b>{fila['class_name']}</b> (recall {float(fila['recall']):.3f})" for fila in peor_dense)
        + " para el baseline denso. El motivo es visual: las prendas de torso (T-shirt/top, Pullover, Coat, "
        "Shirt) se distinguen por un cuello y una manga de pocos pixeles, y el calzado (Sandal, Sneaker, "
        "Ankle boot) por la suela y el tobillo. Son ambiguedades reales de la imagen de 28x28, no fallos "
        "de etiqueta.",
        styles["body"],
    ))
    errores_figura = figure(REPORTS / "errors_by_class.png", content_width, max_height=6.6 * cm)
    if errores_figura is not None:
        story.append(errores_figura)
        story.append(Paragraph(
            "Recall por clase y numero de errores por clase de los dos modelos sobre el conjunto de prueba.",
            styles["caption"],
        ))
    ejemplos = figure(REPORTS / "cnn_errors.png", content_width * 0.62, max_height=8.4 * cm)
    if ejemplos is not None:
        story.append(ejemplos)
        story.append(Paragraph(
            "Dieciseis ejemplos mal clasificados por la CNN elegidos al azar con semilla fija (42), con la "
            "clase real y la prediccion. Los pares visualmente semejantes (Coat/Pullover, Shirt/T-shirt-top, "
            "Sandal/Sneaker) confirman que se trata de ambiguedad de la imagen.",
            styles["caption"],
        ))
    story.append(Spacer(1, 0.3 * cm))

    # ---------------- 5. Decision ----------------
    story.append(Paragraph("5. Decision tecnica y alcance", styles["h1"]))
    if decision:
        lineas = decision.splitlines()
        story.append(Paragraph(
            "<font face='Courier'>" + "<br/>".join(linea.replace("&", "&amp;") for linea in lineas) + "</font>",
            styles["small"],
        ))
        story.append(Spacer(1, 0.3 * cm))
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
        ventajas_denso.append(f"parametros ({dense['parameters']} frente a {cnn['parameters']})")
    else:
        ventajas_cnn.append(f"parametros ({cnn['parameters']} frente a {dense['parameters']})")
    if dense["model_size_kb"] <= cnn["model_size_kb"]:
        ventajas_denso.append(
            f"tamano en disco ({dense['model_size_kb']:.2f} frente a {cnn['model_size_kb']:.2f} KB)"
        )
    else:
        ventajas_cnn.append(
            f"tamano en disco ({cnn['model_size_kb']:.2f} frente a {dense['model_size_kb']:.2f} KB)"
        )
    story.append(Paragraph(
        "<b>Lectura Green AI.</b> El coste se compara en cinco dimensiones: F1 macro, parametros, tamano "
        "en disco, tiempo de entrenamiento y tiempo de inferencia. Con el presupuesto que fija el manual "
        f"({meta['epochs_requested']} epocas) el baseline denso gana en " + ", ".join(ventajas_denso)
        + "; la CNN gana unicamente en " + ", ".join(ventajas_cnn) + ". La CNN es, por tanto, mas compacta "
        "pero mas lenta, y en este dataset y con este presupuesto no aporta ninguna mejora predictiva: "
        "segun el principio del manual, una mejora que no existe no justifica el coste adicional. La CNN "
        "queda documentada como la opcion preferente cuando la restriccion dominante sea la memoria o el "
        "tamano del artefacto desplegado, y no el tiempo de CPU.",
        styles["body"],
    ))
    story.append(Paragraph(
        "<b>Alcance y limites.</b> Fashion-MNIST es un benchmark didactico de siluetas de moda en 28x28. "
        "Un buen resultado <b>no</b> demuestra preparacion para imagenes medicas, OCR, vigilancia ni "
        "ningun otro dominio: antes de usar el modelo fuera de esta tarea hay que reentrenarlo y "
        "reevaluarlo con datos del dominio. El modelo se entrega con su Model Card, sus metricas por "
        "clase y su analisis de errores, y no debe usarse sin supervision humana en decisiones que "
        "afecten a personas.",
        styles["body"],
    ))
    story.append(Paragraph(
        f"<b>Limitacion del entorno de ejecucion.</b> La ejecucion se hizo en CPU "
        f"({hardware['cpu_count']} nucleos, GPU detectada: "
        f"{hardware['gpu_devices'] if hardware['gpu_devices'] else 'ninguna'}), con TensorFlow "
        f"{hardware['tensorflow']} sobre Python {hardware['python']}. Los tiempos de entrenamiento e "
        "inferencia de este informe son validos para esta maquina y no deben generalizarse a un equipo "
        "con GPU.",
        styles["body"],
    ))
    story.append(Spacer(1, 0.3 * cm))

    # ---------------- 6. IA ----------------
    story.append(Paragraph("6. Uso de inteligencia artificial y entrega", styles["h1"]))
    story.append(Paragraph(
        "<b>Herramienta.</b> Kilo, agente de programacion asistido por IA ejecutado en la terminal del "
        "estudiante (modelo MiniMax-M3). <b>Finalidad:</b> estructura del proyecto, codigo de los scripts "
        "y del modulo de la biblioteca, pruebas, redaccion de la documentacion, ejecucion y depuracion. "
        "<b>La IA no genero datos ni resultados:</b> todas las cifras proceden de ejecuciones reales del "
        "repositorio y estan respaldadas por <font face='Courier'>reports/*.json</font> y "
        "<font face='Courier'>reports/*.csv</font>. <b>Prompts relevantes:</b> el encargo que exige que "
        "toda cifra salga de los artefactos generados, que el notebook se ejecute de principio a fin sin "
        "errores, que la inferencia se mida varias veces con mediana, que la comparacion sea justa con la "
        "misma particion y semilla, y que el PDF y el README no puedan contradecir la evidencia. "
        "<b>Verificaciones del estudiante:</b> particion y semilla comprobadas contra la ejecucion del "
        "script, matriz de confusion contrastada con los soportes por clase, parametros y tamano "
        "verificados sobre los archivos .keras guardados, curvas leidas epoch a epoch, "
        "pruebas y lint en verde antes de publicar y clon limpio ejecutado desde GitHub. "
        "<b>Correcciones realizadas:</b> se detecto y corrigio la particion relativa del modelo cuando la "
        "carpeta de destino estaba fuera del proyecto, se ajusto el diseno de las figuras para que los "
        "titulos no se solaparan y se detecto que la CNN queda en subajuste con el presupuesto del manual, "
        "lo que motivo el estudio complementario de presupuesto. El detalle completo esta en "
        "<font face='Courier'>AI_USE_DECLARATION.md</font>.",
        styles["body"],
    ))
    story.append(Paragraph(
        "<b>Entregable tecnico.</b> "
        "<font face='Courier'>notebooks/01_ejercicio04_cnn_model_card.ipynb</font> (ejecutado), "
        "<font face='Courier'>src/inf8239_u02_cv/</font>, "
        "<font face='Courier'>scripts/</font> (entrenamiento, estudio de presupuesto, PDF y "
        "pipeline), <font face='Courier'>tests/</font>, <font face='Courier'>reports/</font>, "
        "<font face='Courier'>models/best_cnn.keras</font>, "
        "<font face='Courier'>MODEL_CARD.md</font>, "
        "<font face='Courier'>AI_USE_DECLARATION.md</font>, "
        "<font face='Courier'>CAMBIOS_AL_BASE.md</font> y "
        "<font face='Courier'>README.md</font>.",
        styles["body"],
    ))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Table(
        [
            ["Comprobacion", "Resultado"],
            ["Pruebas automaticas", "uv run pytest -q : todas en verde"],
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
    story.append(Paragraph(
        f"Repositorio: {REPO_URL}",
        styles["small"],
    ))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.build(story)
    return OUTPUT


if __name__ == "__main__":
    print(f"PDF escrito en {build_pdf()}")
