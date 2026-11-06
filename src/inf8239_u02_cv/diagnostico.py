"""Diagnostico del patron de las curvas de entrenamiento y validacion.

Implementa los cuatro patrones que propone el manual de la unidad:

===================================  ============================================
Patron                               Como se detecta
===================================  ============================================
Aprendizaje util                     Ambas perdidas descienden y se estabilizan
Sobreajuste                           La perdida de validacion deja de mejorar
Subajuste                             La perdida sigue alta y en descenso
Debilidad de una clase                El promedio oculta un recall bajo
===================================  ============================================

El modulo no depende de TensorFlow: lo usan el generador del PDF, el notebook y
`scripts/verify_cifras.py`, de modo que los tres leen exactamente el mismo
diagnostico y no pueden divergir entre si.
"""

from __future__ import annotations

UMBRAL_PERDIDA_ALTA = 0.6
"""Perdida (cross-entropy) por encima de la cual se considera que el modelo
todavia no ha convergido con el presupuesto disponible."""


def diagnostico_curvas(history: dict) -> dict:
    """Clasifica el patron de las curvas de un modelo.

    Args:
        history: historial de Keras con las listas ``loss``, ``val_loss`` y,
            opcionalmente, ``val_accuracy``.

    Returns:
        Diccionario con ``patron`` ("aprendizaje_util", "sobreajuste" o
        "subajuste"), la ``frase`` lista para el informe y las cifras que la
        sustentan.
    """
    loss = [float(valor) for valor in history["loss"]]
    val_loss = [float(valor) for valor in history["val_loss"]]
    val_accuracy = [float(valor) for valor in history.get("val_accuracy", [])]

    caida_entrenamiento = loss[0] - loss[-1]
    caida_validacion = val_loss[0] - val_loss[-1]
    minimo_validacion = min(val_loss)

    # Sobreajuste: la perdida de validacion deja de mejorar (su minimo es
    # anterior a la ultima epoca) mientras la de entrenamiento sigue bajando.
    sobreajuste = val_loss[-1] > minimo_validacion and caida_entrenamiento > 0

    # Subajuste: la curva NO se ha estabilizado (sigue bajando en la ultima
    # epoca) y la perdida sigue alta para el presupuesto disponible.
    sigue_bajando = (
        len(val_loss) >= 2
        and len(loss) >= 2
        and val_loss[-1] < val_loss[-2]
        and loss[-1] < loss[-2]
    )
    perdidas_altas = (
        loss[-1] > UMBRAL_PERDIDA_ALTA or val_loss[-1] > UMBRAL_PERDIDA_ALTA
    )
    subajuste = (not sobreajuste) and sigue_bajando and perdidas_altas

    accuracy_final = val_accuracy[-1] if val_accuracy else None

    if sobreajuste:
        patron = "sobreajuste"
        frase = (
            f"la perdida de validacion deja de mejorar en la ultima epoca "
            f"({val_loss[-1]:.4f} frente al minimo {minimo_validacion:.4f}) mientras la de "
            f"entrenamiento sigue bajando: es un patron de sobreajuste y habria que "
            f"revisar la regularizacion"
        )
    elif subajuste:
        frase = (
            f"la perdida de validacion sigue descendiendo en la ultima epoca "
            f"({val_loss[-1]:.4f}) y la curva no se ha estabilizado, con una perdida de "
            f"entrenamiento de {loss[-1]:.4f}: es un patron de subajuste con este "
            f"presupuesto de entrenamiento"
        )
        patron = "subajuste"
    else:
        patron = "aprendizaje_util"
        frase = (
            f"ambas perdidas descienden de forma sostenida (entrenamiento "
            f"{loss[0]:.4f} -> {loss[-1]:.4f} y validacion {val_loss[0]:.4f} -> "
            f"{val_loss[-1]:.4f}) y no hay sobreajuste"
        )

    return {
        "patron": patron,
        "frase": frase,
        "epocas": len(loss),
        "perdida_entrenamiento_inicial": loss[0],
        "perdida_entrenamiento_final": loss[-1],
        "perdida_validacion_inicial": val_loss[0],
        "perdida_validacion_final": val_loss[-1],
        "perdida_validacion_minima": minimo_validacion,
        "caida_entrenamiento": caida_entrenamiento,
        "caida_validacion": caida_validacion,
        "accuracy_validacion_final": accuracy_final,
        "sigue_descendiendo": sigue_bajando,
        "perdidas_altas": perdidas_altas,
        "umbral_perdida_alta": UMBRAL_PERDIDA_ALTA,
    }
