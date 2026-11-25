"""Panel de información sobre el video en vivo (solo pantalla, no se graba).

Los textos no llevan acentos porque OpenCV 4 no los sabe dibujar.
"""
from datetime import datetime

import cv2
import numpy as np

from expresiones import ETIQUETAS, Cara

BLANCO, GRIS, MAGENTA = (255, 255, 255), (150, 150, 150), (255, 0, 255)
VERDE, ROJO = (0, 255, 0), (0, 0, 255)
FUENTE = cv2.FONT_HERSHEY_SIMPLEX
MAX_PERSONAS = 2  # Con más, el panel no cabe; las demás conservan su etiqueta en la cara.


def lineas_estado(grabando: bool, movimiento: bool, confirmado: bool, objetos: list,
                  con_yolo: bool) -> list[tuple[str, tuple]]:
    lineas = [(datetime.now().strftime("%d/%m/%Y  %H:%M:%S"), BLANCO),
              ("GRABANDO" if grabando else "VIGILANDO", ROJO if grabando else VERDE),
              (f"Movimiento: {'si' if movimiento else 'no'}", VERDE if movimiento else GRIS)]
    if grabando:
        if confirmado:
            que = ", ".join(sorted({o[0] for o in objetos})) or "movimiento"
            que = que.replace("person", "persona")
            lineas.append((f"Confirmado: {que}", BLANCO))
        elif con_yolo:
            lineas.append(("Confirmando...", GRIS))
    return lineas


def dibujar(vista: np.ndarray, estado: list[tuple[str, tuple]], caras: list[Cara] | None,
            confianza: float, ahora: float) -> None:
    """caras=None significa que el análisis de expresiones está apagado."""
    alto, ancho = vista.shape[:2]
    e = ancho / 1280  # Todo escala con la resolución de la cámara.
    px = lambda v: max(int(v * e), 1)

    visibles = (caras or [])[:MAX_PERSONAS]
    filas = len(estado) * 28 + 20
    if caras is not None:
        filas += 34 + sum(70 + len(ETIQUETAS) * 22 for _ in visibles)
    margen, w, h = px(12), px(330), min(px(filas), alto - 2 * px(12))

    # Fondo semitransparente para que se lea sobre cualquier imagen.
    recorte = vista[margen:margen + h, margen:margen + w]
    recorte[:] = (recorte * 0.3).astype(np.uint8)

    x, y = margen + px(12), margen

    def texto(t, avance, color=BLANCO, tam=0.6, dx=0):
        nonlocal y
        y += px(avance)
        cv2.putText(vista, t, (x + px(dx), y), FUENTE, tam * e, color, px(1.6), cv2.LINE_AA)

    for t, color in estado:
        texto(t, 28, color)
    if caras is None:
        return
    texto(f"Caras: {len(caras)}", 34, BLANCO)
    for cara in visibles:
        texto(f"Persona {cara.id} {cara.frase(confianza)}", 34, MAGENTA)
        texto(f"{cara.orientacion} | hace {max(ahora - cara.desde, 0):.0f} s", 24, GRIS, 0.5)
        y += px(6)
        for nombre, p in zip(ETIQUETAS, cara.probabilidades):
            principal = p == cara.probabilidades.max()
            color = MAGENTA if principal else GRIS
            texto(nombre, 22, BLANCO if principal else GRIS, 0.48)
            x0, largo = x + px(95), px(150)
            cv2.rectangle(vista, (x0, y - px(11)), (x0 + largo, y), (60, 60, 60), -1)
            cv2.rectangle(vista, (x0, y - px(11)), (x0 + int(largo * p), y), color, -1)
            cv2.putText(vista, f"{p:.0%}", (x0 + largo + px(8), y), FUENTE, 0.45 * e,
                        BLANCO, px(1), cv2.LINE_AA)
