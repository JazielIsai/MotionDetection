"""Zona de interés (idea 4): vigilar solo una parte de la imagen.

La zona se elige con el mouse y se guarda en un JSON para no tener que
elegirla cada vez que abres el programa.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from detector import Caja


def cargar(ruta: Path) -> Caja | None:
    if not ruta.exists():
        return None
    datos = json.loads(ruta.read_text())
    return tuple(datos["zona"])


def guardar(ruta: Path, zona: Caja | None) -> None:
    if zona is None:
        ruta.unlink(missing_ok=True)
    else:
        ruta.write_text(json.dumps({"zona": list(zona)}))


def seleccionar(frame: np.ndarray) -> Caja | None:
    """Abre una ventana para dibujar un rectángulo con el mouse.

    ENTER o ESPACIO confirma, C cancela.
    """
    titulo = "Arrastra para elegir la zona - ENTER confirma, C cancela"
    x, y, w, h = cv2.selectROI(titulo, frame, showCrosshair=True)
    cv2.destroyWindow(titulo)
    if w == 0 or h == 0:
        return None
    return (int(x), int(y), int(w), int(h))


def crear_mascara(forma: tuple, zona: Caja) -> np.ndarray:
    """Imagen negra del tamaño del frame con la zona pintada de blanco (255)."""
    mascara = np.zeros(forma[:2], dtype=np.uint8)
    x, y, w, h = zona
    mascara[y : y + h, x : x + w] = 255
    return mascara


def contiene(zona: Caja | None, punto: tuple[int, int]) -> bool:
    if zona is None:
        return True
    x, y, w, h = zona
    px, py = punto
    return x <= px <= x + w and y <= py <= y + h


def dibujar(frame: np.ndarray, zona: Caja) -> None:
    x, y, w, h = zona
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 120, 0), 2)
    cv2.putText(frame, "ZONA", (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)
