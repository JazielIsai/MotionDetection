"""Clasificación con YOLO (idea 6): saber QUÉ se movió.

El detector de movimiento es barato pero "tonto": una sombra o una cortina
también son movimiento. YOLO es una red neuronal que reconoce 80 tipos de
objetos (personas, gatos, autos...), pero es caro. Por eso lo combinamos:
el detector barato decide CUÁNDO vale la pena preguntarle a YOLO.
"""
import numpy as np

import zona as zona_mod
from detector import Caja

# (nombre, confianza, (x1, y1, x2, y2))
Objeto = tuple[str, float, tuple[int, int, int, int]]


class Clasificador:
    def __init__(self, modelo: str, clases_objetivo: set[str], confianza: float):
        # Importamos aquí porque ultralytics/PyTorch tardan unos segundos en cargar:
        # si USAR_YOLO = False, ni siquiera se importan.
        from ultralytics import YOLO

        print(f"[yolo] Cargando {modelo}...")
        self.modelo = YOLO(modelo)
        self.clases_objetivo = clases_objetivo
        self.confianza = confianza

    def detectar(self, frame: np.ndarray, zona: Caja | None = None) -> list[Objeto]:
        resultado = self.modelo(frame, conf=self.confianza, verbose=False)[0]
        objetos: list[Objeto] = []
        for caja in resultado.boxes:
            nombre = self.modelo.names[int(caja.cls)]
            if nombre not in self.clases_objetivo:
                continue
            x1, y1, x2, y2 = (int(v) for v in caja.xyxy[0])
            centro = ((x1 + x2) // 2, (y1 + y2) // 2)
            # Si hay zona, solo cuentan los objetos cuyo centro está dentro.
            if zona_mod.contiene(zona, centro):
                objetos.append((nombre, float(caja.conf), (x1, y1, x2, y2)))
        return objetos
