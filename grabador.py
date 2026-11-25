"""Grabación de evidencia (idea 1): fotos y videos con fecha y hora."""
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class Grabador:
    def __init__(self, carpeta: Path, fps: float):
        self.carpeta = carpeta
        self.fps = fps
        self.carpeta.mkdir(parents=True, exist_ok=True)
        self._writer: cv2.VideoWriter | None = None
        self.ruta_video: Path | None = None
        self.nombre_evento = ""

    @property
    def grabando(self) -> bool:
        return self._writer is not None

    def iniciar(self, frame: np.ndarray) -> None:
        # Nombre tipo 2026-11-19_23-15-02: se ordena solo por fecha en el Finder.
        self.nombre_evento = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.ruta_video = self.carpeta / f"{self.nombre_evento}.mp4"
        alto, ancho = frame.shape[:2]
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.ruta_video), codec, self.fps, (ancho, alto))

    def escribir(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame)

    def guardar_foto(self, frame: np.ndarray) -> Path:
        nombre = self.nombre_evento or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ruta = self.carpeta / f"{nombre}.jpg"
        cv2.imwrite(str(ruta), frame)
        return ruta

    def detener(self, conservar: bool = True) -> Path | None:
        """Cierra el video. Si conservar=False lo borra (falsa alarma)."""
        if self._writer is None:
            return None
        self._writer.release()  # Sin esto el .mp4 queda corrupto.
        self._writer = None
        ruta = self.ruta_video
        if not conservar and ruta is not None:
            ruta.unlink(missing_ok=True)
            ruta = None
        self.nombre_evento = ""
        return ruta
