"""Detección de movimiento: la evolución de tu main.py original.

Pipeline:  frame -> gris -> blur -> MOG2 (fondo adaptativo) -> umbral -> dilatar -> contornos
"""
import cv2
import numpy as np

Caja = tuple[int, int, int, int]  # (x, y, ancho, alto)


class DetectorMovimiento:
    def __init__(self, blur: int, sensibilidad: int, area_minima: float):
        self.blur = blur
        self.area_minima = area_minima
        # MOG2 modela cada píxel como una mezcla de gaussianas y va actualizando
        # el fondo con el tiempo. Resuelve el problema de tu versión original,
        # donde el fondo se tomaba UNA vez (frame 20) y nunca se actualizaba.
        self.fondo = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=sensibilidad, detectShadows=True
        )

    def detectar(self, frame: np.ndarray, mascara_zona: np.ndarray | None = None):
        """Devuelve (lista de cajas con movimiento, imagen binaria para depurar)."""
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # El blur elimina el "granito" del sensor que si no se detecta como movimiento.
        gris = cv2.GaussianBlur(gris, (self.blur, self.blur), 0)

        primer_plano = self.fondo.apply(gris)

        # MOG2 marca las sombras con 127 y el movimiento real con 255.
        # Umbral en 200 => nos quedamos solo con lo que no es sombra.
        _, binaria = cv2.threshold(primer_plano, 200, 255, cv2.THRESH_BINARY)

        # Dilatar "engorda" las manchas blancas para unir pedazos de un mismo
        # objeto (si no, una persona puede salir como 10 contornos chiquitos).
        binaria = cv2.dilate(binaria, None, iterations=2)

        # Zona de interés: AND bit a bit => fuera de la zona todo queda en negro.
        if mascara_zona is not None:
            binaria = cv2.bitwise_and(binaria, mascara_zona)

        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        alto, ancho = frame.shape[:2]
        area_min_px = self.area_minima * alto * ancho
        cajas: list[Caja] = [
            cv2.boundingRect(c) for c in contornos if cv2.contourArea(c) > area_min_px
        ]
        return cajas, binaria
