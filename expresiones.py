"""Expresiones faciales.

Ojo: esto reconoce la EXPRESIÓN visible, no lo que la persona siente.
Solo se muestra en pantalla: no dispara alertas ni se guarda.

Pipeline:  frame -> YuNet (cara + 5 puntos) -> alinear a 112x112 -> red -> suavizar
"""
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import zona as zona_mod
from detector import Caja

# Mismo orden que las salidas de la red.
ETIQUETAS = ("enojo", "asco", "miedo", "feliz", "neutral", "triste", "sorpresa")
FRASES = ("muestra enojo", "muestra asco", "muestra miedo", "se ve feliz",
          "se ve neutral", "se ve triste", "muestra sorpresa")

_BASE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
YUNET = ("face_detection_yunet/face_detection_yunet_2023mar.onnx",
         "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4")
FER = ("facial_expression_recognition/facial_expression_recognition_mobilefacenet_2022july.onnx",
       "4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9")

# Dónde deben quedar ojos, nariz y comisuras en la imagen de 112x112.
PUNTOS_ESTANDAR = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                            [41.5493, 92.3655], [70.7299, 92.2041]])
ANCHO_DETECCION = 640  # YuNet busca en una copia reducida: mucho más rápido.


@dataclass
class Cara:
    caja: Caja
    probabilidades: np.ndarray
    visto: float
    orientacion: str = "de frente"
    id: int = 0
    nombre: str = ""    # Expresión mostrada actualmente
    desde: float = 0.0  # Cuándo empezó esa expresión

    def frase(self, confianza_minima: float) -> str:
        i = int(np.argmax(self.probabilidades))
        if self.probabilidades[i] < confianza_minima:
            return "- no seguro"
        return FRASES[i]

    def etiqueta(self, confianza_minima: float) -> str:
        i = int(np.argmax(self.probabilidades))
        conf = float(self.probabilidades[i])
        return f"{ETIQUETAS[i]} {conf:.0%}" if conf >= confianza_minima else "no seguro"


def obtener_modelo(modelo: tuple[str, str], carpeta: Path) -> Path:
    """Descarga el modelo si falta y verifica que sea exactamente el esperado."""
    ruta_remota, sha = modelo
    destino = carpeta / Path(ruta_remota).name
    if destino.exists() and hashlib.sha256(destino.read_bytes()).hexdigest() == sha:
        return destino
    import requests

    print(f"[expresiones] Descargando {destino.name}...")
    r = requests.get(_BASE + ruta_remota, timeout=60)
    r.raise_for_status()
    if hashlib.sha256(r.content).hexdigest() != sha:
        raise RuntimeError(f"{destino.name}: la descarga no coincide con el hash esperado")
    carpeta.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(r.content)
    return destino


def matriz_alineacion(puntos: np.ndarray) -> np.ndarray:
    """Rotación + escala + traslación (mínimos cuadrados) hacia PUNTOS_ESTANDAR."""
    x, y = puntos[:, 0], puntos[:, 1]
    unos, ceros = np.ones(len(x)), np.zeros(len(x))
    a = np.vstack([np.column_stack([x, -y, unos, ceros]),
                   np.column_stack([y, x, ceros, unos])])
    b = np.concatenate([PUNTOS_ESTANDAR[:, 0], PUNTOS_ESTANDAR[:, 1]])
    c, s, tx, ty = np.linalg.lstsq(a, b, rcond=None)[0]
    return np.array([[c, -s, tx], [s, c, ty]])


def orientacion(puntos: np.ndarray) -> str:
    """Compara la nariz con el centro de los ojos (en la cara ya alineada)."""
    m = matriz_alineacion(puntos)
    p = puntos @ m[:, :2].T + m[:, 2]
    ojos = abs(p[1, 0] - p[0, 0])
    desvio = abs(p[2, 0] - (p[0, 0] + p[1, 0]) / 2)
    return "de lado" if desvio > 0.25 * max(ojos, 1e-6) else "de frente"


def softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()


def suavizar(anteriores: list[Cara], nuevas: list[Cara], segundos: float) -> list[Cara]:
    """Promedia cada cara con la más cercana de la vez anterior, para que no salte."""
    libres = list(anteriores)
    for cara in nuevas:
        x, y, w, h = cara.caja

        def distancia(otra: Cara) -> float:
            ox, oy, ow, oh = otra.caja
            return math.hypot(x + w / 2 - ox - ow / 2, y + h / 2 - oy - oh / 2)

        previa = min(libres, key=distancia, default=None)
        # Si se movió más de un ancho de cara, asumimos que es otra persona.
        if previa is None or distancia(previa) > w:
            continue
        libres.remove(previa)
        peso = math.exp(-max(cara.visto - previa.visto, 0.0) / segundos)
        cara.id, cara.nombre, cara.desde = previa.id, previa.nombre, previa.desde
        cara.probabilidades = peso * previa.probabilidades + (1 - peso) * cara.probabilidades
    return nuevas


class AnalizadorExpresiones:
    def __init__(self, carpeta: Path, confianza_cara: float, confianza_expresion: float,
                 ancho_minimo: int, segundos_suavizado: float):
        yunet = obtener_modelo(YUNET, carpeta)
        fer = obtener_modelo(FER, carpeta)
        self.detector = cv2.FaceDetectorYN.create(str(yunet), "", (320, 320), confianza_cara)
        self.red = cv2.dnn.readNet(str(fer))
        self.confianza = confianza_expresion
        self.ancho_minimo = ancho_minimo
        self.segundos = segundos_suavizado
        self.caras: list[Cara] = []
        self.siguiente_id = 1

    def analizar(self, frame: np.ndarray, ahora: float, zona: Caja | None = None) -> None:
        alto, ancho = frame.shape[:2]
        escala = min(1.0, ANCHO_DETECCION / ancho)
        chico = cv2.resize(frame, None, fx=escala, fy=escala)
        self.detector.setInputSize((chico.shape[1], chico.shape[0]))
        filas = self.detector.detect(chico)[1]

        nuevas: list[Cara] = []
        for fila in filas if filas is not None else []:
            x, y, w, h = (int(v) for v in fila[:4] / escala)
            if w < self.ancho_minimo or not zona_mod.contiene(zona, (x + w // 2, y + h // 2)):
                continue
            puntos = fila[4:14].reshape(5, 2).astype(float) / escala
            alineada = cv2.warpAffine(frame, matriz_alineacion(puntos), (112, 112))
            # La red espera RGB con valores de -1 a 1.
            blob = cv2.dnn.blobFromImage(alineada, 1 / 127.5, (112, 112),
                                         (127.5, 127.5, 127.5), swapRB=True)
            self.red.setInput(blob)
            nuevas.append(Cara((x, y, w, h), softmax(self.red.forward()[0]), ahora,
                               orientacion(puntos)))
        self.caras = suavizar(self.caras, nuevas, self.segundos)
        for cara in self.caras:
            if cara.id == 0:
                cara.id = self.siguiente_id
                self.siguiente_id += 1
            nombre = cara.frase(self.confianza)
            if nombre != cara.nombre:
                cara.nombre, cara.desde = nombre, ahora
        self.caras.sort(key=lambda c: c.id)

    def dibujar(self, frame: np.ndarray) -> None:
        color = (255, 0, 255)
        for cara in self.caras:
            x, y, w, h = cara.caja
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, cara.etiqueta(self.confianza), (x, max(y - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
