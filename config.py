"""Configuración central del sistema de vigilancia.

Todos los "números mágicos" viven aquí con un nombre y una explicación,
para que puedas experimentar cambiando valores sin buscar por todo el código.
"""
import os
from pathlib import Path


def _cargar_env(ruta: Path = Path(".env")) -> None:
    """Lee un archivo .env (CLAVE=valor por línea) y lo pasa a os.environ.

    Así los secretos (token de Telegram) no quedan escritos en el código
    ni se suben a git (.env está en .gitignore).
    """
    if not ruta.exists():
        return
    for linea in ruta.read_text().splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, valor = linea.split("=", 1)
        os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))


_cargar_env()

# --- Cámara ---------------------------------------------------------------
CAMARA = 0  # 0 = webcam integrada. También acepta la ruta a un video.

# --- Detección de movimiento ----------------------------------------------
FRAMES_CALENTAMIENTO = 30  # Frames ignorados al inicio: la cámara ajusta exposición
                           # y el modelo de fondo aprende cómo se ve la escena.
BLUR = 21                  # Tamaño del desenfoque gaussiano (debe ser impar).
                           # Más grande = menos ruido, pero se pierden movimientos pequeños.
SENSIBILIDAD = 50          # varThreshold de MOG2. Más bajo = más sensible (y más falsos positivos).
AREA_MINIMA = 0.01         # Área mínima de un contorno como fracción del frame (0.01 = 1%).
                           # Usar fracción hace que funcione igual a cualquier resolución.

# --- Grabación (idea 1) ---------------------------------------------------
CARPETA_CAPTURAS = Path("capturas")
SEGUNDOS_SIN_MOVIMIENTO = 3.0  # Tras este tiempo sin movimiento, se cierra el video.
FPS_VIDEO = 20                 # FPS del archivo si la cámara no reporta los suyos.

# --- Zona de interés (idea 4) ---------------------------------------------
ARCHIVO_ZONA = Path("zona.json")  # Aquí se guarda la zona para la próxima vez.

# --- Clasificación con YOLO (idea 6) --------------------------------------
USAR_YOLO = True
MODELO_YOLO = "yolo11n.pt"  # "n" = nano: el más rápido. Se descarga solo la primera vez.
CLASES_OBJETIVO = {"person"}  # Otras: "cat", "dog", "car", "bicycle", "bird"...
CONFIANZA_MINIMA = 0.5        # 0 a 1. Qué tan seguro debe estar YOLO para aceptar.
SEGUNDOS_ENTRE_CLASIFICACIONES = 1.0  # YOLO es caro: no lo corremos en cada frame.
BORRAR_VIDEOS_SIN_CONFIRMAR = True    # Si YOLO no vio nada objetivo, se borra el video
                                      # (era una sombra, una cortina, un cambio de luz...).

# --- Alertas por Telegram (idea 2) ----------------------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
SEGUNDOS_ENTRE_ALERTAS = 30  # Cooldown para no recibir 50 mensajes seguidos.

# --- Expresiones faciales -------------------------------------------------
USAR_EXPRESIONES = True  # También: --expresiones / --no-expresiones
CARPETA_MODELOS = Path("modelos")
CONFIANZA_CARA = 0.8              # Qué tan seguro debe estar YuNet de que es una cara.
CONFIANZA_EXPRESION = 0.5         # Debajo de esto se muestra "no seguro".
ANCHO_MINIMO_CARA = 60            # En píxeles: caras más chicas no dan buen resultado.
SEGUNDOS_ENTRE_EXPRESIONES = 0.2  # 5 veces por segundo es suficiente.
SEGUNDOS_SUAVIZADO = 1.0          # Más alto = etiqueta más estable pero más lenta.
