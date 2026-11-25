"""Alertas al celular por Telegram (idea 2).

Se usa la API HTTP de bots de Telegram: es un simple POST con la foto.
El envío corre en un hilo aparte para que el video no se congele
mientras se sube la imagen.
"""
import threading
import time
from pathlib import Path

import requests


class NotificadorTelegram:
    def __init__(self, token: str, chat_id: str, cooldown: float):
        self.token = token
        self.chat_id = chat_id
        self.cooldown = cooldown
        self._ultimo_envio = 0.0
        self.activo = bool(token and chat_id)
        if not self.activo:
            print("[telegram] Sin TELEGRAM_TOKEN/TELEGRAM_CHAT_ID en .env: alertas desactivadas.")

    def enviar_foto(self, ruta: Path, texto: str) -> bool:
        """Devuelve True si la alerta se mandó (o se intentó mandar)."""
        if not self.activo:
            return False
        ahora = time.time()
        if ahora - self._ultimo_envio < self.cooldown:
            print("[telegram] Alerta omitida por cooldown.")
            return False
        self._ultimo_envio = ahora
        # daemon=True: si cierras el programa, el hilo no lo mantiene vivo.
        threading.Thread(target=self._enviar, args=(ruta, texto), daemon=True).start()
        return True

    def _enviar(self, ruta: Path, texto: str) -> None:
        url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
        try:
            with open(ruta, "rb") as foto:
                r = requests.post(
                    url,
                    data={"chat_id": self.chat_id, "caption": texto},
                    files={"photo": foto},
                    timeout=15,
                )
            if r.ok:
                print("[telegram] Alerta enviada.")
            else:
                print(f"[telegram] Error {r.status_code}: {r.text[:200]}")
        except requests.RequestException as e:
            print(f"[telegram] No se pudo enviar: {e}")
