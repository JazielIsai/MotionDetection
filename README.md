# Vigilancia con detección de movimiento

Convierte tu webcam en una cámara de seguridad que:

1. **Graba** video y foto cuando hay movimiento (`capturas/`).
2. **Te avisa por Telegram** con la foto.
3. **Vigila solo una zona** que tú eliges con el mouse.
4. **Confirma con IA (YOLO)** que fue una persona (u otro objeto) antes de avisarte, para evitar falsas alarmas por sombras, cortinas o cambios de luz.

## Instalación

```bash
uv venv .venv
uv pip install -p .venv/bin/python -r requirements.txt
```

La primera vez que corra con YOLO se descarga el modelo `yolo11n.pt` (~5 MB).

## Uso

```bash
.venv/bin/python main.py                 # webcam, con YOLO
.venv/bin/python main.py --sin-yolo      # alerta con cualquier movimiento
.venv/bin/python main.py --fuente video.mp4
.venv/bin/python main.py --no-expresiones # sin análisis de expresión facial
.venv/bin/python main.py --sin-ventana   # sin mostrar video (modo "servidor")
```

| Tecla | Acción |
|-------|--------|
| `q` | Salir |
| `z` | Elegir zona con el mouse (ENTER confirma) |
| `x` | Borrar zona |
| `p` | Ver u ocultar el panel de información |
| `m` | Ver la máscara de movimiento (útil para calibrar) |

En macOS, la primera vez te pedirá permiso de cámara para tu terminal o IDE.

## Configurar Telegram

1. En Telegram, abre **@BotFather**, envía `/newbot` y sigue los pasos. Te da un **token**.
2. Abre un chat con tu bot nuevo y mándale cualquier mensaje.
3. Abre en el navegador `https://api.telegram.org/bot<TU_TOKEN>/getUpdates` y busca `"chat":{"id":...}`. Ese número es tu **chat_id**.
4. Copia `.env.ejemplo` como `.env` y pon ahí ambos valores.

`.env` está en `.gitignore`: tu token nunca se sube a git.

## Cómo está organizado

```
main.py          Orquesta todo: el ciclo de cada frame y el ciclo de vida de un evento
config.py        Todos los parámetros ajustables, con explicación
detector.py      Detección de movimiento (gris → blur → MOG2 → umbral → dilatar → contornos)
zona.py          Selección, guardado y máscara de la zona de interés
grabador.py      Videos y fotos con fecha/hora
notificador.py   Envío a Telegram en un hilo aparte
clasificador.py  YOLO: ¿qué objeto es?
expresiones.py   Expresión facial en pantalla (no dispara alertas)
panel.py         Panel con estado y expresiones sobre el video en vivo
```

### Idea clave: barato primero, caro después

El detector de movimiento corre en **cada frame** porque es muy barato (restar imágenes).
YOLO es una red neuronal y es mucho más caro, así que solo se consulta **cuando ya hay
movimiento**, y como máximo una vez por segundo. Esta cascada "filtro barato → filtro caro"
es un patrón muy común en visión por computadora.

## Calibrar

Todo se ajusta en `config.py`:

- **Demasiadas detecciones falsas** → sube `SENSIBILIDAD`, `AREA_MINIMA` o `BLUR`, o usa una zona.
- **No detecta cosas** → baja esos mismos valores. Presiona `m` para ver qué está viendo el detector.
- **Detectar mascotas o autos** → agrégalos a `CLASES_OBJETIVO` (`"cat"`, `"dog"`, `"car"`).
