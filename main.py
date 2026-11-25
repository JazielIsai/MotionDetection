"""Sistema de vigilancia con detección de movimiento.

Flujo de un "evento":
  1. El detector ve movimiento     -> empieza a grabar video
  2. Cada segundo se consulta YOLO -> ¿es una persona? (o lo que esté en CLASES_OBJETIVO)
  3. Si se confirma                -> guarda foto anotada y manda alerta por Telegram
  4. N segundos sin movimiento     -> cierra el video (lo borra si nunca se confirmó)

Teclas:  q = salir   z = elegir zona   x = borrar zona   m = ver/ocultar máscara
         p = ver/ocultar panel
"""
import argparse
import time

import cv2

import config
import panel
import zona as zona_mod
from clasificador import Clasificador
from detector import DetectorMovimiento
from expresiones import AnalizadorExpresiones
from grabador import Grabador
from notificador import NotificadorTelegram

VERDE = (0, 255, 0)
ROJO = (0, 0, 255)
AMARILLO = (0, 220, 255)


def parsear_argumentos():
    p = argparse.ArgumentParser(description="Vigilancia con detección de movimiento")
    p.add_argument("--fuente", default=config.CAMARA,
                   help="Índice de cámara (0, 1...) o ruta a un archivo de video")
    p.add_argument("--sin-yolo", action="store_true", help="Alertar con cualquier movimiento")
    p.add_argument("--sin-ventana", action="store_true", help="Correr sin mostrar video")
    p.add_argument("--expresiones", action=argparse.BooleanOptionalAction,
                   default=config.USAR_EXPRESIONES,
                   help="Mostrar la expresión facial en pantalla")
    args = p.parse_args()
    # "0" desde la terminal llega como texto: lo convertimos a número de cámara.
    if isinstance(args.fuente, str) and args.fuente.isdigit():
        args.fuente = int(args.fuente)
    return args


def dibujar_objetos(frame, objetos):
    for nombre, conf, (x1, y1, x2, y2) in objetos:
        cv2.rectangle(frame, (x1, y1), (x2, y2), AMARILLO, 2)
        cv2.putText(frame, f"{nombre} {conf:.0%}", (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, AMARILLO, 2)


def main():
    args = parsear_argumentos()

    camara = cv2.VideoCapture(args.fuente)
    if not camara.isOpened():
        print(f"No se pudo abrir la fuente {args.fuente!r}. "
              "En macOS revisa el permiso de cámara para tu terminal/IDE.")
        return

    fps = camara.get(cv2.CAP_PROP_FPS) or config.FPS_VIDEO
    es_archivo = isinstance(args.fuente, str)
    detector = DetectorMovimiento(config.BLUR, config.SENSIBILIDAD, config.AREA_MINIMA)
    grabador = Grabador(config.CARPETA_CAPTURAS, fps)
    notificador = NotificadorTelegram(config.TELEGRAM_TOKEN, config.TELEGRAM_CHAT_ID,
                                      config.SEGUNDOS_ENTRE_ALERTAS)
    clasificador = None
    if config.USAR_YOLO and not args.sin_yolo:
        clasificador = Clasificador(config.MODELO_YOLO, config.CLASES_OBJETIVO,
                                    config.CONFIANZA_MINIMA)

    expresiones = None
    if args.expresiones and not args.sin_ventana:  # Solo se muestran en pantalla.
        try:
            expresiones = AnalizadorExpresiones(
                config.CARPETA_MODELOS, config.CONFIANZA_CARA, config.CONFIANZA_EXPRESION,
                config.ANCHO_MINIMO_CARA, config.SEGUNDOS_SUAVIZADO)
        except (OSError, RuntimeError, cv2.error) as e:
            # Es un extra: si falla (p. ej. sin internet), la vigilancia sigue.
            print(f"[expresiones] Desactivadas: {e}")

    zona = zona_mod.cargar(config.ARCHIVO_ZONA)
    mascara = None
    mostrar_mascara = False
    mostrar_panel = True

    # Estado del evento actual
    num_frame = 0
    ultimo_movimiento = 0.0
    ultima_clasificacion = 0.0
    ultima_expresion = 0.0
    confirmado = False
    objetos = []

    print("Vigilando... (q para salir)")
    try:
        while True:
            ok, frame = camara.read()
            if not ok:  # Fin del video o cámara desconectada.
                break
            num_frame += 1
            # Con cámara usamos el reloj real. Con un archivo usamos el tiempo DEL video,
            # porque se procesa más rápido que tiempo real y los "3 s sin movimiento"
            # nunca pasarían en el reloj.
            ahora = num_frame / fps if es_archivo else time.time()

            if zona is not None and mascara is None:
                mascara = zona_mod.crear_mascara(frame.shape, zona)

            cajas, binaria = detector.detectar(frame, mascara)
            if num_frame <= config.FRAMES_CALENTAMIENTO:
                cajas = []  # El modelo de fondo todavía está aprendiendo.
            hay_movimiento = bool(cajas)
            if hay_movimiento:
                ultimo_movimiento = ahora

            # 1. Inicio de evento
            if hay_movimiento and not grabador.grabando:
                grabador.iniciar(frame)
                confirmado, objetos, ultima_clasificacion = False, [], 0.0
                print(f"[evento] Movimiento detectado: {grabador.nombre_evento}")

            # 2. Confirmar con YOLO (máximo 1 vez por segundo, y solo hasta confirmar)
            if (grabador.grabando and not confirmado
                    and ahora - ultima_clasificacion >= config.SEGUNDOS_ENTRE_CLASIFICACIONES):
                ultima_clasificacion = ahora
                if clasificador is None:
                    confirmado = True
                else:
                    objetos = clasificador.detectar(frame, zona)
                    confirmado = bool(objetos)

                # 3. Evidencia + alerta
                if confirmado:
                    evidencia = frame.copy()
                    dibujar_objetos(evidencia, objetos)
                    for x, y, w, h in cajas:
                        cv2.rectangle(evidencia, (x, y), (x + w, y + h), VERDE, 2)
                    foto = grabador.guardar_foto(evidencia)
                    que = ", ".join(sorted({o[0] for o in objetos})) or "movimiento"
                    print(f"[evento] Confirmado ({que}). Foto: {foto}")
                    notificador.enviar_foto(foto, f"🚨 {que} detectado - {grabador.nombre_evento}")

            # Se graba el frame limpio (sin rectángulos) como evidencia.
            if grabador.grabando:
                grabador.escribir(frame)

            # 4. Fin de evento
            if grabador.grabando and ahora - ultimo_movimiento > config.SEGUNDOS_SIN_MOVIMIENTO:
                conservar = confirmado or not config.BORRAR_VIDEOS_SIN_CONFIRMAR
                ruta = grabador.detener(conservar=conservar)
                print(f"[evento] Terminó. Video: {ruta}" if ruta
                      else "[evento] Terminó sin confirmar: video descartado.")

            if args.sin_ventana:
                continue

            # --- Dibujo en pantalla ---
            vista = frame.copy()
            if zona is not None:
                zona_mod.dibujar(vista, zona)
            for x, y, w, h in cajas:
                cv2.rectangle(vista, (x, y), (x + w, y + h), VERDE, 2)
            if grabador.grabando:
                dibujar_objetos(vista, objetos)
                ancho = vista.shape[1]
                cv2.circle(vista, (ancho - 80, 20), 8, ROJO, -1)
                cv2.putText(vista, "REC", (ancho - 65, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROJO, 2)
            if expresiones is not None:
                if ahora - ultima_expresion >= config.SEGUNDOS_ENTRE_EXPRESIONES:
                    ultima_expresion = ahora
                    expresiones.analizar(frame, ahora, zona)
                expresiones.dibujar(vista)
            if mostrar_panel:
                estado = panel.lineas_estado(grabador.grabando, hay_movimiento, confirmado,
                                             objetos, clasificador is not None)
                panel.dibujar(vista, estado, expresiones.caras if expresiones else None,
                              config.CONFIANZA_EXPRESION, ahora)
            cv2.imshow("Vigilancia", vista)
            if mostrar_mascara:
                cv2.imshow("Mascara de movimiento", binaria)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            elif tecla == ord("z"):
                zona = zona_mod.seleccionar(frame)
                zona_mod.guardar(config.ARCHIVO_ZONA, zona)
                mascara = None
            elif tecla == ord("x"):
                zona, mascara = None, None
                zona_mod.guardar(config.ARCHIVO_ZONA, None)
            elif tecla == ord("p"):
                mostrar_panel = not mostrar_panel
            elif tecla == ord("m"):
                mostrar_mascara = not mostrar_mascara
                if not mostrar_mascara:
                    cv2.destroyWindow("Mascara de movimiento")
    except KeyboardInterrupt:
        pass
    finally:
        # Pase lo que pase (q, Ctrl+C o un error), liberamos todo bien.
        # Un evento sin confirmar también se descarta aquí.
        grabador.detener(conservar=confirmado or not config.BORRAR_VIDEOS_SIN_CONFIRMAR)
        camara.release()  # <- el "video.relace()" de la versión original ;)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
