"""Demo de MediaPipe: deletrear con la webcam, sin guante.

Las letras salen de:
- estáticas: el modelo de vision_teacher.py (models/vision/teacher.pkl) o, si no existe,
  reglas simples por dedos estirados (A, B, D, I, L, U, V, W, Y);
- con movimiento (J, K, Ñ, Q, X, Z): el modelo de vision_dynamic.py (models/vision/dynamic.pkl).

Cuándo se confirma una letra (ver src/vision/letter_state.py):
- estática: mano quieta sosteniéndola ~0.7 s (la barra de progreso se llena);
- con movimiento: al terminar el gesto, cuando el modelo la ve 2 veces seguidas;
- para repetir una letra, mueve un poco la mano o sácala de cuadro.

    python scripts/vision_demo.py              # Q o Esc salir, C borrar texto, Backspace borrar letra
    python scripts/vision_demo.py --camera 1
    python scripts/vision_demo.py --no-tts     # sin voz (por defecto dice cada letra confirmada)
    python scripts/vision_demo.py --min-conf 0.3 --enhance contraste   # manos difíciles (guante negro); E cambia el realce
    python scripts/vision_demo.py --video ~/Downloads/MSL-dynamic-signs/test          # carpeta de videos
    python scripts/vision_demo.py --video mi_video.mov                                # un video tuyo
"""

import argparse
import sys
import time
import unicodedata
from collections import deque
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config import settings  # noqa: E402
from src.inference import tts  # noqa: E402
from src.vision import features as vision_features  # noqa: E402
from src.vision import letter_rules  # noqa: E402
from src.vision.classifier import SignClassifier  # noqa: E402
from src.vision.letter_state import MOTION_THRESHOLD, LetterStateMachine  # noqa: E402
from src.vision.sequence import NON_LETTER_LABELS, DynamicLetterDetector  # noqa: E402
from src.vision.tracker import ENHANCE_MODES, HandTracker, draw_hand, draw_hand_box, enhance, should_quit  # noqa: E402

WINDOW = "Manitas - demo MediaPipe"
TEACHER_PATH = settings.MODELS_DIR / "vision" / "teacher.pkl"
DYNAMIC_PATH = settings.MODELS_DIR / "vision" / "dynamic.pkl"
PANEL_WIDTH = 300
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".m4v"}
FINGER_ORDER = ["pulgar", "indice", "medio", "anular", "menique"]
GREEN, ORANGE, GRAY, LIGHT = (80, 220, 120), (60, 170, 255), (140, 140, 140), (230, 230, 230)
STATE_COLORS = {"ESPERANDO": GRAY, "SOSTENIENDO": LIGHT, "MOVIMIENTO": ORANGE, "DECIDIENDO": ORANGE, "BLOQUEADO": GREEN}


def put_text(img, text: str, org, color=(255, 255, 255), scale: float = 0.7, thickness: int = 2) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def ascii_letter(letter: str) -> str:
    # Las fuentes de OpenCV solo tienen ASCII.
    return "N~" if letter == "Ñ" else letter


def draw_big_letter(panel, letter: str | None, color) -> None:
    text = "N" if letter == "Ñ" else (letter or "-")
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 5, 8)
    x = (PANEL_WIDTH - tw) // 2
    cv2.putText(panel, text, (x, 70 + th), cv2.FONT_HERSHEY_DUPLEX, 5, color, 8, cv2.LINE_AA)
    if letter == "Ñ":
        (sw, _), _ = cv2.getTextSize("~", cv2.FONT_HERSHEY_DUPLEX, 3, 6)
        cv2.putText(panel, "~", (x + (tw - sw) // 2, 80), cv2.FONT_HERSHEY_DUPLEX, 3, color, 6, cv2.LINE_AA)


def draw_panel(height: int, machine: LetterStateMachine, shown: tuple, lines: list[tuple[str, tuple]], fingers) -> np.ndarray:
    panel = np.full((height, PANEL_WIDTH, 3), (38, 30, 24), np.uint8)
    small = lambda text, y, color=GRAY: cv2.putText(  # noqa: E731
        panel, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
    )

    if machine.state == "DECIDIENDO" and machine.candidate:
        small("DECIDIENDO... (termina el gesto)", 40, ORANGE)
        draw_big_letter(panel, machine.candidate, (120, 200, 255))
    elif machine.state == "SOSTENIENDO" and machine.candidate:
        small("SOSTEN LA LETRA...", 40)
        draw_big_letter(panel, machine.candidate, LIGHT)
        cv2.rectangle(panel, (24, 208), (PANEL_WIDTH - 24, 220), (70, 70, 70), -1)
        fill = 24 + int((PANEL_WIDTH - 48) * machine.progress)
        cv2.rectangle(panel, (24, 208), (fill, 220), GREEN, -1)
    else:
        letter, dynamic = shown
        small("ULTIMA LETRA" + (" (movimiento)" if dynamic else ""), 40)
        draw_big_letter(panel, letter, ORANGE if dynamic else GREEN if letter else GRAY)

    small(f"estado: {machine.state}", 248, STATE_COLORS.get(machine.state, GRAY))
    moving = machine.motion > MOTION_THRESHOLD
    small(f"movimiento: {machine.motion:.1f} palmas/s" + (" (moviendo)" if moving else ""), 272, ORANGE if moving else GRAY)
    y = 296
    for text, color in lines:
        small(text, y, color)
        y += 24

    if fingers:
        small("DEDOS ESTIRADOS", y + 12)
        for i, name in enumerate(FINGER_ORDER):
            fy = y + 38 + i * 22
            if fy > height - 8:
                break
            on = fingers[name]
            cv2.circle(panel, (34, fy - 5), 7, GREEN if on else (90, 90, 90), -1 if on else 2, cv2.LINE_AA)
            cv2.putText(panel, name, (52, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, LIGHT, 1, cv2.LINE_AA)
    return panel


def camera_frames(index: int, width: int | None = None, height: int | None = None):
    """(cuadro, t_ms, ms por cuadro, nombre) desde la webcam; el tiempo es el reloj real."""
    cap = cv2.VideoCapture(index)
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        sys.exit(
            f"No se pudo abrir la cámara {index}. En macOS da permiso de cámara a la "
            "terminal (Ajustes del Sistema > Privacidad y seguridad > Cámara) o prueba --camera 1."
        )
    last = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            last = max(int(time.monotonic() * 1000), last + 1)
            yield frame, last, 1, None
    finally:
        cap.release()


def video_frames(path: Path):
    """Igual que camera_frames pero desde un video o carpeta de videos; el tiempo sale de los fps
    del video (así el resultado no depende de qué tan rápido procese la computadora)."""
    paths = sorted(p for p in path.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS) if path.is_dir() else [path]
    if not paths:
        sys.exit(f"No hay videos en {path}")
    offset = 0
    for p in paths:
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        name = unicodedata.normalize("NFC", p.name)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            continue
        # Los videos del dataset cortan justo al terminar el gesto. Se congela el primer cuadro
        # 0.3 s y el último 1 s (como una persona real, que no desaparece al acabar la seña) y
        # luego 0.6 s sin mano para reiniciar la detección antes del siguiente video.
        blank = np.zeros_like(frames[0])
        sequence = [frames[0]] * int(0.3 * fps) + frames + [frames[-1]] * int(1.0 * fps) + [blank] * int(0.6 * fps)
        for i, frame in enumerate(sequence):
            yield frame, offset + int(i * 1000 / fps), 1000 / fps, name
        offset += int(len(sequence) * 1000 / fps) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=Path, help="video o carpeta de videos en lugar de la webcam")
    parser.add_argument("--no-flip", action="store_true", help="no reflejar la imagen (la webcam se refleja como espejo)")
    parser.add_argument("--min-conf", type=float, default=0.5, help="umbral de MediaPipe (default 0.5; ~0.3 para guante)")
    parser.add_argument("--enhance", choices=ENHANCE_MODES, default="ninguno", help="realce de imagen (tecla E lo cambia)")
    parser.add_argument("--resolution", help="resolución de la webcam, ej. 1280x720")
    parser.add_argument("--no-tts", action="store_true", help="sin voz")
    args = parser.parse_args()
    width, height = (int(v) for v in args.resolution.lower().split("x")) if args.resolution else (None, None)
    source = video_frames(args.video) if args.video else camera_frames(args.camera, width, height)
    enhance_mode = args.enhance
    detections: deque = deque()  # (t_ms, mano detectada) de los últimos 3 s

    teacher = SignClassifier.load(TEACHER_PATH) if TEACHER_PATH.exists() else None
    detector = DynamicLetterDetector(DYNAMIC_PATH) if DYNAMIC_PATH.exists() else None
    if not detector:
        print("Sin modelo de letras con movimiento (models/vision/dynamic.pkl); solo letras estáticas.")
    tracker = HandTracker(min_confidence=args.min_conf, hand=settings.SIGNING_HAND)
    machine = LetterStateMachine(NON_LETTER_LABELS)
    typed: list[str] = []
    shown: tuple = (None, False)
    dyn_text = ""
    prev_time, fps = time.monotonic(), 0.0

    try:
        for frame, ts, frame_ms, video_name in source:
            if not args.no_flip:
                frame = cv2.flip(frame, 1)
            if max(frame.shape[:2]) > 720:
                scale = 720 / max(frame.shape[:2])
                frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
            h, w = frame.shape[:2]
            loop_start = time.monotonic()

            if enhance_mode != "ninguno":
                frame = enhance(frame, enhance_mode)  # se muestra realzado: es lo que ve MediaPipe
            det = tracker.detect(frame, ts)
            detections.append((ts, det is not None))
            while detections and ts - detections[0][0] > 3000:
                detections.popleft()
            rate = sum(d for _, d in detections) / len(detections)
            pts, is_left = det if det else (None, False)
            fingers, static_pred, static_text = None, None, "sin mano"
            if det:
                pts_px = pts[:, :2] * [w, h]
                fingers = letter_rules.finger_states(pts_px)
                if teacher:
                    static_pred = teacher.predict(vision_features.landmarks_to_vector(pts, is_left))
                else:
                    guess = letter_rules.guess_letter(pts_px)
                    static_pred = (guess, 1.0) if guess else None
                static_text = f"estatica: {ascii_letter(static_pred[0])} ({static_pred[1]:.0%})" if static_pred else "estatica: ?"
                draw_hand(frame, pts)
                draw_hand_box(frame, pts, f"Mano {'izquierda' if is_left else 'derecha'}")
            elif not video_name:
                put_text(frame, "Muestra la mano a la camara", (12, 32), (0, 200, 255))
            if video_name:
                put_text(frame, video_name, (12, 32), (0, 215, 255), 0.6)

            dynamic_pred = detector.push(ts, pts, is_left) if detector else None
            if dynamic_pred:
                dyn_text = f"con movimiento: {ascii_letter(dynamic_pred[0])} ({dynamic_pred[1]:.0%})"

            event = machine.update(ts, pts, static_pred, dynamic_pred)
            if event:
                typed.append(event.letter)
                shown = (event.letter, event.dynamic)
                if not args.no_tts:
                    tts.say_label(event.letter)  # no bloquea el video

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
            prev_time = now
            text = "".join(ascii_letter(c) for c in typed)[-28:]
            cv2.rectangle(frame, (0, h - 64), (w, h), (20, 20, 20), -1)
            put_text(frame, f"Texto: {text}_", (12, h - 34), GREEN, 0.9)
            put_text(frame, f"{fps:.0f} FPS | E realce | C borrar | Backspace letra | Q salir", (12, h - 10), (200, 200, 200), 0.45, 1)
            rate_color = GREEN if rate >= 0.9 else ORANGE if rate >= 0.5 else (60, 60, 255)
            put_text(frame, f"mano detectada {rate:.0%} (3 s) | realce: {enhance_mode} | umbral {args.min_conf}", (12, 60), rate_color, 0.55, 1)

            lines = [(static_text, GRAY)]
            if detector:
                lines.append((dyn_text, ORANGE))
                top3 = sorted(detector.last_proba.items(), key=lambda kv: -kv[1])[:3]
                lines.append(("  " + "  ".join(f"{ascii_letter(k)[:5]} {v:.0%}" for k, v in top3), GRAY))
            cv2.imshow(WINDOW, np.hstack([frame, draw_panel(h, machine, shown, lines, fingers)]))
            # Con video se respeta su velocidad real; con la webcam no se espera.
            wait = max(1, int(frame_ms - (time.monotonic() - loop_start) * 1000)) if video_name else 1
            key = cv2.waitKey(wait) & 0xFF
            if key in (ord("e"), ord("E")):
                enhance_mode = ENHANCE_MODES[(ENHANCE_MODES.index(enhance_mode) + 1) % len(ENHANCE_MODES)]
                detections.clear()
            elif key in (ord("c"), ord("C")):
                typed.clear()
            elif key in (8, 127) and typed:
                typed.pop()
            elif should_quit(key, WINDOW):
                break
    finally:
        source.close()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    main()
