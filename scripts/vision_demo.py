"""Demo de MediaPipe: abre la webcam, dibuja la mano y muestra la letra en un panel lateral.

No necesita el guante ni guarda nada. La letra sale de:
- el modelo entrenado con vision_teacher.py (models/vision/teacher.pkl), si existe;
- si no, reglas simples según qué dedos están estirados (A, B, D, I, L, U, V, W, Y).

    python scripts/vision_demo.py            # Q para salir
    python scripts/vision_demo.py --camera 1 # si tienes otra cámara
"""

import argparse
import sys
import time
from collections import Counter, deque
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config import settings  # noqa: E402
from src.vision import features as vision_features  # noqa: E402
from src.vision import letter_rules  # noqa: E402
from src.vision.classifier import SignClassifier  # noqa: E402
from src.vision.tracker import HandTracker, draw_hand, draw_hand_box, should_quit  # noqa: E402

WINDOW = "Manitas - demo MediaPipe"
TEACHER_PATH = settings.MODELS_DIR / "vision" / "teacher.pkl"
PANEL_WIDTH = 280
VOTE_FRAMES = 8
CONFIDENCE_THRESHOLD = 0.6
FINGER_ORDER = ["pulgar", "indice", "medio", "anular", "menique"]


def put_text(img, text: str, org, color=(255, 255, 255), scale: float = 0.7, thickness: int = 2) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_panel(height: int, letter, source: str, detail: str, fingers, orientation: str = "") -> np.ndarray:
    panel = np.full((height, PANEL_WIDTH, 3), (38, 30, 24), np.uint8)
    cv2.putText(panel, "LETRA", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)

    text = letter or "-"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 5, 8)
    color = (80, 220, 120) if letter else (110, 110, 110)
    cv2.putText(panel, text, ((PANEL_WIDTH - tw) // 2, 70 + th), cv2.FONT_HERSHEY_DUPLEX, 5, color, 8, cv2.LINE_AA)

    cv2.putText(panel, source, (24, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(panel, detail, (24, 274), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
    if orientation:
        cv2.putText(panel, f"mano: {orientation}", (24, 298), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1, cv2.LINE_AA)

    if fingers:
        cv2.putText(panel, "DEDOS ESTIRADOS", (24, 336), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
        for i, name in enumerate(FINGER_ORDER):
            y = 364 + i * 26
            on = fingers[name]
            cv2.circle(panel, (34, y - 5), 8, (80, 220, 120) if on else (90, 90, 90), -1 if on else 2, cv2.LINE_AA)
            cv2.putText(panel, name, (54, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(
            f"No se pudo abrir la cámara {args.camera}. En macOS da permiso de cámara a la "
            "terminal (Ajustes del Sistema > Privacidad y seguridad > Cámara) o prueba --camera 1."
        )

    teacher = SignClassifier.load(TEACHER_PATH) if TEACHER_PATH.exists() else None
    source = "modelo entrenado" if teacher else "reglas (aproximado)"
    tracker = HandTracker()
    recent: deque = deque(maxlen=VOTE_FRAMES)
    last_ts, prev_time, fps = -1, time.monotonic(), 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            ts = max(int(time.monotonic() * 1000), last_ts + 1)
            last_ts = ts

            det = tracker.detect(frame, ts)
            fingers, detail, orientation = None, "sin mano", ""
            if det:
                pts, is_left = det
                pts_px = pts[:, :2] * [w, h]
                fingers = letter_rules.finger_states(pts_px)
                orient_name, angle = letter_rules.hand_orientation(pts_px)
                orientation = f"{orient_name} ({angle:.0f} grados)"
                if teacher:
                    guess, conf = teacher.predict(vision_features.landmarks_to_vector(pts, is_left))
                    detail = f"cuadro: {guess} ({conf:.0%})"
                    recent.append(guess if conf >= CONFIDENCE_THRESHOLD else None)
                else:
                    guess = letter_rules.guess_letter(pts_px)
                    detail = f"cuadro: {guess or '?'}"
                    recent.append(guess)
                draw_hand(frame, pts)
                draw_hand_box(frame, pts, f"Mano {'izquierda' if is_left else 'derecha'}")
            else:
                recent.append(None)
                put_text(frame, "Muestra la mano a la camara", (12, 32), (0, 200, 255))

            top, count = Counter(recent).most_common(1)[0]
            letter = top if top and count >= VOTE_FRAMES * 0.75 else None

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
            prev_time = now
            put_text(frame, f"{fps:.0f} FPS  |  Q o Esc para salir", (12, h - 16), (200, 200, 200), 0.55)

            cv2.imshow(WINDOW, np.hstack([frame, draw_panel(h, letter, source, detail, fingers, orientation)]))
            if should_quit(cv2.waitKey(1) & 0xFF, WINDOW):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    main()
