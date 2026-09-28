import subprocess
import urllib.error
import urllib.request
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from config import settings

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = settings.MODELS_DIR / "vision" / "hand_landmarker.task"


def ensure_model():
    if not MODEL_PATH.exists():
        print("Descargando modelo de MediaPipe (~8 MB)...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODEL_PATH.with_suffix(".part")
        try:
            urllib.request.urlretrieve(MODEL_URL, tmp)
        except urllib.error.URLError:
            # El Python de python.org en macOS no trae certificados SSL; curl usa los del sistema.
            subprocess.run(["curl", "-fsSL", "-o", str(tmp), MODEL_URL], check=True)
        tmp.rename(MODEL_PATH)
    return MODEL_PATH


class HandTracker:
    """Detecta una mano por cuadro y devuelve sus 21 puntos (x, y, z).

    images=True es para fotos sueltas (datasets); por defecto es modo video (webcam).
    min_confidence más bajo detecta manos difíciles (p. ej. con guante), con más falsos positivos.
    hand: "derecha"/"izquierda" fija la mano (en vivo, settings.SIGNING_HAND); "auto" usa la de MediaPipe.
    """

    def __init__(self, images: bool = False, min_confidence: float = 0.5, hand: str = "auto"):
        self._images = images
        self._hand = hand
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.IMAGE if images else vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int = 0) -> Optional[tuple[np.ndarray, bool]]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._images:
            result = self._landmarker.detect(image)
        else:
            result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return None
        pts = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]])
        if self._hand == "auto":
            is_left = result.handedness[0][0].category_name == "Left"
        else:
            is_left = self._hand == "izquierda"
        return pts, is_left

    def close(self) -> None:
        self._landmarker.close()


ENHANCE_MODES = ["ninguno", "contraste", "aclarar", "ambos"]
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
_GAMMA_LUT = np.array([255 * (i / 255) ** 0.6 for i in range(256)], dtype=np.uint8)


def enhance(frame: np.ndarray, mode: str) -> np.ndarray:
    """Realce para manos oscuras (guante negro): contraste local (CLAHE) y/o aclarar sombras (gamma)."""
    if mode in ("aclarar", "ambos"):
        frame = cv2.LUT(frame, _GAMMA_LUT)
    if mode in ("contraste", "ambos"):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return frame


def draw_hand(frame: np.ndarray, pts: np.ndarray) -> None:
    h, w = frame.shape[:2]
    px = [(int(x * w), int(y * h)) for x, y, _ in pts]
    for c in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, px[c.start], px[c.end], (80, 220, 120), 2, cv2.LINE_AA)
    for p in px:
        cv2.circle(frame, p, 4, (255, 255, 255), -1, cv2.LINE_AA)


def should_quit(key: int, window_name: str) -> bool:
    """Q/q/Esc, o la ventana cerrada con el botón rojo."""
    if key in (ord("q"), ord("Q"), 27):
        return True
    return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1


def draw_hand_box(frame: np.ndarray, pts: np.ndarray, label: str = "Mano") -> None:
    h, w = frame.shape[:2]
    yellow = (0, 215, 255)
    pad = 20
    x1, y1 = int(pts[:, 0].min() * w) - pad, int(pts[:, 1].min() * h) - pad
    x2, y2 = int(pts[:, 0].max() * w) + pad, int(pts[:, 1].max() * h) + pad
    cv2.rectangle(frame, (x1, y1), (x2, y2), yellow, 2, cv2.LINE_AA)
    cv2.putText(frame, label, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, yellow, 2, cv2.LINE_AA)
