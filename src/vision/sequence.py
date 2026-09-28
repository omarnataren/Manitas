"""Letras con movimiento (J, K, Ñ, Q, X, Z) a partir de secuencias de la mano.

Mismo principio que el guante: la secuencia se remuestrea por tiempo a RATE_HZ
(no importa si la cámara va a 30 o 60 fps) y el modelo ve una ventana de
WINDOW_SECONDS a la velocidad real. Se usa igual al entrenar y en vivo.

Features por ventana: forma de la mano al inicio/medio/final (features de
src/vision/features.py), cuánto cambia y la trayectoria de la muñeca relativa
al inicio de la ventana, medida en "tamaños de palma".
"""

import csv
import unicodedata
from collections import deque
from pathlib import Path

import joblib
import numpy as np

from src.data.windowing import make_windows
from src.vision import features as vf

RATE_HZ = 15
WINDOW_SECONDS = 2.0  # videos del dataset: mediana 1.8 s; probado 1.5/2.0/2.5 s -> 94/96/97% con RF
WINDOW = round(RATE_HZ * WINDOW_SECONDS)
TRAIN_STEP = 2
MAX_GAP_MS = 300  # si la mano se pierde más que esto, la secuencia se reinicia
MIN_BUFFER_MS = 400  # no predecir con menos mano acumulada: las predicciones no son confiables
NON_LETTER_LABELS = {"estatica", "transicion"}
TRAJECTORY_POINTS = 5


def label_from_dir(path: Path) -> str:
    return unicodedata.normalize("NFC", path.name)


def load_sequence(csv_path: Path) -> tuple[np.ndarray, np.ndarray, bool]:
    """CSV de import_video_dataset.py -> (t_ms (T,), puntos (T, 21, 3), mano izquierda)."""
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        rows = [r for r in reader if r]
    t = np.array([float(r[0]) for r in rows])
    is_left = np.array([r[1] == "1" for r in rows])
    pts = np.array([r[2:] for r in rows], dtype=float).reshape(-1, vf.NUM_LANDMARKS, 3)
    return t, pts, bool(is_left.mean() > 0.5)


def resample(t_ms: np.ndarray, pts: np.ndarray, rate: int = RATE_HZ) -> np.ndarray:
    """Interpola los puntos a tiempos uniformes (rate por segundo)."""
    if len(t_ms) == 1:
        return pts.copy()
    n = int((t_ms[-1] - t_ms[0]) * rate / 1000) + 1
    grid = t_ms[0] + np.arange(n) * 1000 / rate
    flat = pts.reshape(len(pts), -1)
    out = np.stack([np.interp(grid, t_ms, flat[:, j]) for j in range(flat.shape[1])], axis=1)
    return out.reshape(n, vf.NUM_LANDMARKS, 3)


def frame_vectors(pts_seq: np.ndarray, is_left: bool) -> np.ndarray:
    """(N, 21, 3) -> (N, 96 + 3): forma de la mano + muñeca (x, y) + tamaño de palma."""
    rows = []
    for pts in pts_seq:
        wrist = pts[vf.WRIST, :2].copy()
        if is_left:
            wrist[0] = -wrist[0]
        palm = np.linalg.norm(pts[vf.MIDDLE_MCP, :2] - pts[vf.WRIST, :2])
        rows.append(np.concatenate([vf.landmarks_to_vector(pts, is_left), wrist, [palm]]))
    return np.array(rows)


def window_features(windows: np.ndarray) -> np.ndarray:
    """(N, WINDOW, D) -> (N, features)."""
    out = []
    for win in windows:
        shape, wrist, palm = win[:, : vf.NUM_VISION_FEATURES], win[:, -3:-1], max(win[:, -1].mean(), 1e-6)
        traj = (wrist - wrist[0]) / palm
        steps = np.diff(traj, axis=0)
        speed = np.linalg.norm(steps, axis=1)
        idx = np.linspace(0, len(traj) - 1, TRAJECTORY_POINTS).round().astype(int)
        thirds = np.array_split(shape, 3)
        out.append(np.concatenate([
            *(t.mean(axis=0) for t in thirds),
            shape.std(axis=0),
            traj[idx].flatten(),
            traj.std(axis=0),
            [speed.sum(), speed.max() if len(speed) else 0.0],
        ]))
    return np.array(out)


def sequence_windows(t_ms: np.ndarray, pts: np.ndarray, is_left: bool) -> np.ndarray:
    """Secuencia completa -> ventanas (N, WINDOW, D) listas para window_features."""
    return make_windows(frame_vectors(resample(t_ms, pts), is_left), WINDOW, TRAIN_STEP)


class DynamicLetterDetector:
    """En vivo: acumula los puntos de la mano y predice sobre los últimos WINDOW_SECONDS."""

    def __init__(self, model_path: Path, predict_every_ms: int = 150):
        payload = joblib.load(model_path)
        self.model, self.labels = payload["model"], payload["labels"]
        self.predict_every_ms = predict_every_ms
        self._buffer: deque = deque()
        self._last_prediction_ms = -10**9
        self.last_proba: dict[str, float] = {}

    def push(self, t_ms: int, pts: np.ndarray | None, is_left: bool = False):
        """Agrega un cuadro (pts=None si no hay mano). Devuelve (letra, confianza) o None."""
        if pts is None:
            if self._buffer and t_ms - self._buffer[-1][0] > MAX_GAP_MS:
                self._buffer.clear()
            return None
        self._buffer.append((t_ms, pts, is_left))
        while self._buffer and t_ms - self._buffer[0][0] > WINDOW_SECONDS * 1000 + 200:
            self._buffer.popleft()
        warming_up = t_ms - self._buffer[0][0] < MIN_BUFFER_MS  # recién apareció la mano
        if t_ms - self._last_prediction_ms < self.predict_every_ms or warming_up:
            return None
        self._last_prediction_ms = t_ms

        t = np.array([b[0] for b in self._buffer], dtype=float)
        seq = np.array([b[1] for b in self._buffer])
        left = np.mean([b[2] for b in self._buffer]) > 0.5
        vectors = frame_vectors(resample(t, seq), left)
        window = make_windows(vectors[-WINDOW:], WINDOW, TRAIN_STEP)[-1:]
        proba = self.model.predict_proba(window_features(window))[0]
        self.last_proba = {label: float(p) for label, p in zip(self.labels, proba)}
        i = int(proba.argmax())
        return self.labels[i], float(proba[i])
