"""Cómo se convierte una secuencia de lecturas en ventanas de WINDOW_SIZE.

Se usa igual al entrenar (make_windows) y en vivo (SlidingWindow): siempre a la
velocidad real del guante, sin estirar ni comprimir, para que el modelo vea
los gestos a la misma velocidad en ambos casos.
"""

from collections import deque

import numpy as np

from config import settings


def make_windows(frames: np.ndarray, window_size: int = None, step: int = None) -> np.ndarray:
    """(L, F) -> (N, window_size, F).

    Si la muestra es más corta que la ventana, se rellena al inicio repitiendo la
    primera lectura: en vivo la ventana termina en "ahora", así que una seña
    corta aparece al final precedida de la mano quieta.
    """
    window_size = window_size or settings.WINDOW_SIZE
    step = step or settings.TRAIN_STEP
    if len(frames) < window_size:
        pad = np.repeat(frames[:1], window_size - len(frames), axis=0)
        return np.concatenate([pad, frames])[None]

    starts = list(range(0, len(frames) - window_size + 1, step))
    if starts[-1] != len(frames) - window_size:
        starts.append(len(frames) - window_size)  # que la última ventana incluya el final de la seña
    return np.stack([frames[s : s + window_size] for s in starts])


class SlidingWindow:
    """Buffer de las últimas WINDOW_SIZE lecturas para inferencia en vivo."""

    def __init__(self, window_size: int = None, stride: int = None):
        self.window_size = window_size or settings.WINDOW_SIZE
        self.stride = stride or settings.INFERENCE_STRIDE
        self._buffer: deque = deque(maxlen=self.window_size)
        self._since_last = 0

    def push(self, values) -> np.ndarray | None:
        """Agrega una lectura. Devuelve la ventana (W, F) cada `stride` lecturas, si está llena."""
        self._buffer.append(values)
        self._since_last += 1
        if len(self._buffer) < self.window_size or self._since_last < self.stride:
            return None
        self._since_last = 0
        return np.asarray(self._buffer, dtype=float)
