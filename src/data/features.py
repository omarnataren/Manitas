"""Features estadísticas por ventana para Random Forest.

Random Forest no aprovecha el orden temporal de 60x11 valores crudos; con
resúmenes por sensor aprende más con menos datos. Los tercios (inicio, medio,
final) conservan la dirección del movimiento (p. ej. subir vs. bajar la mano).
"""

import numpy as np

STATS = ["media", "std", "min", "max", "rango", "vel_media", "vel_max", "tercio_1", "tercio_2", "tercio_3"]


def window_stats(windows: np.ndarray) -> np.ndarray:
    """(N, W, F) -> (N, F * len(STATS))."""
    vel = np.abs(np.diff(windows, axis=1))
    thirds = np.array_split(windows, 3, axis=1)
    parts = [
        windows.mean(axis=1),
        windows.std(axis=1),
        windows.min(axis=1),
        windows.max(axis=1),
        windows.max(axis=1) - windows.min(axis=1),
        vel.mean(axis=1),
        vel.max(axis=1),
        *(t.mean(axis=1) for t in thirds),
    ]
    return np.concatenate(parts, axis=1)


def feature_names(sensor_names: list[str]) -> list[str]:
    return [f"{s}_{stat}" for stat in STATS for s in sensor_names]
