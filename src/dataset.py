import csv
import time
from pathlib import Path
from typing import Optional

import numpy as np

from shared import config
from src import features


def label_dir(label: str) -> Path:
    d = config.DATA_DIR / label
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_recording_path(label: str) -> Path:
    return label_dir(label) / f"{int(time.time() * 1000)}.csv"


def _load_csv(path: Path) -> np.ndarray:
    with open(path, newline="") as f:
        rows = [[float(v) for v in row] for row in csv.reader(f) if row]
    return np.array(rows)


def build_dataset(step: Optional[int] = None) -> tuple[np.ndarray, list[str], list[str]]:
    """Recorre data/raw/<label>/*.csv, corta cada grabación en ventanas
    (con solapamiento `step`, por defecto la mitad de WINDOW_SIZE) y devuelve
    X (vectores planos), y (etiquetas) y groups (grabación de origen de cada ventana).
    """
    step = step or max(1, config.WINDOW_SIZE // 2)
    X, y, groups = [], [], []

    if not config.DATA_DIR.exists():
        return np.empty((0,)), [], []

    for label_path in sorted(config.DATA_DIR.iterdir()):
        if not label_path.is_dir():
            continue
        label = label_path.name

        for csv_path in sorted(label_path.glob("*.csv")):
            samples = _load_csv(csv_path)
            for start in range(0, len(samples) - config.WINDOW_SIZE + 1, step):
                window = samples[start : start + config.WINDOW_SIZE]
                X.append(features.window_to_vector(window))
                y.append(label)
                groups.append(f"{label}/{csv_path.stem}")

    return np.array(X), y, groups
