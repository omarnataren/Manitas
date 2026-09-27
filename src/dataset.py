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
    """Carga un CSV crudo. Tolera filas vacías o no numéricas (las salta
    avisando); si el archivo queda sin filas válidas devuelve array vacío."""
    rows = []
    bad = 0
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                rows.append([float(v) for v in row])
            except ValueError:
                bad += 1
    if bad:
        print(f"  AVISO {path.name}: {bad} filas no numéricas saltadas (¿se abrió en Excel?).")
    return np.array(rows, dtype=float) if rows else np.empty((0,))


def build_dataset(step: Optional[int] = None) -> tuple[np.ndarray, list[str], list[str]]:
    """Recorre data/raw/<label>/*.csv, corta cada grabación en ventanas
    (con solapamiento `step`, por defecto la mitad de WINDOW_SIZE) y devuelve
    X (vectores planos), y (etiquetas) y groups (grabación de origen de cada ventana).
    """
    step = step or max(1, config.WINDOW_SIZE // 2)
    X, y, groups = [], [], []

    if not config.DATA_DIR.exists():
        return np.empty((0,)), [], []

    skipped = {"cortas": [], "columnas": [], "etiqueta": []}
    for label_path in sorted(config.DATA_DIR.iterdir()):
        if not label_path.is_dir():
            continue
        label = label_path.name

        if label not in config.SIGN_LABELS:
            skipped["etiqueta"].append(label)
            continue

        for csv_path in sorted(label_path.glob("*.csv")):
            samples = _load_csv(csv_path)
            if samples.ndim != 2 or samples.shape[1] != config.NUM_FEATURES:
                cols = samples.shape[1] if samples.ndim == 2 else 0
                print(
                    f"  AVISO {label}/{csv_path.name}: {cols} columnas "
                    f"(se esperaban {config.NUM_FEATURES}), archivo ignorado."
                )
                skipped["columnas"].append(f"{label}/{csv_path.name}")
                continue
            if len(samples) < config.WINDOW_SIZE:
                print(
                    f"  AVISO {label}/{csv_path.name}: solo {len(samples)} muestras "
                    f"(mínimo {config.WINDOW_SIZE}), archivo ignorado."
                )
                skipped["cortas"].append(f"{label}/{csv_path.name}")
                continue
            for start in range(0, len(samples) - config.WINDOW_SIZE + 1, step):
                window = samples[start : start + config.WINDOW_SIZE]
                X.append(features.window_to_vector(window))
                y.append(label)
                groups.append(f"{label}/{csv_path.stem}")

    if skipped["etiqueta"]:
        print(f"  AVISO etiquetas fuera de SIGN_LABELS (ignoradas): {sorted(set(skipped['etiqueta']))}")
    total_skip = len(skipped["cortas"]) + len(skipped["columnas"])
    if total_skip:
        print(f"  Resumen: {total_skip} archivos ignorados "
              f"({len(skipped['cortas'])} cortos, {len(skipped['columnas'])} con columnas distintas).")

    return np.array(X), y, groups
