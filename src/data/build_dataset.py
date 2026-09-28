"""Única fuente de verdad del preprocesamiento: muestras -> ventanas por split.

Guarda en data/processed/:
    X_<split>.npy     (N, WINDOW_SIZE, NUM_FEATURES) ventanas crudas
    y_<split>.npy     índice de etiqueta por ventana
    ids_<split>.npy   sample_id de origen de cada ventana
    labels.json       orden de las etiquetas (índice -> nombre)
    scaler.json       media/std calculadas solo con train
    splits.csv        sample_id, participant_id, label, split
    meta.json         columnas, ventana y personas por split
"""

import csv
import json
from datetime import datetime

import numpy as np

from config import settings
from src.data.io import Sample
from src.data.normalize import Scaler
from src.data.split import SPLITS
from src.data.windowing import make_windows


def build_and_save(samples: list[Sample], mapping: dict[str, str], by_participant: bool) -> dict:
    present = {s.label for s in samples}
    labels = [label for label in settings.SIGN_LABELS if label in present]
    label_idx = {label: i for i, label in enumerate(labels)}

    out = settings.PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in SPLITS:
        X, y, ids = [], [], []
        for s in samples:
            if mapping[s.sample_id] != split:
                continue
            windows = make_windows(s.frames)
            X.append(windows)
            y += [label_idx[s.label]] * len(windows)
            ids += [s.sample_id] * len(windows)
        X = np.concatenate(X) if X else np.empty((0, settings.WINDOW_SIZE, settings.NUM_FEATURES))
        np.save(out / f"X_{split}.npy", X)
        np.save(out / f"y_{split}.npy", np.array(y, dtype=int))
        np.save(out / f"ids_{split}.npy", np.array(ids))
        summary[split] = {
            "windows": len(y),
            "samples": len(set(ids)),
            "participants": sorted({s.participant for s in samples if mapping[s.sample_id] == split}),
        }

    X_train = np.load(out / "X_train.npy")
    if len(X_train):
        Scaler.fit(X_train).save(out / "scaler.json")

    with open(out / "labels.json", "w") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    with open(out / "splits.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "participant_id", "label", "split"])
        for s in sorted(samples, key=lambda s: s.sample_id):
            writer.writerow([s.sample_id, s.participant, s.label, mapping[s.sample_id]])
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "feature_names": settings.FEATURE_NAMES,
        "window_size": settings.WINDOW_SIZE,
        "train_step": settings.TRAIN_STEP,
        "split_by_participant": by_participant,
        "splits": summary,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def load_processed() -> dict:
    """Carga data/processed y verifica que coincida con config/settings.py."""
    out = settings.PROCESSED_DIR
    if not (out / "meta.json").exists():
        raise SystemExit("No hay datos procesados. Corre primero: python scripts/prepare_data.py")
    with open(out / "meta.json") as f:
        meta = json.load(f)
    if meta["feature_names"] != settings.FEATURE_NAMES or meta["window_size"] != settings.WINDOW_SIZE:
        raise SystemExit(
            "data/processed se generó con otras columnas o tamaño de ventana que config/settings.py. "
            "Vuelve a correr: python scripts/prepare_data.py"
        )
    with open(out / "labels.json") as f:
        labels = json.load(f)
    data = {"meta": meta, "labels": labels}
    for split in SPLITS:
        data[split] = (
            np.load(out / f"X_{split}.npy"),
            np.load(out / f"y_{split}.npy"),
            np.load(out / f"ids_{split}.npy"),
        )
    return data
