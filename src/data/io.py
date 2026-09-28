"""Formato en disco de las muestras del guante.

data/raw/<persona>/<seña>/<sample_id>/
    glove.csv       t_ms,<FEATURE_NAMES...>   una fila por lectura
    metadata.json   persona, seña, tiempos, frecuencia medida, columnas
"""

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config import settings

SAMPLE_ID_RE = re.compile(r"_(\d{4})$")


@dataclass
class Sample:
    sample_id: str
    participant: str
    label: str
    t_ms: np.ndarray  # (L,)
    frames: np.ndarray  # (L, NUM_FEATURES)
    metadata: dict = field(default_factory=dict)
    path: Path = None


def sample_dir(participant: str, label: str, sample_id: str) -> Path:
    return settings.RAW_DIR / participant / label / sample_id


def next_sample_id(participant: str, label: str) -> str:
    label_dir = settings.RAW_DIR / participant / label
    numbers = [
        int(m.group(1))
        for p in label_dir.glob("*") if p.is_dir() and (m := SAMPLE_ID_RE.search(p.name))
    ] if label_dir.exists() else []
    return f"{participant}_{label}_{max(numbers, default=0) + 1:04d}"


def save_sample(participant: str, label: str, rows: list[tuple[int, list[float]]], extra: dict | None = None) -> Path:
    sample_id = next_sample_id(participant, label)
    out = sample_dir(participant, label, sample_id)
    out.mkdir(parents=True)

    with open(out / "glove.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_ms", *settings.FEATURE_NAMES])
        for t_ms, values in rows:
            writer.writerow([t_ms, *values])

    t = np.array([r[0] for r in rows])
    duration_s = (t[-1] - t[0]) / 1000 if len(t) > 1 else 0.0
    metadata = {
        "sample_id": sample_id,
        "participant_id": participant,
        "label": label,
        "start_timestamp_ms": int(t[0]) if len(t) else None,
        "end_timestamp_ms": int(t[-1]) if len(t) else None,
        "num_frames": len(rows),
        "duration_s": round(duration_s, 3),
        "sampling_rate_hz": round((len(t) - 1) / duration_s, 1) if duration_s > 0 else None,
        "features": settings.FEATURE_NAMES,
        **(extra or {}),
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return out


def load_sample(path: Path) -> Sample:
    """path = carpeta de la muestra. Lanza ValueError si el CSV no se puede leer."""
    metadata = {}
    if (path / "metadata.json").exists():
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

    with open(path / "glove.csv", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = [row for row in reader if row]
    if header is None:
        raise ValueError("glove.csv vacío")

    expected = ["t_ms", *settings.FEATURE_NAMES]
    missing = [c for c in expected if c not in header]
    if missing:
        raise ValueError(f"faltan columnas: {missing}")
    idx = [header.index(c) for c in expected]
    try:
        data = np.array([[float(row[i]) for i in idx] for row in rows], dtype=float).reshape(-1, len(expected))
    except (ValueError, IndexError) as e:
        raise ValueError(f"fila no numérica o incompleta ({e})") from e

    return Sample(
        sample_id=path.name,
        participant=path.parent.parent.name,
        label=path.parent.name,
        t_ms=data[:, 0],
        frames=data[:, 1:],
        metadata=metadata,
        path=path,
    )


def find_sample_dirs(raw_dir: Path | None = None) -> list[Path]:
    raw_dir = raw_dir or settings.RAW_DIR
    return sorted(p.parent for p in raw_dir.glob("*/*/*/glove.csv"))
