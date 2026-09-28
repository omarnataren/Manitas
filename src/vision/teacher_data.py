import csv
import re
from collections import defaultdict

import numpy as np

from config import settings
from src.vision import features as vf

PARTICIPANT_RE = re.compile(r"(S\d+)")


def load_static_frames(max_per_file: int = 15, seed: int = 0) -> dict[str, list[tuple[str, np.ndarray, bool]]]:
    """Puntos crudos de las letras estáticas (data/vision), agrupados por persona.

    Devuelve {persona: [(letra, puntos (21, 3), mano_izquierda), ...]}, tomando
    hasta max_per_file cuadros al azar de cada CSV.
    """
    rng = np.random.default_rng(seed)
    frames: dict[str, list] = defaultdict(list)
    if not settings.VISION_DATA_DIR.exists():
        return frames
    for label_dir in sorted(p for p in settings.VISION_DATA_DIR.iterdir() if p.is_dir()):
        for csv_path in sorted(label_dir.glob("*.csv")):
            with open(csv_path, newline="") as f:
                reader = csv.reader(f)
                next(reader, None)
                rows = [r for r in reader if r]
            if not rows:
                continue
            match = PARTICIPANT_RE.search(csv_path.stem)
            person = match.group(1) if match else csv_path.stem
            for i in rng.choice(len(rows), size=min(max_per_file, len(rows)), replace=False):
                row = rows[i]
                pts = np.array(row[2:], dtype=float).reshape(vf.NUM_LANDMARKS, 3)
                frames[person].append((label_dir.name, pts, row[1] == "1"))
    return frames
