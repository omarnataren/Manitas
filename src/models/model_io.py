"""Formato común de un modelo guardado en models/<nombre>/.

Cada carpeta tiene meta.json con el tipo de modelo, las etiquetas y la copia de
FEATURE_NAMES / WINDOW_SIZE con que se entrenó. load_model() se niega a cargar
un modelo que no coincide con config/settings.py, para que nunca se use en vivo
un modelo que espera otros datos.
"""

import json
from datetime import datetime
from pathlib import Path

from config import settings


def write_meta(model_dir: Path, kind: str, labels: list[str], extra: dict | None = None) -> None:
    meta = {
        "kind": kind,
        "labels": labels,
        "feature_names": settings.FEATURE_NAMES,
        "window_size": settings.WINDOW_SIZE,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        **(extra or {}),
    }
    with open(model_dir / "meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def read_meta(model_dir: Path) -> dict:
    path = model_dir / "meta.json"
    if not path.exists():
        raise SystemExit(f"No hay modelo en {model_dir}. Entrénalo primero.")
    with open(path) as f:
        meta = json.load(f)
    if meta["feature_names"] != settings.FEATURE_NAMES or meta["window_size"] != settings.WINDOW_SIZE:
        raise SystemExit(
            f"El modelo en {model_dir} se entrenó con otras columnas o tamaño de ventana "
            f"({len(meta['feature_names'])} columnas, ventana {meta['window_size']}) que config/settings.py "
            f"({settings.NUM_FEATURES} columnas, ventana {settings.WINDOW_SIZE}). Reentrénalo."
        )
    return meta


def load_model(model_dir: Path):
    """Carga cualquier modelo del guante según su meta.json."""
    meta = read_meta(model_dir)
    if meta["kind"] == "random_forest":
        from src.models.random_forest_model import RandomForestSignModel

        return RandomForestSignModel.load(model_dir, meta)
    if meta["kind"] == "cnn_bigru":
        from src.models.cnn_bigru import CnnBiGruSignModel

        return CnnBiGruSignModel.load(model_dir, meta)
    raise SystemExit(f"Tipo de modelo desconocido: {meta['kind']}")


def dir_size_bytes(model_dir: Path) -> int:
    return sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file())
