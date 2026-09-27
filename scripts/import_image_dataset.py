"""Convierte un dataset de imágenes (una carpeta por letra) en datos para vision_teacher.py.

Pasa cada imagen por MediaPipe, extrae los 21 puntos de la mano y los guarda en
data/vision/<LETRA>/<nombre>__<subcarpeta>.csv, el mismo formato que graba
`vision_teacher.py collect`. Después solo se corre `vision_teacher.py train`.

Pensado para MSL-ABC (https://zenodo.org/doi/10.5281/zenodo.10067508), pero sirve
para cualquier dataset con estructura .../<letra>/<imagen>.jpg:

    python scripts/import_image_dataset.py ~/Downloads/MSL-ABC --name msl-abc
    python scripts/import_image_dataset.py ~/Downloads/MSL-ABC --name msl-abc --max-per-folder 100
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from shared import config  # noqa: E402
from src import vision_features  # noqa: E402
from src.vision import HandTracker  # noqa: E402

VISION_DATA_DIR = config.ROOT_DIR / "data" / "vision"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
# MSL-ABC nombra las imágenes S<persona>-<letra>-...jpg
PARTICIPANT_RE = re.compile(r"^(S\d+)-")


def find_image_folders(root: Path) -> dict[Path, list[Path]]:
    folders: dict[Path, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith("."):
            folders[path.parent].append(path)
    return folders


def sample_evenly(images: list[Path], limit: int) -> list[Path]:
    images = sorted(images)
    if limit <= 0 or len(images) <= limit:
        return images
    idx = np.linspace(0, len(images) - 1, limit).round().astype(int)
    return [images[i] for i in idx]


def select_images(images: list[Path], max_per_folder: int, max_per_person: int) -> list[Path]:
    """Si los nombres traen a la persona, toma hasta max_per_person de cada una
    (así ninguna mano pesa más que otra); si no, hasta max_per_folder de la carpeta."""
    by_person: dict[str, list[Path]] = defaultdict(list)
    for path in images:
        match = PARTICIPANT_RE.match(path.name)
        if not match:
            return sample_evenly(images, max_per_folder)
        by_person[match.group(1)].append(path)
    if max_per_person <= 0:
        return sorted(images)
    return [p for person in sorted(by_person) for p in sample_evenly(by_person[person], max_per_person)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="carpeta raíz del dataset ya descomprimido")
    parser.add_argument("--name", required=True, help="nombre corto del dataset, ej. msl-abc")
    parser.add_argument(
        "--max-per-folder", type=int, default=200,
        help="máximo de imágenes por carpeta de letra si los nombres no traen persona (default 200, 0 = todas)",
    )
    parser.add_argument(
        "--max-per-person", type=int, default=40,
        help="máximo de imágenes por persona en cada carpeta de letra (default 40, 0 = todas)",
    )
    args = parser.parse_args()

    if not args.root.is_dir():
        sys.exit(f"No existe la carpeta {args.root}")

    folders = find_image_folders(args.root)
    if not folders:
        sys.exit(f"No se encontraron imágenes en {args.root}")

    tracker = HandTracker(images=True)
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # letra: [procesadas, con mano]
    rows: dict[tuple[str, str], list] = defaultdict(list)  # (letra, grupo) -> filas
    selected = {
        folder: select_images(images, args.max_per_folder, args.max_per_person)
        for folder, images in sorted(folders.items())
    }
    total = sum(len(v) for v in selected.values())
    processed = 0
    print(f"Procesando {total} imágenes de {len(folders)} carpetas...")

    try:
        for folder, images in selected.items():
            label = folder.name.upper()
            source = "_".join(folder.parent.relative_to(args.root).parts) or "root"
            for image_path in images:
                processed += 1
                if processed % 1000 == 0:
                    print(f"  {processed}/{total}")
                frame = cv2.imread(str(image_path))
                if frame is None:
                    continue
                stats[label][0] += 1
                det = tracker.detect(frame)
                if not det:
                    continue
                stats[label][1] += 1
                pts, is_left = det
                # Un CSV por persona (si el nombre la trae): así la evaluación deja
                # fuera personas completas y mide si funciona con manos nuevas.
                match = PARTICIPANT_RE.match(image_path.name)
                group = f"{args.name}__{match.group(1)}" if match else f"{args.name}__{source}"
                rows[(label, group)].append(["", int(is_left), *np.round(pts.flatten(), 5)])
    finally:
        tracker.close()

    for (label, group), label_rows in rows.items():
        out_path = VISION_DATA_DIR / label / f"{group}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(vision_features.CSV_HEADER)
            writer.writerows(label_rows)

    print(f"\nListo. Datos en {VISION_DATA_DIR}")
    print("Letra   imágenes   con mano detectada")
    for label, (total, found) in sorted(stats.items()):
        pct = found / total if total else 0
        flag = "  <- revisar" if pct < 0.7 else ""
        print(f"  {label:<5} {total:>8}   {found:>6} ({pct:.0%}){flag}")
    print("\nSiguiente paso: python scripts/vision_teacher.py train")


if __name__ == "__main__":
    main()
