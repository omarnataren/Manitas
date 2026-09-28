"""Convierte videos de señas con movimiento en secuencias de puntos de la mano.

Pasa cada cuadro por MediaPipe y guarda un CSV por video en
data/vision_dynamic/<LETRA>/<nombre>__<video>.csv con columnas t (ms desde el
inicio del video), is_left y las 63 coordenadas; solo cuadros con mano detectada.

Pensado para MSL dynamic signs (https://doi.org/10.5281/zenodo.14689869), con
nombres S<persona>-<letra>-<vista>-<repetición>.mp4, pero sirve para cualquier
carpeta de videos con ese patrón:

    python scripts/import_video_dataset.py ~/Downloads/MSL-dynamic-signs --name msl-dyn
    python scripts/import_video_dataset.py ~/Downloads/MSL-dynamic-signs --name msl-dyn --view frontal
"""

import argparse
import csv
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config import settings  # noqa: E402
from src.vision import features as vision_features  # noqa: E402
from src.vision.tracker import HandTracker  # noqa: E402

VIDEO_RE = re.compile(r"^(S\d+)-([A-ZÑ])-([a-z]+)", re.IGNORECASE)
MAX_SIDE = 640  # reducir cuadros grandes acelera MediaPipe sin perder precisión en la mano


def parse_name(path: Path):
    # macOS guarda la Ñ descompuesta (N + tilde); se normaliza para que coincida con "Ñ".
    name = unicodedata.normalize("NFC", path.stem)
    match = VIDEO_RE.match(name)
    if not match:
        return None
    return match.group(1).upper(), match.group(2).upper(), match.group(3).lower(), name


def process_video(path: Path) -> tuple[list[list], int, float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = HandTracker()  # uno por video: el seguimiento no debe arrastrarse entre videos
    rows, frames = [], 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if max(h, w) > MAX_SIDE:
                scale = MAX_SIDE / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            t_ms = int(round(frames * 1000 / fps))
            frames += 1
            det = tracker.detect(frame, t_ms)
            if det:
                pts, is_left = det
                rows.append([t_ms, int(is_left), *np.round(pts.flatten(), 5)])
    finally:
        tracker.close()
        cap.release()
    return rows, frames, frames / fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="carpeta con los videos (se busca recursivamente)")
    parser.add_argument("--name", required=True, help="nombre corto del dataset, ej. msl-dyn")
    parser.add_argument("--view", help="solo esta vista (ej. frontal); por defecto todas")
    args = parser.parse_args()

    videos = []
    for path in sorted(args.root.rglob("*")):
        if path.suffix.lower() not in {".mp4", ".mov", ".avi"}:
            continue
        parsed = parse_name(path)
        if not parsed:
            print(f"  nombre no reconocido, se omite: {path.name}")
            continue
        participant, letter, view, stem = parsed
        if view.startswith("frontal"):
            view = "frontal"
        if args.view and view != args.view.lower():
            continue
        videos.append((path, participant, letter, view, stem))
    if not videos:
        sys.exit(f"No se encontraron videos en {args.root}")

    print(f"Procesando {len(videos)} videos...")
    stats = defaultdict(lambda: [0, 0])  # letra: [videos, con mano en >=80% de cuadros]
    durations, views = [], Counter()
    for i, (path, participant, letter, view, stem) in enumerate(videos, 1):
        rows, frames, duration = process_video(path)
        durations.append(duration)
        views[view] += 1
        stats[letter][0] += 1
        if frames and len(rows) / frames >= 0.8:
            stats[letter][1] += 1
        if len(rows) < 3:
            print(f"  sin mano suficiente, se omite: {path.name}")
            continue
        out = settings.VISION_DYNAMIC_DIR / letter / f"{args.name}__{stem}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(vision_features.CSV_HEADER)
            writer.writerows(rows)
        if i % 50 == 0:
            print(f"  {i}/{len(videos)}")

    d = np.array(durations)
    print(f"\nListo. Datos en {settings.VISION_DYNAMIC_DIR}")
    print(f"Vistas: {dict(views)}")
    print(f"Duración de los videos: mediana {np.median(d):.2f} s, p90 {np.percentile(d, 90):.2f} s, máx {d.max():.2f} s")
    print("Letra   videos   con mano en >=80% de cuadros")
    for letter, (total, good) in sorted(stats.items()):
        print(f"  {letter:<5} {total:>6}   {good:>6} ({good / total:.0%})")
    print("\nSiguiente paso: python scripts/vision_dynamic.py train")


if __name__ == "__main__":
    main()
