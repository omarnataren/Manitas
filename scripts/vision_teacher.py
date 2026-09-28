"""Clasificador de letras con la webcam (MediaPipe Hands), independiente del guante.

Será el "maestro" que después etiquete los datos del guante. Tres modos:

    python scripts/vision_teacher.py collect --label A   # grabar cuadros de una letra
    python scripts/vision_teacher.py train               # entrenar y evaluar (Random Forest)
    python scripts/vision_teacher.py train --compare     # comparar rf/et/mlp/svm y guardar el mejor
    python scripts/vision_teacher.py live                # predecir en vivo

En collect: ESPACIO empieza/para una grabación, Q sale.
Cada grabación se guarda en data/vision/<letra>/<timestamp>.csv con la hora
de cada cuadro, para poder alinearla después con las grabaciones del guante.
"""

import argparse
import csv
import sys
import time
from collections import Counter, deque
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from config import settings  # noqa: E402
from src.vision import features as vision_features  # noqa: E402
from src.vision.classifier import SignClassifier  # noqa: E402
from src.vision.tracker import HandTracker, draw_hand, should_quit  # noqa: E402

WINDOW = "Manitas - vision"

VISION_DATA_DIR = settings.VISION_DATA_DIR
MODEL_PATH = settings.MODELS_DIR / "vision" / "teacher.pkl"
HEADER = vision_features.CSV_HEADER

CONFIDENCE_THRESHOLD = 0.6
VOTE_FRAMES = 8
BACKENDS = ["rf", "et", "mlp", "svm"]


def make_backend(name: str, with_proba: bool = True):
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    if name == "et":
        return ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 128), early_stopping=True, max_iter=300, random_state=42),
        )
    if name == "svm":
        svc = SVC(C=10, gamma="scale", random_state=42)
        # La calibración (para tener confianza por letra) hace el entrenamiento ~5x más lento;
        # solo se activa para el modelo final.
        return make_pipeline(StandardScaler(), CalibratedClassifierCV(svc, ensemble=False) if with_proba else svc)
    raise ValueError(f"Modelo desconocido: {name}")


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(
            f"No se pudo abrir la cámara {index}. En macOS da permiso de cámara a la "
            "terminal (Ajustes del Sistema > Privacidad y seguridad > Cámara) o prueba --camera 1."
        )
    return cap


def frames(cap: cv2.VideoCapture):
    last_ts = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        # Efecto espejo: así la mano izquierda/derecha que reporta MediaPipe es la real.
        frame = cv2.flip(frame, 1)
        ts = max(int(time.monotonic() * 1000), last_ts + 1)
        last_ts = ts
        yield frame, ts


def put_text(frame, text: str, y: int, color=(255, 255, 255), scale: float = 0.8) -> None:
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def collect(label: str, camera: int) -> None:
    out_dir = VISION_DATA_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker, cap = HandTracker(), open_camera(camera)
    f, writer, path, n = None, None, None, 0

    def stop_recording():
        nonlocal f, writer
        f.close()
        print(f"Guardado {path} ({n} cuadros con mano)")
        f, writer = None, None

    try:
        for frame, ts in frames(cap):
            det = tracker.detect(frame, ts)
            if det:
                draw_hand(frame, det[0])
                if writer:
                    pts, is_left = det
                    writer.writerow([f"{time.time():.3f}", int(is_left), *np.round(pts.flatten(), 5)])
                    n += 1

            if writer:
                put_text(frame, f"GRABANDO '{label}': {n} cuadros  (ESPACIO para parar)", 32, (60, 60, 255))
            else:
                put_text(frame, f"Letra '{label}'  |  ESPACIO: grabar  |  Q: salir", 32)
            if not det:
                put_text(frame, "No se detecta la mano", 64, (0, 200, 255))

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if writer:
                    stop_recording()
                else:
                    path = out_dir / f"{int(time.time() * 1000)}.csv"
                    f = open(path, "w", newline="")
                    writer = csv.writer(f)
                    writer.writerow(HEADER)
                    n = 0
            elif should_quit(key, WINDOW):
                break
    finally:
        if writer:
            stop_recording()
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


def load_dataset() -> tuple[np.ndarray, list[str], list[str]]:
    X, y, groups = [], [], []
    if not VISION_DATA_DIR.exists():
        return np.empty((0,)), y, groups
    for label_dir in sorted(p for p in VISION_DATA_DIR.iterdir() if p.is_dir()):
        for csv_path in sorted(label_dir.glob("*.csv")):
            with open(csv_path, newline="") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    pts = np.array(row[2:], dtype=float).reshape(vision_features.NUM_LANDMARKS, 3)
                    X.append(vision_features.landmarks_to_vector(pts, is_left=row[1] == "1"))
                    y.append(label_dir.name)
                    # El grupo es el nombre del archivo sin la letra: en datasets importados es
                    # la persona (todas sus letras quedan juntas en train o en test).
                    groups.append(csv_path.stem)
    return np.array(X), y, groups


def compare_backends(X: np.ndarray, y: list[str], groups: list[str], n_splits: int) -> str:
    y_enc = LabelEncoder().fit_transform(y)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}
    print(f"\n=== Comparando modelos ({n_splits}-fold por grupo) ===")
    for name in BACKENDS:
        t0 = time.monotonic()
        scores = cross_val_score(make_backend(name, with_proba=False), X, y_enc, groups=groups, cv=cv)
        results[name] = scores.mean()
        print(f"  {name:<4} {scores.mean():.2%} (+/- {scores.std():.2%})   {time.monotonic() - t0:.0f} s")
    best = max(results, key=results.get)
    print(f"Mejor: {best}")
    return best


def train(model_name: str = "rf", compare: bool = False) -> None:
    X, y, groups = load_dataset()
    if not y:
        print("No hay datos en data/vision/. Corre primero: vision_teacher.py collect --label <letra>")
        return

    recordings = Counter(label for label, _ in set(zip(y, groups)))
    print(f"Dataset: {len(y)} cuadros, {len(set(groups))} grupos (grabaciones/personas), {len(recordings)} letras")
    for label, n in sorted(Counter(y).items()):
        print(f"  - {label}: {n} cuadros en {recordings[label]} grupos")

    min_recordings = min(recordings.values())
    if len(recordings) < 2 or min_recordings < 2:
        print("\nSe necesitan al menos 2 letras y 2 grabaciones por letra para evaluar.")
        return

    if compare and min_recordings >= 3:
        model_name = compare_backends(X, y, groups, min(5, min_recordings))

    # Cuadros seguidos de una misma grabación son casi iguales: se evalúa por grabación.
    print(f"\nModelo: {model_name}")
    clf = SignClassifier(backend=make_backend(model_name, with_proba=False))
    splitter = StratifiedGroupKFold(n_splits=min(4, min_recordings), shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    clf.train(X[train_idx], [y[i] for i in train_idx])
    y_pred = clf.label_encoder.inverse_transform(clf.backend.predict(X[test_idx]))
    y_test = [y[i] for i in test_idx]
    print(f"\n=== Reporte sobre grupos de test: {sorted(set(groups[i] for i in test_idx))} ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    confusions = Counter((t, p) for t, p in zip(y_test, y_pred) if t != p)
    if confusions:
        print("Confusiones más comunes (real -> predicho):")
        for (t, p), n in confusions.most_common(8):
            print(f"  {t} -> {p}: {n}")

    if min_recordings >= 3 and not compare:
        cv = StratifiedGroupKFold(n_splits=min(5, min_recordings), shuffle=True, random_state=42)
        scores = cross_val_score(
            make_backend(model_name, with_proba=False), X, clf.label_encoder.transform(y), groups=groups, cv=cv
        )
        print(f"\nAccuracy CV por grupo: {scores.mean():.2%} (+/- {scores.std():.2%})")

    # El modelo que se guarda se entrena con todos los datos; la evaluación de arriba es solo para medir.
    clf = SignClassifier(backend=make_backend(model_name))
    clf.train(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    clf.save(MODEL_PATH)
    print(f"\nModelo guardado en {MODEL_PATH}")


def live(camera: int) -> None:
    if not MODEL_PATH.exists():
        sys.exit("No hay modelo. Corre primero: vision_teacher.py train")
    clf = SignClassifier.load(MODEL_PATH)
    tracker, cap = HandTracker(), open_camera(camera)
    recent: deque = deque(maxlen=VOTE_FRAMES)

    try:
        for frame, ts in frames(cap):
            det = tracker.detect(frame, ts)
            if det:
                pts, is_left = det
                draw_hand(frame, pts)
                label, conf = clf.predict(vision_features.landmarks_to_vector(pts, is_left))
                recent.append(label if conf >= CONFIDENCE_THRESHOLD else None)
                put_text(frame, f"cuadro: {label} ({conf:.0%})", 64, (200, 200, 200), 0.6)
            else:
                recent.append(None)
                put_text(frame, "No se detecta la mano", 64, (0, 200, 255), 0.6)

            top, count = Counter(recent).most_common(1)[0]
            if top and count >= VOTE_FRAMES * 0.75:
                put_text(frame, f"Letra: {top}", 36, (80, 220, 120), 1.1)

            cv2.imshow(WINDOW, frame)
            if should_quit(cv2.waitKey(1) & 0xFF, WINDOW):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="índice de la cámara (default 0)")
    sub = parser.add_subparsers(dest="mode", required=True)
    p_collect = sub.add_parser("collect", help="grabar cuadros de una letra")
    p_collect.add_argument("--label", required=True, help="letra o seña a grabar, ej. A")
    p_train = sub.add_parser("train", help="entrenar y evaluar el clasificador de visión")
    p_train.add_argument("--model", choices=BACKENDS, default="rf", help="modelo a entrenar (default rf)")
    p_train.add_argument("--compare", action="store_true", help="comparar todos los modelos y guardar el mejor")
    sub.add_parser("live", help="predecir en vivo con la webcam")
    args = parser.parse_args()

    if args.mode == "collect":
        collect(args.label, args.camera)
    elif args.mode == "train":
        train(args.model, args.compare)
    else:
        live(args.camera)


if __name__ == "__main__":
    main()
