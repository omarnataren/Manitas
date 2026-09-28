"""Entrena el detector de letras con movimiento (J, K, Ñ, Q, X, Z) con la cámara.

    python scripts/import_video_dataset.py ~/Downloads/MSL-dynamic-signs --name msl-dyn   # primero
    python scripts/vision_dynamic.py train              # Random Forest
    python scripts/vision_dynamic.py train --compare    # compara rf/et/mlp/svm y guarda el mejor

Para que funcione con tu cámara y tu forma de hacer las letras, graba repeticiones propias
(se suman al dataset con más peso) y revisa qué detecta:

    python scripts/vision_dynamic.py collect --label J --person omar   # ESPACIO empieza/termina, D descarta
    python scripts/vision_dynamic.py collect --label transicion --person omar
    python scripts/vision_dynamic.py check              # qué habría confirmado el modelo en cada grabación
    python scripts/vision_dynamic.py train

La forma más rápida de iterar es el examen en vivo: te pide letras al azar, dice si el modelo
acertó y guarda cada intento como grabación propia (D descarta el último si lo hiciste mal).
Así cada ronda de prueba también es una ronda de datos:

    python scripts/vision_dynamic.py quiz --person omar            # examen -> datos
    python scripts/vision_dynamic.py train                         # ~1 min, reentrena con todo
    python scripts/vision_dynamic.py quiz --person omar            # ¿subió el % de aciertos?

Además de las 6 letras aprende dos clases negativas, para no confundir cualquier
movimiento con una letra:
- estatica:   la mano quieta con una forma cualquiera, incluida la forma inicial o final
              de una letra con movimiento (la forma de la X sin moverse no es X).
- transicion: la mano pasando de una forma a otra.
Se generan con las fotos de MSL-ABC (data/vision) y los videos, además de las
grabaciones reales de 'estatica' y 'transicion' que hagas con collect.
Se evalúa dejando fuera personas completas.
"""

import argparse
import csv
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score  # noqa: E402

from config import settings  # noqa: E402
from src.data.windowing import make_windows  # noqa: E402
from src.vision import sequence as seq  # noqa: E402
from src.vision import features as vision_features  # noqa: E402
from src.vision.classifier import BACKENDS, make_backend  # noqa: E402
from src.vision.letter_state import LetterStateMachine  # noqa: E402
from src.vision.teacher_data import load_static_frames  # noqa: E402
from src.vision.tracker import ENHANCE_MODES, HandTracker, draw_hand, enhance, should_quit  # noqa: E402

MODEL_PATH = settings.MODELS_DIR / "vision" / "dynamic.pkl"
PARTICIPANT_RE = re.compile(r"(S\d+)")
DATASET_PREFIX = "msl-dyn__"
OWN_WEIGHT = 3
COLLECT_LABELS = ["J", "K", "Ñ", "Q", "X", "Z", "transicion", "estatica"]
COLLECT_TIPS = {
    "transicion": "mueve la mano de una letra estatica a otra (varias distintas)",
    "estatica": "sosten cualquier letra estatica quieta ~2 s (cambia de letra cada vez)",
}
WINDOW_NAME = "Manitas - grabar letras con movimiento"
DYNAMIC_LETTERS = ["J", "K", "Ñ", "Q", "X", "Z"]
QUIZ_READY_MS = 1500  # "prepárate": tiempo para poner la mano en la forma inicial
QUIZ_TRY_MS = 4000  # tiempo para hacer la letra
QUIZ_TAIL_MS = 500  # se sigue grabando tras la confirmación (la mano quieta al final)
rng = np.random.default_rng(42)


def participant_of(stem: str) -> str:
    """msl-dyn__S3-J-frontal-1 -> S3; grabaciones propias omar__1790... -> omar."""
    match = PARTICIPANT_RE.search(stem)
    if match:
        return match.group(1)
    return stem.split("__")[0]


def is_own_recording(stem: str) -> bool:
    return not stem.startswith(DATASET_PREFIX)


def load_dynamic() -> tuple[list[np.ndarray], list[str], list[str], dict]:
    """Secuencias del dataset y grabaciones propias. Las propias se repiten OWN_WEIGHT veces:
    son pocas pero son justo la cámara y la forma de hacer las letras que importan en vivo.

    También devuelve las formas de mano inicial y final de cada letra con movimiento
    ({persona: [(nombre, puntos, mano_izq)]}), para generar negativos con esas formas quietas.
    """
    X, y, groups = [], [], []
    poses: dict[str, list] = {}
    if not settings.VISION_DYNAMIC_DIR.exists():
        sys.exit("No hay data/vision_dynamic. Corre primero scripts/import_video_dataset.py")
    own = Counter()
    for label_dir in sorted(p for p in settings.VISION_DYNAMIC_DIR.iterdir() if p.is_dir()):
        label = seq.label_from_dir(label_dir)
        for csv_path in sorted(label_dir.glob("*.csv")):
            t, pts, is_left = seq.load_sequence(csv_path)
            if len(t) < 3:
                continue
            if label not in seq.NON_LETTER_LABELS:
                person_poses = poses.setdefault(participant_of(csv_path.stem), [])
                person_poses += [(f"{label}_inicio", pts[0], is_left), (f"{label}_final", pts[-1], is_left)]
            windows = seq.sequence_windows(t, pts, is_left)
            repeat = OWN_WEIGHT if is_own_recording(csv_path.stem) else 1
            if repeat > 1:
                own[label] += 1
            for _ in range(repeat):
                X.append(windows)
                y += [label] * len(windows)
                groups += [participant_of(csv_path.stem)] * len(windows)
    if own:
        print("Grabaciones propias: " + ", ".join(f"{k}={v}" for k, v in sorted(own.items())))
    return X, y, groups, poses


def jitter(pts: np.ndarray, n: int) -> np.ndarray:
    """Repite una postura n cuadros con temblor natural y una deriva lenta de la mano."""
    drift = np.cumsum(rng.normal(0, 0.0015, (n, 1, 2)), axis=0)
    out = np.repeat(pts[None], n, axis=0) + rng.normal(0, 0.002, (n, *pts.shape))
    out[:, :, :2] += drift
    return out


def synthetic_negatives(frames: dict, poses: dict, n_each: int) -> tuple[list[np.ndarray], list[str], list[str]]:
    """frames: {persona: [(letra, puntos (21,3), mano_izq), ...]} de fotos estáticas.
    poses: formas inicial/final de las letras con movimiento, mismo formato.

    La mitad de los negativos usa esas formas: sin esto, una mano quieta con la forma de la X
    (que no está entre las fotos estáticas) se confunde con la X aunque no haya movimiento."""
    X, y, groups = [], [], []
    participants = [p for p, rows in frames.items() if len(rows) >= 2]
    pose_people = [p for p, rows in poses.items() if rows]
    for kind in ("estatica", "transicion"):
        for i in range(n_each):
            use_poses = i % 2 == 1 and pose_people
            person = (pose_people if use_poses else participants)[rng.integers(len(pose_people if use_poses else participants))]
            rows = poses[person] if use_poses else frames[person]
            _, a, left = rows[rng.integers(len(rows))]
            if use_poses and kind == "transicion" and person in frames:
                rows = frames[person]  # de la forma de una letra con movimiento a una estática
            if kind == "estatica":
                pts = jitter(a, seq.WINDOW + int(rng.integers(0, seq.WINDOW)))
            else:
                _, b, _ = rows[rng.integers(len(rows))]
                n_move = int(rng.integers(4, 15))  # 0.3-1 s de movimiento
                alpha = np.linspace(0, 1, n_move)[:, None, None]
                move = (1 - alpha) * a + alpha * b
                pts = np.concatenate([jitter(a, int(rng.integers(3, 10))), move, jitter(b, int(rng.integers(0, 6)))])
            vectors = seq.frame_vectors(pts, left)
            window = make_windows(vectors, seq.WINDOW, seq.TRAIN_STEP)
            X.append(window[-1:])
            y.append(kind)
            groups.append(person)
    return X, y, groups


def train(model_name: str, compare: bool) -> None:
    t0 = time.monotonic()
    Xd, yd, gd, poses = load_dynamic()
    if not yd:
        sys.exit("No hay secuencias en data/vision_dynamic.")
    per_letter = Counter(label for label in yd if label not in seq.NON_LETTER_LABELS)
    frames = load_static_frames(max_per_file=15)
    Xn, yn, gn = synthetic_negatives(frames, poses, n_each=int(np.mean(list(per_letter.values()))))

    windows = np.concatenate(Xd + Xn)
    X = seq.window_features(windows)
    y_names = yd + yn
    groups = gd + gn
    labels = sorted(set(y_names))
    y = np.array([labels.index(v) for v in y_names])
    print(f"Ventanas: {len(y)} de {len(set(groups))} personas ({time.monotonic() - t0:.0f} s)")
    for label, n in sorted(Counter(y_names).items()):
        print(f"  {label:<11} {n}")

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    if compare:
        results = {}
        print("\n=== Comparando modelos (5-fold por persona) ===")
        for name in BACKENDS:
            t1 = time.monotonic()
            scores = cross_val_score(make_backend(name, with_proba=False), X, y, groups=groups, cv=cv)
            results[name] = scores.mean()
            print(f"  {name:<4} {scores.mean():.2%} (+/- {scores.std():.2%})   {time.monotonic() - t1:.0f} s")
        model_name = max(results, key=results.get)
        print(f"Mejor: {model_name}")

    train_idx, test_idx = next(cv.split(X, y, groups))
    model = make_backend(model_name, with_proba=False)
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    test_people = sorted({groups[i] for i in test_idx})
    print(f"\n=== {model_name}: reporte con personas que no vio ({len(test_people)}) ===")
    print(classification_report(y[test_idx], pred, labels=range(len(labels)), target_names=labels, zero_division=0))
    confusions = Counter((labels[t], labels[p]) for t, p in zip(y[test_idx], pred) if t != p)
    if confusions:
        print("Confusiones más comunes (real -> predicho):")
        for (t, p), n in confusions.most_common(8):
            print(f"  {t} -> {p}: {n}")

    final = make_backend(model_name)
    final.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": final, "labels": labels, "rate_hz": seq.RATE_HZ, "window_seconds": seq.WINDOW_SECONDS},
        MODEL_PATH,
    )
    print(f"\nModelo guardado en {MODEL_PATH}")


def normalize_label(label: str) -> str:
    label = unicodedata.normalize("NFC", label.strip())
    if label.lower() in ("n~", "ene", "enie", "ñ"):
        return "Ñ"
    return label.upper() if len(label) == 1 else label.lower()


def put_text(frame, text: str, y: int, color=(255, 255, 255), scale: float = 0.7) -> None:
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def open_camera(args) -> tuple[cv2.VideoCapture, HandTracker]:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir la cámara {args.camera}. Da permiso de cámara a la terminal o prueba --camera 1.")
    return cap, HandTracker(min_confidence=args.min_conf, hand=settings.SIGNING_HAND)


def read_frame(cap, args):
    ok, frame = cap.read()
    if not ok:
        return None
    frame = cv2.flip(frame, 1)
    return enhance(frame, args.enhance) if args.enhance != "ninguno" else frame


def save_recording(label: str, person: str, rows: list, tag: str = "") -> Path:
    out_dir = settings.VISION_DYNAMIC_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{person}__{tag}{int(time.time() * 1000)}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(vision_features.CSV_HEADER)
        writer.writerows(rows)
    return path


def collect(label: str, person: str, args) -> None:
    """Graba repeticiones propias con la webcam, en el mismo formato que import_video_dataset.py."""
    out_dir = settings.VISION_DYNAMIC_DIR / label
    cap, tracker = open_camera(args)
    tip = COLLECT_TIPS.get(label, "haz la letra de corrido y deja la mano en camara ~0.5 s al terminar")
    shown = "N~" if label == "Ñ" else label
    rows, recording, t_start, last_ts, saved = [], False, 0, -1, []
    message = ""
    print(f"'{label}' para '{person}': ESPACIO empieza/termina cada repetición, D descarta la última, Q sale.")
    try:
        while True:
            frame = read_frame(cap, args)
            if frame is None:
                break
            ts = max(int(time.monotonic() * 1000), last_ts + 1)
            last_ts = ts
            det = tracker.detect(frame, ts)
            if det:
                pts, is_left = det
                draw_hand(frame, pts)
                if recording:
                    rows.append([ts - t_start, int(is_left), *np.round(pts.flatten(), 5)])
            h = frame.shape[0]
            if recording:
                put_text(frame, f"GRABANDO '{shown}' ({(ts - t_start) / 1000:.1f} s) - ESPACIO al terminar", 34, (60, 60, 255))
            else:
                put_text(frame, f"'{shown}': ESPACIO para grabar  |  guardadas: {len(saved)}", 34)
                put_text(frame, tip, 66, (200, 200, 200), 0.55)
            if not det:
                put_text(frame, "No se detecta la mano", 98, (0, 200, 255), 0.6)
            if message:
                put_text(frame, message, h - 20, (80, 220, 120), 0.6)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if not recording:
                    rows, recording, t_start, message = [], True, ts, ""
                else:
                    recording = False
                    if len(rows) < 10:
                        message = "Muy corta o sin mano; no se guardo."
                        continue
                    saved.append(save_recording(label, person, rows))
                    message = f"Guardada ({len(saved)}). ESPACIO para la siguiente."
            elif key in (ord("d"), ord("D")) and saved and not recording:
                saved.pop().unlink()
                message = f"Ultima descartada. Guardadas: {len(saved)}"
            elif should_quit(key, WINDOW_NAME):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
    print(f"{len(saved)} repeticiones de '{label}' guardadas en {out_dir}")


def expected_outcome(label: str) -> str:
    return "(nada)" if label in seq.NON_LETTER_LABELS else label


def quiz(person: str, letters: list[str], rounds: int, save: bool, args) -> None:
    """Examen en vivo: pide letras al azar y usa el mismo detector y máquina de estados que el demo.

    Cada intento se guarda como grabación propia con la letra pedida (aunque el modelo falle:
    esos son los datos que más le enseñan). 'transicion' en la lista pide mover la mano sin
    hacer letra; acierta si no se confirma nada. Deja un resumen en experiments/ para comparar
    entre entrenamientos."""
    if not MODEL_PATH.exists():
        sys.exit("No hay modelo. Corre primero: vision_dynamic.py train")
    cap, tracker = open_camera(args)
    trials = [label for label in letters for _ in range(rounds)]
    rng.shuffle(trials)
    results: list[tuple[str, str, float]] = []  # (pedida, confirmada o "(nada)", segundos)
    saved: list[Path] = []
    last_ts, message = -1, ""
    print(f"Examen: {len(trials)} intentos. D descarta el último intento guardado, ESPACIO salta, Q sale.")
    try:
        for i, target in enumerate(trials):
            detector, machine = seq.DynamicLetterDetector(MODEL_PATH), LetterStateMachine(seq.NON_LETTER_LABELS)
            shown = "N~" if target == "Ñ" else target
            expected = expected_outcome(target)
            phase_start, rows, outcome, done_at = None, [], None, None
            t_try = None
            while True:
                frame = read_frame(cap, args)
                if frame is None:
                    return
                ts = max(int(time.monotonic() * 1000), last_ts + 1)
                last_ts = ts
                phase_start = phase_start if phase_start is not None else ts
                det = tracker.detect(frame, ts)
                pts, is_left = det if det else (None, False)
                if pts is not None:
                    draw_hand(frame, pts)

                ready = ts - phase_start < QUIZ_READY_MS
                if not ready:
                    t_try = t_try if t_try is not None else ts
                # Se graba y se decide desde "prepárate": si empiezas antes de tiempo también cuenta,
                # y la grabación incluye la forma inicial quieta, como los videos del dataset.
                recording = outcome is None or ts - done_at <= QUIZ_TAIL_MS
                if pts is not None and recording:
                    rows.append([ts - phase_start, int(is_left), *np.round(pts.flatten(), 5)])
                pred = detector.push(ts, pts, is_left)
                event = machine.update(ts, pts, None, pred)
                if event and outcome is None:
                    outcome, done_at = event.letter, ts
                timeout = t_try is not None and ts - t_try >= QUIZ_TRY_MS
                if outcome is None and timeout:
                    outcome, done_at = "(nada)", ts

                h = frame.shape[0]
                task = "mueve la mano SIN hacer letra" if target == "transicion" else f"Haz: {shown}"
                put_text(frame, f"{i + 1}/{len(trials)}  {task}", 40, (0, 215, 255), 1.1)
                if ready:
                    put_text(frame, f"preparate... {(QUIZ_READY_MS - (ts - phase_start)) / 1000:.1f}", 80, (200, 200, 200))
                elif outcome is None:
                    put_text(frame, f"AHORA ({(QUIZ_TRY_MS - (ts - t_try)) / 1000:.1f} s)  {machine.state}", 80, (60, 60, 255))
                else:
                    ok_trial = outcome == expected
                    put_text(frame, ("BIEN: " if ok_trial else "MAL: ") + outcome, 80, (80, 220, 120) if ok_trial else (60, 60, 255))
                if pts is None:
                    put_text(frame, "No se detecta la mano", 115, (0, 200, 255), 0.6)
                hits = sum(o == expected_outcome(t) for t, o, _ in results)
                put_text(frame, f"aciertos {hits}/{len(results)}   {message}", h - 20, (255, 255, 255), 0.6)
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if should_quit(key, WINDOW_NAME):
                    return
                if key in (ord("d"), ord("D")) and saved:
                    saved.pop().unlink()
                    message = "ultimo intento descartado"
                if key == ord(" ") and outcome is None:
                    message = f"{shown} saltada"
                    break
                if outcome is not None and ts - done_at >= QUIZ_TAIL_MS + 700:
                    break
            if outcome is None:
                continue
            results.append((target, outcome, (done_at - phase_start) / 1000))
            message = ""
            if save and len(rows) >= 10:
                saved.append(save_recording(target, person, rows, tag="quiz"))
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        report_quiz(person, results, len(saved))


def report_quiz(person: str, results: list[tuple[str, str, float]], n_saved: int) -> None:
    if not results:
        return
    print("\nResultado (pedida -> lo que confirmó el detector):")
    by_target: dict[str, list] = {}
    for target, outcome, secs in results:
        by_target.setdefault(target, []).append((outcome, secs))
    total_hits = 0
    for target, rows in sorted(by_target.items()):
        expected = expected_outcome(target)
        hits = [s for o, s in rows if o == expected]
        total_hits += len(hits)
        wrong = Counter(o for o, _ in rows if o != expected)
        timing = f"  ~{np.mean(hits):.1f} s" if hits and expected != "(nada)" else ""
        print(f"  {target:<10} {len(hits)}/{len(rows)}{timing:<9} " + ", ".join(f"{k}: {v}" for k, v in wrong.most_common()))
    print(f"Total: {total_hits}/{len(results)} ({total_hits / len(results):.0%})")
    settings.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    log = settings.EXPERIMENTS_DIR / "quiz_log.csv"
    new = not log.exists()
    model_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(MODEL_PATH.stat().st_mtime))
    with open(log, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["fecha", "persona", "modelo_entrenado", "pedida", "confirmada", "segundos"])
        now = time.strftime("%Y-%m-%d %H:%M")
        writer.writerows([now, person, model_time, t, o, f"{s:.2f}"] for t, o, s in results)
    print(f"{n_saved} intentos guardados como datos. Historial en {log}")
    if n_saved:
        print("Reentrena para que los aprenda: python scripts/vision_dynamic.py train")


def check() -> None:
    """Pasa tus grabaciones por el detector como en vivo y dice qué letra habría confirmado."""
    if not MODEL_PATH.exists():
        sys.exit("No hay modelo. Corre primero: vision_dynamic.py train")
    results: dict[str, Counter] = {}
    for label_dir in sorted(p for p in settings.VISION_DYNAMIC_DIR.iterdir() if p.is_dir()):
        label = seq.label_from_dir(label_dir)
        for csv_path in sorted(label_dir.glob("*.csv")):
            if not is_own_recording(csv_path.stem):
                continue
            t, pts, is_left = seq.load_sequence(csv_path)
            # Igual que el demo: detector + máquina de estados (sin letras estáticas), y 0.5 s
            # de mano quieta al final por si la grabación se cortó justo al terminar el gesto.
            t = np.concatenate([t, t[-1] + np.arange(1, 16) * 33])
            pts = np.concatenate([pts, np.repeat(pts[-1:], 15, axis=0)])
            detector, machine = seq.DynamicLetterDetector(MODEL_PATH), LetterStateMachine(seq.NON_LETTER_LABELS)
            confirmed, seen = [], Counter()
            for ti, p in zip(t, pts):
                pred = detector.push(int(ti), p, is_left)
                if pred:
                    seen[pred[0]] += 1
                event = machine.update(int(ti), p, None, pred)
                if event:
                    confirmed.append(event.letter)
            outcome = "+".join(confirmed) or "(nada)"
            results.setdefault(label, Counter())[outcome] += 1
            top = ", ".join(f"{k} {v}" for k, v in seen.most_common(3))
            print(f"  {label:<10} {csv_path.stem:<28} -> {outcome:<11} predicciones: {top}")
    if not results:
        sys.exit("No hay grabaciones propias. Graba con: vision_dynamic.py collect --label J --person <nombre>")
    print("\nResumen (seña real -> lo que confirmó el detector):")
    for label, counts in sorted(results.items()):
        expected = "(nada)" if label in seq.NON_LETTER_LABELS else label
        total = sum(counts.values())
        print(f"  {label:<10} {counts[expected]}/{total} bien   " + ", ".join(f"{k}: {v}" for k, v in counts.most_common()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    p_collect = sub.add_parser("collect", help="grabar repeticiones propias con la webcam")
    p_collect.add_argument("--label", required=True, help=f"una de {COLLECT_LABELS}")
    p_collect.add_argument("--person", required=True, help="tu nombre o id, ej. omar")
    p_quiz = sub.add_parser("quiz", help="examen en vivo: pide letras, mide aciertos y guarda los intentos")
    p_quiz.add_argument("--person", required=True, help="tu nombre o id, ej. omar")
    p_quiz.add_argument("--letters", nargs="+", default=DYNAMIC_LETTERS, help="letras a pedir (puede incluir transicion)")
    p_quiz.add_argument("--rounds", type=int, default=3, help="intentos por letra")
    p_quiz.add_argument("--no-save", action="store_true", help="solo medir, no guardar los intentos")
    for p in (p_collect, p_quiz):
        p.add_argument("--camera", type=int, default=0)
        p.add_argument("--min-conf", type=float, default=0.5, help="umbral de MediaPipe; 0.3 para guante")
        p.add_argument("--enhance", choices=ENHANCE_MODES, default="ninguno", help="realce de imagen (guante oscuro)")
    sub.add_parser("check", help="probar el modelo actual con tus grabaciones")
    p_train = sub.add_parser("train", help="entrenar y evaluar")
    p_train.add_argument("--model", choices=BACKENDS, default="rf")
    p_train.add_argument("--compare", action="store_true", help="comparar todos los modelos y guardar el mejor")
    args = parser.parse_args()

    if args.mode in ("collect", "quiz"):
        person = args.person.strip().lower()
        if not re.fullmatch(r"[a-z0-9]+", person) or person.startswith("s") and person[1:].isdigit():
            sys.exit("--person solo minúsculas y números, y no del tipo s1, s2 (esos son del dataset).")
    if args.mode == "collect":
        label = normalize_label(args.label)
        if label not in COLLECT_LABELS:
            sys.exit(f"'{args.label}' no es válida. Usa una de {COLLECT_LABELS} (para Ñ puedes escribir 'enie').")
        collect(label, person, args)
    elif args.mode == "quiz":
        letters = [normalize_label(v) for v in args.letters]
        bad = [v for v in letters if v not in COLLECT_LABELS or v == "estatica"]
        if bad:
            sys.exit(f"No válidas para el examen: {bad}. Usa {DYNAMIC_LETTERS} y/o transicion.")
        quiz(person, letters, args.rounds, not args.no_save, args)
    elif args.mode == "check":
        check()
    else:
        train(args.model, args.compare)


if __name__ == "__main__":
    main()
