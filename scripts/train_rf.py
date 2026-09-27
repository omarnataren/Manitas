"""Entrena el baseline de RandomForest sobre lo recolectado en data/raw/
y reporta métricas para decidir si el dataset ya es suficiente o si conviene
pasar a un modelo secuencial (BiLSTM) antes de seguir recolectando.

Uso:
    python scripts/train_rf.py
"""

import sys
import traceback
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
# Sin buffer: cada print sale al instante aunque el proceso muera despues.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(write_through=True)

from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score  # noqa: E402

from shared import config  # noqa: E402
from src import dataset  # noqa: E402
from src.model import SignClassifier  # noqa: E402

MIN_SAMPLES_TO_TRAIN = 20
MIN_RECORDINGS_PER_CLASS_FOR_CV = 3


def main() -> None:
    print("=== train_rf: iniciando ===", flush=True)
    print(f"Python: {sys.version.split()[0]} | cwd: {Path.cwd()}", flush=True)
    # Se separa por grabación: las ventanas solapadas de una misma grabación
    # son casi idénticas y mezclarlas entre train y test infla el accuracy.
    X, y, groups = dataset.build_dataset()

    if len(y) == 0:
        print("No hay datos en data/raw/. Corre scripts/collect_data.py primero.")
        return

    counts = Counter(y)
    recordings_per_class = Counter(g.split("/")[0] for g in set(groups))
    print(f"Dataset: {len(y)} ventanas, {len(set(groups))} grabaciones, {len(counts)} clases")
    for label, n in sorted(counts.items()):
        print(f"  - {label}: {n} ventanas en {recordings_per_class[label]} grabaciones")

    pobres = sorted(l for l, n in counts.items() if n < 10)
    if pobres:
        print(f"\nAVISO: clases con <10 ventanas ({', '.join(pobres)}): graba mas tomas de esas.")

    if len(counts) < 2:
        print("\nSolo hay 1 clase: el modelo resultante predecirá esa clase al 100% siempre.")
        print("Graba al menos otra clase ('reposo') y vuelve a entrenar.")
        if min(recordings_per_class.values()) < 1 or len(y) < MIN_SAMPLES_TO_TRAIN:
            print("Además hay muy pocos datos incluso para eso. Abortando sin guardar.")
            return
        clf = SignClassifier()
        clf.train(X, list(y))
        config.ARTIFACTS_DIR.mkdir(exist_ok=True)
        out_path = config.ARTIFACTS_DIR / "model_rf.pkl"
        clf.save(out_path)
        print(f"\nModelo de 1 clase guardado en {out_path} (solo útil para pruebas, no distingue nada).")
        return

    faltantes = sorted(l for l, c in recordings_per_class.items() if c < 2)
    if len(y) < MIN_SAMPLES_TO_TRAIN or faltantes:
        print(f"\nMuy pocos datos: estas clases tienen <2 grabaciones: {', '.join(faltantes) or 'ninguna'}.")
        print("Se necesitan al menos 2 grabaciones por sena para evaluar.")
        return

    min_recordings = min(recordings_per_class.values())
    splitter = StratifiedGroupKFold(n_splits=min(4, min_recordings), shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]

    clf = SignClassifier()
    clf.train(X_train, y_train)

    y_pred_idx = clf.backend.predict(X_test)
    y_pred = clf.label_encoder.inverse_transform(y_pred_idx)

    print("\n=== Reporte sobre el split de test ===")
    print(classification_report(y_test, y_pred))
    print("Matriz de confusion (filas=real, columnas=predicho):")
    labels_sorted = sorted(counts.keys())
    print(labels_sorted)
    print(confusion_matrix(y_test, y_pred, labels=labels_sorted))

    if min_recordings >= MIN_RECORDINGS_PER_CLASS_FOR_CV:
        n_splits = min(5, min_recordings)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_encoded = clf.label_encoder.transform(y)
        scores = cross_val_score(clf.backend, X, y_encoded, groups=groups, cv=cv)
        print(f"\nAccuracy {n_splits}-fold CV por grabacion: {scores.mean():.2%} (+/- {scores.std():.2%})")
        print("Si este numero es alto y estable, el dataset probablemente ya alcanza para RandomForest.")
        print("Si es bajo, inestable entre folds, o hay clases que se confunden mucho entre si,")
        print("considera recolectar mas datos o pasar a un modelo secuencial (BiLSTM).")
    else:
        print(f"\nPara cross-validation se necesitan >={MIN_RECORDINGS_PER_CLASS_FOR_CV} grabaciones por sena.")

    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    out_path = config.ARTIFACTS_DIR / "model_rf.pkl"
    clf.save(out_path)
    print(f"\nModelo guardado en {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("\nEl train fallo con el error de arriba. Pegalo para diagnosticarlo.", flush=True)
        raise SystemExit(1)
