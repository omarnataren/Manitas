"""Entrena el baseline Random Forest con data/processed y lo guarda en models/random_forest/.

    python scripts/prepare_data.py   # primero
    python scripts/train_rf.py
"""

import sys
import traceback
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(write_through=True)

from config import settings  # noqa: E402
from src.data.build_dataset import load_processed  # noqa: E402
from src.models.random_forest_model import RandomForestSignModel  # noqa: E402
from src.training.evaluate import evaluate, print_metrics, save_experiment  # noqa: E402

MODEL_DIR = settings.MODELS_DIR / "random_forest"
MIN_SAMPLES_PER_LABEL = 5


def main() -> None:
    data = load_processed()
    labels = data["labels"]
    X_train, y_train, ids_train = data["train"]
    if len(X_train) == 0:
        sys.exit("No hay ventanas de train.")

    samples_per_label = Counter(labels[y] for y in dict(zip(ids_train, y_train)).values())
    print("Muestras de train por seña: " + ", ".join(f"{k}={v}" for k, v in sorted(samples_per_label.items())))
    if len(samples_per_label) < 2:
        print("AVISO: solo hay 1 seña en train; el modelo siempre predecirá esa. Graba 'reposo' y otra seña.")
    few = sorted(k for k, v in samples_per_label.items() if v < MIN_SAMPLES_PER_LABEL)
    if few:
        print(f"AVISO: señas con menos de {MIN_SAMPLES_PER_LABEL} muestras en train: {', '.join(few)}. Graba más.")

    model = RandomForestSignModel(labels)
    model.fit(X_train, y_train)

    results = {}
    for split in ("val", "test"):
        X, y, ids = data[split]
        if len(X):
            results[split] = evaluate(model, X, y, ids, labels)
            print_metrics(split, results[split])

    model.save(MODEL_DIR)
    print(f"\nModelo guardado en {MODEL_DIR}")
    if results:
        out = save_experiment("rf", labels, results, {"model": "random_forest", "data": data["meta"]})
        print(f"Métricas en {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("\nEl entrenamiento falló con el error de arriba.", flush=True)
        raise SystemExit(1)
