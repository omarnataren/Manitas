"""Entrena la CNN 1D + BiGRU con data/processed y la guarda en models/cnn_bigru/.

    python scripts/prepare_data.py   # primero
    python scripts/train_cnn_bigru.py [--epochs 100] [--batch-size 32]

Necesita bastantes más datos que Random Forest; compáralos con evaluate_models.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from src.data.build_dataset import load_processed  # noqa: E402
from src.data.normalize import Scaler  # noqa: E402
from src.models.cnn_bigru import CnnBiGruSignModel  # noqa: E402
from src.training.evaluate import evaluate, print_metrics, save_experiment  # noqa: E402

MODEL_DIR = settings.MODELS_DIR / "cnn_bigru"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data = load_processed()
    labels = data["labels"]
    X_train, y_train, _ = data["train"]
    X_val, y_val, _ = data["val"]
    if len(X_train) == 0:
        sys.exit("No hay ventanas de train.")

    scaler = Scaler.load(settings.PROCESSED_DIR / "scaler.json")
    model = CnnBiGruSignModel(labels, scaler)
    history = model.fit(X_train, y_train, X_val, y_val, MODEL_DIR, epochs=args.epochs, batch_size=args.batch_size)

    results = {}
    for split in ("val", "test"):
        X, y, ids = data[split]
        if len(X):
            results[split] = evaluate(model, X, y, ids, labels)
            print_metrics(split, results[split])

    model.save(MODEL_DIR)
    print(f"\nModelo guardado en {MODEL_DIR}")
    config = {"model": "cnn_bigru", "epochs": args.epochs, "batch_size": args.batch_size, "data": data["meta"]}
    out = save_experiment("cnn_bigru", labels, results, config, history)
    print(f"Métricas en {out}")


if __name__ == "__main__":
    main()
