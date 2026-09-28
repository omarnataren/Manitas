"""Compara todos los modelos entrenados en models/ sobre el mismo test.

    python scripts/evaluate_models.py [--split test|val]
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from src.data.build_dataset import load_processed  # noqa: E402
from src.models.model_io import dir_size_bytes, load_model  # noqa: E402
from src.training.evaluate import evaluate, print_metrics  # noqa: E402

GLOVE_MODELS = ["random_forest", "cnn_bigru"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["test", "val"], default="test")
    args = parser.parse_args()

    data = load_processed()
    X, y, ids = data[args.split]
    if len(X) == 0:
        sys.exit(f"'{args.split}' está vacío. Graba más personas o elige otras con prepare_data.py.")

    rows = []
    for name in GLOVE_MODELS:
        model_dir = settings.MODELS_DIR / name
        if not (model_dir / "meta.json").exists():
            continue
        model = load_model(model_dir)
        if model.labels != data["labels"]:
            print(f"AVISO: {name} se entrenó con otras etiquetas; reentrénalo. Se omite.")
            continue
        m = evaluate(model, X, y, ids, data["labels"])
        print_metrics(f"{name} ({args.split})", m)
        rows.append((name, m, dir_size_bytes(model_dir) / 1e6))

    if not rows:
        sys.exit("No hay modelos entrenados en models/.")
    print(f"\n{'modelo':<14}{'acc ventana':>12}{'macro F1':>10}{'acc muestra':>13}{'ms/ventana':>12}{'MB':>7}")
    for name, m, mb in rows:
        print(
            f"{name:<14}{m['accuracy']:>12.2%}{m['macro_f1']:>10.2%}{m['sample_accuracy']:>13.2%}"
            f"{m['latency_ms_per_window']:>12.2f}{mb:>7.1f}"
        )


if __name__ == "__main__":
    main()
