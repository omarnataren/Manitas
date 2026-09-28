"""Valida data/raw, divide por persona y genera las ventanas en data/processed.

    python scripts/prepare_data.py                        # automático: última persona = test, penúltima = val
    python scripts/prepare_data.py --val p04 --test p05   # elegir personas

Es la única fuente de verdad del preprocesamiento: los scripts de entrenamiento
leen data/processed y nunca vuelven a cortar ventanas por su cuenta.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.build_dataset import build_and_save  # noqa: E402
from src.data.split import assign_splits  # noqa: E402
from src.data.validate import validate_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val", nargs="*", help="personas para validación")
    parser.add_argument("--test", nargs="*", help="personas para test")
    args = parser.parse_args()

    samples, invalid = validate_dataset(verbose=False)
    if invalid:
        print(f"AVISO: {invalid} muestras con errores se excluyen. Detalle: python scripts/validate_dataset.py")
    if not samples:
        sys.exit("No hay muestras válidas en data/raw. Graba con: python scripts/collect_data.py")

    mapping, by_participant = assign_splits(samples, args.val, args.test)
    if not by_participant:
        print(
            "AVISO: hay menos de 3 personas; se divide por muestra dentro de cada seña. "
            "Test NO mide si funciona con personas nuevas. Graba al menos 3 personas."
        )

    meta = build_and_save(samples, mapping, by_participant)
    print(f"\nData procesada (ventana {meta['window_size']}, paso {meta['train_step']}):")
    for split, info in meta["splits"].items():
        print(f"  {split:<5} {info['windows']:>6} ventanas de {info['samples']:>4} muestras  personas: {info['participants']}")
    if meta["splits"]["test"]["windows"] == 0:
        print("AVISO: test vacío; graba más muestras o más personas para poder evaluar.")
    if meta["splits"]["val"]["windows"] == 0:
        print("Sin val (con 4+ personas se aparta una); la CNN se detendrá según la pérdida de train.")


if __name__ == "__main__":
    main()
