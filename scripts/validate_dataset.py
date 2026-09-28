"""Revisa todas las muestras de data/raw antes de entrenar.

    python scripts/validate_dataset.py

[ERROR] invalida la muestra (prepare_data.py la excluye); [WARNING] solo avisa.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from src.data.validate import validate_dataset  # noqa: E402


def main() -> None:
    valid, invalid = validate_dataset(verbose=True)
    print(f"\n{len(valid)} muestras válidas, {invalid} con errores.")
    short = sum(len(s.frames) < settings.WINDOW_SIZE for s in valid)
    if short:
        print(f"{short} muestras duran menos que la ventana ({settings.WINDOW_SIZE} lecturas); se rellenan al preparar.")
    if valid:
        print("\nMuestras válidas por seña y persona:")
        counts = Counter((s.label, s.participant) for s in valid)
        participants = sorted({s.participant for s in valid})
        print(f"  {'seña':<12}" + "".join(f"{p:>6}" for p in participants))
        for label in sorted({s.label for s in valid}):
            print(f"  {label:<12}" + "".join(f"{counts[(label, p)]:>6}" for p in participants))
    sys.exit(1 if invalid else 0)


if __name__ == "__main__":
    main()
