"""Conecta con el guante por BLE y guarda una grabación cruda etiquetada.

Uso:
    python scripts/collect_data.py

Te pide una etiqueta (debe coincidir con las que uses en el entrenamiento),
mantén la seña sostenida durante toda la grabación y para con Ctrl+C.
Cada grabación se guarda como CSV crudo en data/raw/<etiqueta>/<timestamp>.csv;
el corte en ventanas de WINDOW_SIZE ocurre después, en src/dataset.py, para
poder ajustar tamaño de ventana/solapamiento sin volver a recolectar datos.
"""

import asyncio
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from shared.ble_client import GloveBLEClient  # noqa: E402
from src import dataset  # noqa: E402


async def _record(label: str) -> None:
    path = dataset.new_recording_path(label)
    print(f"Grabando en {path}")
    print("Mantén la seña sostenida y presiona Ctrl+C para terminar esta grabación.\n")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        def on_sample(valores):
            writer.writerow(valores)
            f.flush()

        client = GloveBLEClient(on_sample=on_sample)
        await client.run()


def main() -> None:
    label = input("Etiqueta de la seña a grabar (a'): ").strip()
    if not label:
        print("Etiqueta vacía, abortando.")
        return

    try:
        asyncio.run(_record(label))
    except KeyboardInterrupt:
        print("\nGrabación detenida por el usuario.")


if __name__ == "__main__":
    main()
