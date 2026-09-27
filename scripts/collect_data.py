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
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from shared import config  # noqa: E402
from shared.ble_client import GloveBLEClient  # noqa: E402
from src import dataset  # noqa: E402


async def _record(label: str) -> None:
    # La ruta se calcula ahora pero el archivo solo se crea al llegar
    # la primera muestra válida (o sea, ya conectado al ESP32).
    # Así los intentos fallidos de conexión no dejan CSVs vacíos.
    path = dataset.new_recording_path(label)
    print(f"Destino: {path}")
    print("Mantén la seña sostenida y presiona Ctrl+C para terminar esta grabación.\n")
    count = 0
    f = None
    writer = None
    t0 = None

    def on_sample(valores):
        nonlocal count, f, writer, t0
        if f is None:
            f = open(path, "w", newline="")
            writer = csv.writer(f)
            t0 = time.monotonic()
            print(f"Conectado, grabando en {path}")
        count += 1
        print(f"[{count}] {valores}")
        writer.writerow(valores)
        f.flush()
        if count % 50 == 0:
            rate = count / max(time.monotonic() - t0, 1e-6)
            print(f"  ... {count} muestras ({rate:.0f}/s, esperado ~50/s)")

    client = GloveBLEClient(on_sample=on_sample)
    try:
        await client.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        if f is None:
            print("\nNo se creó ningún archivo: no llegó ninguna muestra (¿falló la conexión?).")
            # Limpia el directorio de la etiqueta si quedó vacío.
            try:
                path.parent.rmdir()
            except OSError:
                pass
            return
        f.flush()
        f.close()
        print(f"\nGrabación cerrada: {count} muestras guardadas en {path}")
        if count < config.WINDOW_SIZE:
            print(f"AVISO: solo {count} muestras; se necesitan >={config.WINDOW_SIZE} para una ventana.")


def main() -> None:
    label = input("Etiqueta de la seña a grabar: ").strip().lower()
    if not label:
        print("Etiqueta vacía, abortando.")
        return
    if label not in config.SIGN_LABELS:
        print(f"AVISO: '{label}' no está en SIGN_LABELS {config.SIGN_LABELS}.")
        print("Se grabará igual, pero considera agregarla en shared/config.py.")

    try:
        asyncio.run(_record(label))
    except KeyboardInterrupt:
        print("\nGrabación detenida por el usuario.")


if __name__ == "__main__":
    main()
