"""Graba muestras del guante: una muestra = una repetición de la seña.

    python scripts/collect_data.py --participant p01 --label hola

Se conecta una vez y graba varias repeticiones seguidas:
    Enter   empieza una muestra / la termina y la guarda
    d       descarta la última muestra guardada
    q       sale

Cada muestra queda en data/raw/<persona>/<seña>/<sample_id>/ con glove.csv y metadata.json.
Graba también 'reposo' (mano quieta) y 'transicion' (moviendo la mano entre señas).
"""

import argparse
import asyncio
import re
import shutil
import sys
import threading
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from src.capture.ble_reader import GloveBLEReader  # noqa: E402
from src.capture.session import SampleRecorder  # noqa: E402

PARTICIPANT_RE = re.compile(r"^[a-z0-9]+$")


async def ticker(recorder: SampleRecorder, reader: GloveBLEReader) -> None:
    """Muestra el estado en una sola línea mientras se graba."""
    last_seen, last_t = 0, time.monotonic()
    while True:
        await asyncio.sleep(0.5)
        now = time.monotonic()
        rate = (recorder.samples_seen - last_seen) / (now - last_t)
        last_seen, last_t = recorder.samples_seen, now
        status = (
            f"GRABANDO {recorder.frames} lecturas, Enter = guardar" if recorder.recording
            else "listo, Enter = grabar"
        )
        values = ", ".join(f"{v:.1f}" for v in recorder.last_values) if recorder.last_values else "sin datos"
        print(f"\r[{status}] {rate:4.0f} lecturas/s  descartadas={reader.dropped}  [{values}]   ", end="", flush=True)


def start_keyboard_thread(loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
    # Hilo daemon: si se sale con Ctrl+C no se queda esperando un Enter.
    queue: asyncio.Queue = asyncio.Queue()

    def read_lines():
        for line in sys.stdin:
            loop.call_soon_threadsafe(queue.put_nowait, line)
        loop.call_soon_threadsafe(queue.put_nowait, "q")

    threading.Thread(target=read_lines, daemon=True).start()
    return queue


async def run(participant: str, label: str, verbose: bool) -> None:
    recorder = SampleRecorder(participant, label)
    reader = GloveBLEReader(recorder.on_sample, verbose=verbose)
    ble_task = asyncio.create_task(reader.run())
    tick_task = asyncio.create_task(ticker(recorder, reader))
    saved: list[Path] = []
    keys = start_keyboard_thread(asyncio.get_running_loop())

    print(
        f"Persona '{participant}', seña '{label}'.\n"
        "  Enter  empieza a grabar una repetición; Enter otra vez la guarda\n"
        "  d      descarta la última guardada\n"
        "  q      sale (Ctrl+C sale sin guardar la repetición en curso)\n"
    )
    try:
        while True:
            cmd = (await keys.get()).strip().lower()
            if cmd == "q":
                if recorder.recording:
                    recorder.stop()
                    print("\nMuestra en curso descartada al salir.")
                break
            if cmd == "d":
                if saved:
                    shutil.rmtree(saved.pop())
                    print(f"\nÚltima muestra descartada. Guardadas en esta sesión: {len(saved)}")
                continue
            if not recorder.recording:
                if recorder.last_values is None:
                    print("\nTodavía no llegan datos del guante; espera a que conecte.")
                    continue
                recorder.start()
                print("\n>> Grabando... haz la seña y presiona Enter al terminar.")
            else:
                path, problem = recorder.stop()
                if problem:
                    print(f"\n!! {problem}")
                else:
                    saved.append(path)
                    print(f"\n✓ {path.name} guardada ({len(saved)} en esta sesión). Enter para la siguiente.")
    finally:
        ble_task.cancel()
        tick_task.cancel()
        print(f"\nSesión terminada: {len(saved)} muestras de '{label}' para '{participant}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--participant", required=True, help="id de la persona, ej. p01 (minúsculas y números)")
    parser.add_argument("--label", required=True, help=f"seña, una de {settings.SIGN_LABELS}")
    parser.add_argument("--verbose", action="store_true", help="imprimir cada trama BLE recibida")
    args = parser.parse_args()

    participant, label = args.participant.strip().lower(), args.label.strip().lower()
    if not PARTICIPANT_RE.match(participant):
        sys.exit("--participant solo puede tener minúsculas y números (ej. p01).")
    if label not in settings.SIGN_LABELS:
        sys.exit(f"'{label}' no está en SIGN_LABELS {settings.SIGN_LABELS}. Agrégala en config/settings.py.")

    try:
        asyncio.run(run(participant, label, args.verbose))
    except KeyboardInterrupt:
        print("\nInterrumpido con Ctrl+C (una muestra en curso no se guarda; usa Enter para guardar y q para salir).")


if __name__ == "__main__":
    main()
