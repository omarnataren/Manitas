"""Reconocimiento en vivo: guante -> ventana -> modelo -> máquina de estados -> texto/voz/WebSocket.

    python scripts/run_realtime.py                          # Random Forest, con voz y WebSocket
    python scripts/run_realtime.py --model cnn_bigru
    python scripts/run_realtime.py --no-tts --verbose       # sin voz, imprime cada predicción
    python scripts/run_realtime.py --replay data/raw/p01    # sin guante: reproduce muestras grabadas

El WebSocket queda en ws://<ip-de-esta-laptop>:8765 (ver src/inference/ws_server.py).
"""

import argparse
import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from src.capture.ble_reader import GloveBLEReader  # noqa: E402
from src.data import io  # noqa: E402
from src.inference.predictor import LivePredictor  # noqa: E402
from src.inference.state_machine import SignStateMachine  # noqa: E402
from src.inference.ws_server import SignBroadcaster  # noqa: E402


async def replay(path: Path, on_sample) -> None:
    """Alimenta el pipeline con muestras grabadas, a la frecuencia del guante."""
    sample_dirs = sorted(p.parent for p in path.rglob("glove.csv"))
    if not sample_dirs:
        raise SystemExit(f"No hay muestras (glove.csv) en {path}")
    period = 1 / settings.SAMPLE_RATE_HZ
    for sample_dir in sample_dirs:
        sample = io.load_sample(sample_dir)
        print(f"\n[replay] {sample.sample_id} (seña real: {sample.label})")
        # Mano quieta en la postura inicial antes de la seña, como el relleno al entrenar.
        lead_in = [sample.frames[0]] * (settings.WINDOW_SIZE // 2)
        for values in [*lead_in, *sample.frames]:
            on_sample(int(time.time() * 1000), list(values))
            await asyncio.sleep(period)
    print("\n[replay] terminado")


async def run(args) -> None:
    predictor = LivePredictor(settings.MODELS_DIR / args.model)
    machine = SignStateMachine()
    print(f"Modelo: {args.model} | señas: {predictor.labels}")

    broadcaster = None
    if not args.no_ws:
        broadcaster = SignBroadcaster({"labels": predictor.labels, "model": args.model}, port=args.port)
        await broadcaster.start()

    speaker = None
    if not args.no_tts:
        from src.inference import tts

        executor = ThreadPoolExecutor(max_workers=1)  # la voz no debe bloquear la lectura BLE
        speaker = lambda text: executor.submit(tts.speak, text)  # noqa: E731

    last_pred_sent = 0.0

    def on_sample(t_ms: int, values: list[float]) -> None:
        nonlocal last_pred_sent
        result = predictor.push(values)
        if result is None:
            return
        label, confidence, _ = result
        event = machine.update(label, confidence, t_ms)
        if args.verbose:
            print(f"  {label:<12} {confidence:5.0%}  estado={machine.state}")
        if broadcaster and time.monotonic() - last_pred_sent >= 0.1:
            broadcaster.send({"type": "prediction", "label": label, "confidence": round(confidence, 3),
                              "state": machine.state, "t_ms": t_ms})
            last_pred_sent = time.monotonic()
        if event:
            print(f"--> {event.label} ({event.confidence:.0%})")
            if broadcaster:
                broadcaster.send({"type": "sign", "label": event.label,
                                  "confidence": round(event.confidence, 3), "t_ms": event.t_ms})
            if speaker:
                speaker(event.label.replace("_", " "))

    try:
        if args.replay:
            await replay(args.replay, on_sample)
        else:
            await GloveBLEReader(on_sample, verbose=args.verbose_ble).run()
    finally:
        if broadcaster:
            await broadcaster.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="random_forest", choices=["random_forest", "cnn_bigru"])
    parser.add_argument("--no-tts", action="store_true", help="sin voz")
    parser.add_argument("--no-ws", action="store_true", help="sin WebSocket")
    parser.add_argument("--port", type=int, default=settings.WS_PORT)
    parser.add_argument("--verbose", action="store_true", help="imprimir cada predicción")
    parser.add_argument("--verbose-ble", action="store_true", help="imprimir cada trama BLE")
    parser.add_argument("--replay", type=Path, help="carpeta de muestras a reproducir en lugar del guante")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nDetenido.")


if __name__ == "__main__":
    main()
