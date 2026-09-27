"""Loop de inferencia en tiempo real: BLE -> ventana -> RandomForest -> texto/voz.

Uso:
    python scripts/run_realtime.py

Requiere haber corrido antes scripts/train_rf.py (necesita artifacts/model_rf.pkl).
"""

import asyncio
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from shared import config  # noqa: E402
from shared.ble_client import GloveBLEClient  # noqa: E402
from src import features  # noqa: E402
from src import tts  # noqa: E402
from src.model import SignClassifier  # noqa: E402

MODEL_PATH = config.ARTIFACTS_DIR / "model_rf.pkl"
CONFIDENCE_THRESHOLD = 0.6
VOTE_WINDOW = 5  # ventanas consecutivas que deben coincidir antes de "hablar"


async def main() -> None:
    if not MODEL_PATH.exists():
        print(f"No se encontró {MODEL_PATH}. Corre scripts/train_rf.py primero.")
        return

    clf = SignClassifier.load(MODEL_PATH)
    print(f"Clases del modelo: {list(clf.label_encoder.classes_)}")
    if len(clf.label_encoder.classes_) < 2:
        print("AVISO: el modelo solo tiene 1 clase, SIEMPRE predecirá esa clase al 100%.")
        print("Graba 'reposo' (3 grabaciones) y reentrena para poder distinguir.\n")
    recent_predictions: deque = deque(maxlen=VOTE_WINDOW)
    last_spoken = None
    last_spoken_time = 0.0
    low_streak = 0
    win_count = 0
    REANNOUNCE_SECONDS = 5.0  # permite repetir la misma seña tras este silencio

    def on_window(window):
        nonlocal last_spoken, last_spoken_time, low_streak, win_count
        win_count += 1
        vector = features.window_to_vector(window)
        label, confidence = clf.predict(vector)
        means = [f"{m:.1f}" for m in np.asarray(window).mean(axis=0)]

        if confidence < CONFIDENCE_THRESHOLD:
            low_streak += 1
            print(f"[ventana #{win_count}] {label} {confidence:.0%} (bajo umbral, se ignora) medias={means}")
            # Racha de duda: limpia votos y libera la última seña para poder re-anunciar.
            if low_streak >= VOTE_WINDOW:
                recent_predictions.clear()
                last_spoken = None
            return
        low_streak = 0

        recent_predictions.append(label)
        most_common, count = Counter(recent_predictions).most_common(1)[0]
        print(f"[ventana #{win_count}] {label} {confidence:.0%} voto={most_common} {count}/{VOTE_WINDOW} medias={means}")

        now = time.monotonic()
        if count >= VOTE_WINDOW and (
            most_common != last_spoken or now - last_spoken_time > REANNOUNCE_SECONDS
        ):
            print(f"--> Seña reconocida: {most_common} ({confidence:.0%})")
            tts.speak(most_common)
            last_spoken = most_common
            last_spoken_time = now

    client = GloveBLEClient(on_window=on_window)
    await client.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nPrograma detenido por el usuario.")
