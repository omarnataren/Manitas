"""Loop de inferencia en tiempo real: BLE -> ventana -> RandomForest -> texto/voz.

Uso:
    python scripts/run_realtime.py

Requiere haber corrido antes scripts/train_rf.py (necesita artifacts/model_rf.pkl).
"""

import asyncio
import sys
from collections import Counter, deque
from pathlib import Path

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
    recent_predictions: deque = deque(maxlen=VOTE_WINDOW)
    last_spoken = None

    def on_window(window):
        nonlocal last_spoken
        vector = features.window_to_vector(window)
        label, confidence = clf.predict(vector)

        if confidence < CONFIDENCE_THRESHOLD:
            return

        recent_predictions.append(label)
        most_common, count = Counter(recent_predictions).most_common(1)[0]

        if count >= VOTE_WINDOW and most_common != last_spoken:
            print(f"Seña reconocida: {most_common} ({confidence:.0%})")
            tts.speak(most_common)
            last_spoken = most_common

    client = GloveBLEClient(on_window=on_window)
    await client.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nPrograma detenido por el usuario.")
