from config import settings
from src.data import io


class SampleRecorder:
    """Acumula lecturas entre start() y stop() y las guarda como una muestra.

    Una muestra = una repetición de la seña. La conexión BLE sigue abierta
    entre muestras, así se graban muchas repeticiones sin reconectar.
    """

    def __init__(self, participant: str, label: str, capture_mode: str = "isolated"):
        self.participant = participant
        self.label = label
        self.capture_mode = capture_mode
        self.recording = False
        self._rows: list[tuple[int, list[float]]] = []
        self.last_values: list[float] | None = None
        self.samples_seen = 0

    def on_sample(self, t_ms: int, values: list[float]) -> None:
        self.last_values = values
        self.samples_seen += 1
        if self.recording:
            self._rows.append((t_ms, values))

    @property
    def frames(self) -> int:
        return len(self._rows)

    def start(self) -> None:
        self._rows = []
        self.recording = True

    def stop(self):
        """Guarda la muestra. Devuelve (ruta, None) o (None, motivo si se descartó)."""
        self.recording = False
        rows, self._rows = self._rows, []
        if len(rows) < settings.MIN_SAMPLE_FRAMES:
            return None, f"solo {len(rows)} lecturas (mínimo {settings.MIN_SAMPLE_FRAMES}); no se guardó"
        path = io.save_sample(self.participant, self.label, rows, {"capture_mode": self.capture_mode})
        return path, None
