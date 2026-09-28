from pathlib import Path

from src.data.windowing import SlidingWindow
from src.models.model_io import load_model


class LivePredictor:
    """Lecturas del guante -> ventana deslizante -> modelo -> (seña, confianza, probabilidades).

    Usa el mismo windowing y el mismo preprocesamiento guardado con el modelo que
    se usaron al entrenar; aquí no se recalcula nada.
    """

    def __init__(self, model_dir: Path):
        self.model = load_model(model_dir)
        self.labels = self.model.labels
        self.window = SlidingWindow()

    def push(self, values: list[float]):
        window = self.window.push(values)
        if window is None:
            return None
        proba = self.model.predict_proba(window[None])[0]
        idx = int(proba.argmax())
        return self.labels[idx], float(proba[idx]), {label: float(p) for label, p in zip(self.labels, proba)}
