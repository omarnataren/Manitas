import json
from pathlib import Path

import numpy as np


class Scaler:
    """Media y desviación por sensor, calculadas SOLO con las ventanas de train.

    Se guarda junto al modelo y en vivo se carga el mismo archivo: nunca se
    recalcula con datos nuevos.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)

    @classmethod
    def fit(cls, windows: np.ndarray) -> "Scaler":
        flat = windows.reshape(-1, windows.shape[-1])
        std = flat.std(axis=0)
        return cls(flat.mean(axis=0), np.where(std > 1e-8, std, 1.0))

    def transform(self, windows: np.ndarray) -> np.ndarray:
        return (windows - self.mean) / self.std

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump({"mean": self.mean.tolist(), "std": self.std.tolist()}, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Scaler":
        with open(path) as f:
            data = json.load(f)
        return cls(data["mean"], data["std"])
