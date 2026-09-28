from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.data.features import window_stats
from src.models import model_io


class RandomForestSignModel:
    """Baseline: features estadísticas por ventana -> Random Forest."""

    kind = "random_forest"

    def __init__(self, labels: list[str], clf: RandomForestClassifier | None = None):
        self.labels = labels
        self.clf = clf or RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.clf.fit(window_stats(X), y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(N, W, F) -> (N, len(labels)). Clases ausentes en train quedan con probabilidad 0."""
        proba = self.clf.predict_proba(window_stats(X))
        full = np.zeros((len(X), len(self.labels)))
        full[:, self.clf.classes_] = proba
        return full

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, model_dir / "model.joblib")
        model_io.write_meta(model_dir, self.kind, self.labels)

    @classmethod
    def load(cls, model_dir: Path, meta: dict) -> "RandomForestSignModel":
        return cls(meta["labels"], joblib.load(model_dir / "model.joblib"))
