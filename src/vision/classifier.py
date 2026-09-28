from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


class SignClassifier:
    """Clasificador del maestro de visión: cualquier estimador de scikit-learn + nombres de letra."""

    def __init__(self, backend=None):
        self.backend = backend or RandomForestClassifier(n_estimators=200, random_state=42)
        self.label_encoder = LabelEncoder()

    def train(self, X: np.ndarray, y: list[str]) -> None:
        self.backend.fit(X, self.label_encoder.fit_transform(y))

    def predict(self, feature_vector: np.ndarray) -> tuple[str, float]:
        proba = self.backend.predict_proba(feature_vector.reshape(1, -1))[0]
        idx = int(np.argmax(proba))
        return self.label_encoder.inverse_transform([idx])[0], float(proba[idx])

    def save(self, path: Path) -> None:
        joblib.dump(
            {"backend": self.backend, "label_encoder": self.label_encoder, "labels": list(self.label_encoder.classes_)},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "SignClassifier":
        payload = joblib.load(path)
        obj = cls(backend=payload["backend"])
        obj.label_encoder = payload["label_encoder"]
        return obj
