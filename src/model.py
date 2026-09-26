from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


class SignClassifier:
    """Envoltura con una interfaz mínima (train/predict/save/load).

    La idea es que scripts/train_rf.py y scripts/run_realtime.py hablen con
    esta interfaz y no con RandomForestClassifier directamente. Así, si el
    baseline de Random Forest no alcanza y hay que pasar a BiLSTM (u otro
    modelo secuencial), solo se cambia el backend de esta clase sin reescribir
    la recolección de datos, el entrenamiento ni el loop de inferencia.
    """

    def __init__(self, backend: Optional[RandomForestClassifier] = None):
        self.backend = backend or RandomForestClassifier(n_estimators=200, random_state=42)
        self.label_encoder = LabelEncoder()

    def train(self, X: np.ndarray, y: list[str]) -> None:
        y_encoded = self.label_encoder.fit_transform(y)
        self.backend.fit(X, y_encoded)

    def predict(self, feature_vector: np.ndarray) -> tuple[str, float]:
        proba = self.backend.predict_proba(feature_vector.reshape(1, -1))[0]
        idx = int(np.argmax(proba))
        label = self.label_encoder.inverse_transform([idx])[0]
        return label, float(proba[idx])

    def save(self, path: Path) -> None:
        joblib.dump({"backend": self.backend, "label_encoder": self.label_encoder}, path)

    @classmethod
    def load(cls, path: Path) -> "SignClassifier":
        payload = joblib.load(path)
        obj = cls(backend=payload["backend"])
        obj.label_encoder = payload["label_encoder"]
        return obj
