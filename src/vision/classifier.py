from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


BACKENDS = ["rf", "et", "mlp", "svm"]


def make_backend(name: str, with_proba: bool = True):
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    if name == "et":
        return ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 128), early_stopping=True, max_iter=300, random_state=42),
        )
    if name == "svm":
        svc = SVC(C=10, gamma="scale", random_state=42)
        # La calibración (para tener confianza por letra) hace el entrenamiento ~5x más lento;
        # solo se activa para el modelo final.
        return make_pipeline(StandardScaler(), CalibratedClassifierCV(svc, ensemble=False) if with_proba else svc)
    raise ValueError(f"Modelo desconocido: {name}")


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
