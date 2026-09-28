"""CNN 1D + BiGRU sobre la ventana (WINDOW_SIZE, NUM_FEATURES) normalizada.

La CNN detecta patrones cortos (flexión de un dedo, giro de muñeca) y la BiGRU
combina la secuencia completa de la ventana. TensorFlow se importa solo al
usar este modelo, para que el resto del proyecto arranque rápido.
"""

from pathlib import Path

import numpy as np

from src.data.normalize import Scaler
from src.models import model_io


def build_cnn_bigru(input_shape: tuple[int, int], num_classes: int):
    import tensorflow as tf
    from tensorflow.keras import Model, layers

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.GRU(64))(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class CnnBiGruSignModel:
    kind = "cnn_bigru"

    def __init__(self, labels: list[str], scaler: Scaler, keras_model=None):
        self.labels = labels
        self.scaler = scaler
        self.model = keras_model

    def fit(self, X_train, y_train, X_val, y_val, model_dir: Path, epochs: int = 100, batch_size: int = 32):
        import tensorflow as tf

        model_dir.mkdir(parents=True, exist_ok=True)
        self.model = build_cnn_bigru(X_train.shape[1:], len(self.labels))
        counts = np.bincount(y_train, minlength=len(self.labels))
        class_weight = {i: len(y_train) / (len(self.labels) * c) for i, c in enumerate(counts) if c > 0}
        has_val = len(X_val) > 0
        monitor = "val_loss" if has_val else "loss"
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(model_dir / "best_model.keras", monitor=monitor, save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=4),
        ]
        history = self.model.fit(
            self.scaler.transform(X_train), y_train,
            validation_data=(self.scaler.transform(X_val), y_val) if has_val else None,
            epochs=epochs, batch_size=batch_size, class_weight=class_weight,
            callbacks=callbacks, verbose=2,
        )
        return history.history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model(self.scaler.transform(X).astype("float32"), training=False).numpy()

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(model_dir / "model.keras")
        self.scaler.save(model_dir / "scaler.json")
        model_io.write_meta(model_dir, self.kind, self.labels)

    @classmethod
    def load(cls, model_dir: Path, meta: dict) -> "CnnBiGruSignModel":
        import tensorflow as tf

        return cls(meta["labels"], Scaler.load(model_dir / "scaler.json"), tf.keras.models.load_model(model_dir / "model.keras"))
