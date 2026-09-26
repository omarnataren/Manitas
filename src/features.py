import numpy as np

from shared import config


def window_to_vector(window: np.ndarray) -> np.ndarray:
    """(WINDOW_SIZE, NUM_FEATURES) -> vector plano de 120 features para el
    baseline de RandomForest (que no modela orden temporal explícito).

    Si más adelante se pasa a un modelo secuencial (BiLSTM/GRU), ese modelo
    puede seguir usando las mismas ventanas (window) sin aplanar.
    """
    assert window.shape == (config.WINDOW_SIZE, config.NUM_FEATURES), (
        f"Ventana con shape inesperado: {window.shape}"
    )
    return window.flatten()
