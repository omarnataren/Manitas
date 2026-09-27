from pathlib import Path

# --- BLE ---
DEVICE_NAME = "GuanteLSM"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

# --- Formato de datos: [izq, der, arr, abj, giroIzq, giroDer, pulgar, indice, medio, anular, menique] ---
# Orientación (MPU): valores continuos del sensor.
# Dedos (flex): escala 1-3 -> 1 = muy flexionado, 3 = nada flexionado.
FEATURE_NAMES = ["izq", "der", "arr", "abj", "giro_izq", "giro_der",
                 "pulgar", "indice", "medio", "anular", "menique"]
NUM_FEATURES = len(FEATURE_NAMES)
WINDOW_SIZE = 60  # 60 muestras x ~20ms = 1200ms de ventana temporal

# --- Rutas del proyecto ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "raw"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# TODO: define aquí el conjunto de señas que va a reconocer el MVP.
# Debe coincidir con las etiquetas que uses en scripts/collect_data.py.
# 'reposo' = mano relajada, necesaria para que el modelo no prediga
# una seña cuando no hay ninguna.
SIGN_LABELS = ["a", "hola", "gracias", "por_favor", "reposo"]
