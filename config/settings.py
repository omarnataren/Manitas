"""Única fuente de verdad del proyecto: contrato BLE, formato de datos, ventana y rutas.

Si cambia algo del guante (valores que envía, frecuencia) se actualiza aquí y se
reentrenan los modelos. Los modelos guardan una copia de FEATURE_NAMES y
WINDOW_SIZE y se niegan a cargar si no coinciden con estos valores.
"""

from pathlib import Path

# --- BLE (debe coincidir con el firmware del ESP32) ---
DEVICE_NAME = "GuanteLSM"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

# --- Formato de cada lectura del guante, en el orden en que llega por BLE ---
# Orientación (MPU): valores continuos. Dedos (flex): 1 = muy flexionado, 3 = nada flexionado.
FEATURE_NAMES = [
    "izq", "der", "arr", "abj", "giro_izq", "giro_der",
    "pulgar", "indice", "medio", "anular", "menique",
]
NUM_FEATURES = len(FEATURE_NAMES)
# Medido con collect_data: 14-16 lecturas/s. Si electrónica sube la frecuencia, cámbiala aquí:
# la ventana y los demás valores se recalculan solos (y hay que volver a preparar y entrenar).
SAMPLE_RATE_HZ = 15

# --- Ventana: lo que el modelo ve de una vez, igual al entrenar y en vivo ---
WINDOW_SECONDS = 1.2
WINDOW_SIZE = round(WINDOW_SECONDS * SAMPLE_RATE_HZ)  # lecturas por ventana
TRAIN_STEP = max(1, WINDOW_SIZE // 6)  # paso al cortar ventanas de una muestra para entrenar
INFERENCE_STRIDE = max(1, round(0.1 * SAMPLE_RATE_HZ))  # en vivo: predecir ~cada 100 ms
MIN_SAMPLE_FRAMES = max(5, round(0.4 * SAMPLE_RATE_HZ))  # muestras de menos de ~0.4 s son inválidas

# --- Señas ---
# 'reposo' (mano relajada) y 'transicion' (moviendo entre señas) son necesarias para
# que el sistema sepa cuándo NO hay seña y cuándo termina una.
SIGN_LABELS = ["reposo", "transicion", "a", "hola", "gracias", "por_favor"]
NON_SIGN_LABELS = {"reposo", "transicion"}

# --- Tiempo real (máquina de estados) ---
MIN_CONFIDENCE = 0.80
CONFIRM_WINDOWS = 3  # predicciones seguidas iguales para confirmar una seña
COOLDOWN_MS = 500  # tiempo mínimo entre dos señas confirmadas

# --- WebSocket ---
WS_HOST = "0.0.0.0"
WS_PORT = 8765

# --- Rutas ---
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"  # data/raw/<persona>/<seña>/<sample_id>/glove.csv
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
VISION_DATA_DIR = ROOT_DIR / "data" / "vision"
MODELS_DIR = ROOT_DIR / "models"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
