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

# --- Voz (src/inference/tts.py) ---
# macOS usa el comando `say` con esta voz (es_MX). Lista las voces con: say -v '?'
# En Windows/Linux se usa pyttsx3 y se busca automáticamente una voz es-MX / es-419 / es.
# Las etiquetas de NON_SIGN_LABELS (reposo, transicion) no se dicen.
TTS_VOICE = "Paulina"
TTS_LANG = "es_MX"
TTS_RATE = 180  # palabras por minuto

# Cómo se pronuncia cada etiqueta. Si no está aquí se dice la etiqueta con "_" -> " ".
# Las letras sueltas se escriben con su nombre: el TTS lee "y" como conjunción, etc.
SPOKEN_TEXT = {
    "por_favor": "por favor",
    "a": "a", "b": "be", "c": "ce", "d": "de", "e": "e", "f": "efe", "g": "ge",
    "h": "hache", "i": "i", "j": "jota", "k": "ka", "l": "ele", "m": "eme",
    "n": "ene", "ñ": "eñe", "o": "o", "p": "pe", "q": "cu", "r": "erre",
    "s": "ese", "t": "te", "u": "u", "v": "ve", "w": "doble u", "x": "equis",
    "y": "i griega", "z": "zeta",
}

# --- WebSocket ---
WS_HOST = "0.0.0.0"
WS_PORT = 8765

# --- Rutas ---
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"  # data/raw/<persona>/<seña>/<sample_id>/glove.csv
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
VISION_DATA_DIR = ROOT_DIR / "data" / "vision"  # letras estáticas: un cuadro por fila
VISION_DYNAMIC_DIR = ROOT_DIR / "data" / "vision_dynamic"  # letras con movimiento: un CSV por video
MODELS_DIR = ROOT_DIR / "models"

# Mano con la que se hacen las señas frente a la cámara. MediaPipe a veces confunde derecha e
# izquierda (sobre todo de perfil o con el dorso a la cámara) y eso espeja los puntos y la
# trayectoria: la J o la Z salen al revés. En vivo se usa este valor en vez de su adivinanza.
# "auto" = confiar en MediaPipe.
SIGNING_HAND = "derecha"  # "derecha", "izquierda" o "auto"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
