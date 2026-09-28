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

# --- Voz (texto a voz) ---
# macOS usa el comando `say` con esta voz (es_MX). Lista las voces con: say -v '?'
# En Windows/Linux se usa pyttsx3 y se busca automáticamente una voz es-MX / es-419 / es.
TTS_VOICE = "Paulina"
TTS_LANG = "es_MX"
TTS_RATE = 180  # palabras por minuto

# Etiquetas que no se dicen en voz alta (p. ej. la mano en reposo).
SILENT_LABELS = {"reposo"}

# Cómo se pronuncia cada etiqueta. Si no está aquí se dice la etiqueta con "_" -> " ".
# Las letras sueltas se escriben con su nombre: el TTS lee "y" como conjunción, etc.
SPOKEN_TEXT = {
    "por_favor": "por favor",
    "a": "a", "b": "be", "c": "ce", "d": "de", "e": "e", "f": "efe", "g": "ge",
    "h": "hache", "i": "i", "j": "jota", "k": "ka", "l": "ele", "m": "eme",
    "n": "ene", "ñ": "eñe", "o": "o", "p": "pe", "q": "cu", "r": "erre",
    "s": "ese", "t": "te", "u": "u", "v": "uve", "w": "doble u", "x": "equis",
    "y": "i griega", "z": "zeta",
}
