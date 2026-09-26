# AGENT.md — LSM Smart Glove (Manitas)

Guante traductor de Lengua de Señas Mexicana (LSM) para hackathon.
Pipeline: ESP32 por BLE → ventana deslizante → RandomForest → texto → voz.

## 1. Contrato BLE (fuente de verdad: `shared/config.py`)

- `DEVICE_NAME = "GuanteLSM"` — no cambiar sin actualizar ambos lados.
- `CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"` con `PROPERTY_NOTIFY`.
- Formato: CSV de 6 floats por notificación: `[izq, der, arr, abj, giro_izq, giro_der]`.
- Frecuencia aproximada: cada ~20 ms.
- Regla: el ESP32 (Arduino) y Python (`bleak`) deben coincidir exactamente en nombre + UUID. Fuera de eso son independientes.

## 2. Ventana y features (fuente de verdad: `shared/config.py`)

- `FEATURE_NAMES = ["izq", "der", "arr", "abj", "giro_izq", "giro_der"]`, `NUM_FEATURES = 6`.
- `WINDOW_SIZE = 60` (60 muestras × ~20 ms ≈ 1200 ms por ventana).
- `window (60,6) → vector (360,)` vía `src/features.py:window_to_vector()` (flatten para RandomForest).
- Si se migra a modelo secuencial (BiLSTM/GRU), usar la ventana sin aplanar; no tocar recolección ni loop en vivo.

## 3. Estructura del repo

```
shared/config.py       # BLE, features, WINDOW_SIZE, rutas, SIGN_LABELS
shared/ble_client.py   # GloveBLEClient: BLE + parseo CSV + ventana deslizante
src/features.py        # ventana -> vector
src/dataset.py         # data/raw/<seña>/*.csv -> ventanas con solapamiento (step = WINDOW_SIZE//2)
src/model.py           # SignClassifier: train/predict/save/load (backend RF actual)
src/tts.py             # texto a voz offline (pyttsx3)
scripts/collect_data.py # graba CSV crudo etiquetado (una corrida = una grabación)
scripts/train_rf.py     # entrena baseline + métricas + guarda artifacts/model_rf.pkl
scripts/run_realtime.py # inferencia viva con voto + voz
data/raw/<seña>/        # grabaciones crudas (no versionar)
artifacts/              # modelos .pkl (no versionar)
```

- `SIGN_LABELS` actual: `["hola", "gracias", "por_favor"]` + se recomienda clase `reposo`.
- `GloveBLEClient` tiene dos callbacks: `on_sample` (crudo, lo usa collect) y `on_window` (ventana llena, lo usa realtime).

## 4. Cómo correr

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # bleak, numpy, scikit-learn, joblib, pyttsx3

python scripts/collect_data.py  # pide etiqueta, sostener seña, Ctrl+C para cerrar grabación
python scripts/train_rf.py      # requiere ≥2 grabaciones por seña para evaluar
python scripts/run_realtime.py  # requiere artifacts/model_rf.pkl
```

## 5. Flujo 2 computadoras (hardware desacoplado por BLE)

- Compañero (Arduino): sube código al ESP32 una vez con Arduino IDE, luego lo desconecta. El ESP32 se alimenta con batería y transmite solo.
- Este repo (lado ML): con `bleak` (`BleakScanner.find_device_by_name` + `BleakClient.start_notify`) se escanea, conecta y reciben notificaciones en vivo. No requiere la laptop del compañero.
- Repositorio compartido sugerido: `/arduino` (código ESP32) + `/ml` (este código Python).

## 6. Reglas de datos y ML

- Guardar grabaciones crudas, nunca ventanas. El corte a ventanas ocurre en `src/dataset.py` para poder cambiar `WINDOW_SIZE`/solapamiento sin recolectar de nuevo.
- Split train/test y CV siempre por grabación (`StratifiedGroupKFold` con `groups`), nunca por ventana: ventanas solapadas de la misma grabación son casi idénticas y mezclarlas infla el accuracy.
- `train_rf.py` exige ≥2 grabaciones por clase para evaluar y ≥3 para CV. Reporta: conteo ventanas/grabaciones por clase, classification report, matriz de confusión, accuracy CV.
- Criterio RF → BiLSTM (ver README): CV alta y estable = RF basta; CV baja que sube con más datos = falta data; confusión entre señas que difieren en movimiento (no postura) = pasar a secuencial.
- Recolección: varias grabaciones cortas por seña (ideal multi-persona), empezar ya con la seña hecha (la transición desde reposo mete ruido), incluir clase `reposo`.
- Tiempo real (`run_realtime.py`): umbral confianza 60%, voto de 5 ventanas consecutivas iguales antes de hablar, no repetir la última seña ya anunciada.

## 7. Qué NO hacer

- No cambiar `DEVICE_NAME`, `CHARACTERISTIC_UUID`, `WINDOW_SIZE` o `FEATURE_NAMES` sin actualizar ESP32 + re-entrenar.
- No hablar directamente con `RandomForestClassifier`; usar siempre la interfaz `SignClassifier`.
- No versionar `data/raw/` ni `artifacts/*.pkl`.
- No inventar modelos BLE, UUIDs, puertos ni rutas fuera de `shared/config.py`.
