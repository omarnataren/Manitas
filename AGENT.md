# AGENT.md — LSM Smart Glove (Manitas)

Guante traductor de Lengua de Señas Mexicana (LSM).
Pipeline: captura → validación → preparación → entrenamiento → evaluación → inferencia en vivo.
En vivo: ESP32 por BLE → ventana deslizante → modelo → máquina de estados → texto / voz / WebSocket.

## 1. Contrato BLE (fuente de verdad: `config/settings.py`)

- `DEVICE_NAME = "GuanteLSM"` y `CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"` (`PROPERTY_NOTIFY`). El ESP32 y Python deben coincidir exactamente.
- Formato: CSV de 11 floats por notificación: `[izq, der, arr, abj, giro_izq, giro_der, pulgar, indice, medio, anular, menique]`.
- Dedos en escala 1-3: 1 = muy flexionado, 3 = nada flexionado.
- Frecuencia: `SAMPLE_RATE_HZ = 15` (medida: 14-16 lecturas/s). `collect_data.py` la muestra en vivo y la guarda en `metadata.json`.

## 2. Ventana y señas (fuente de verdad: `config/settings.py`)

- Todo se define en segundos y se convierte con `SAMPLE_RATE_HZ`: `WINDOW_SECONDS = 1.2` → `WINDOW_SIZE = 18` lecturas a 15 Hz. Misma ventana al entrenar y en vivo, a la velocidad real (sin interpolar).
- `TRAIN_STEP` (paso al cortar ventanas, ~1/6 de ventana), `INFERENCE_STRIDE` (predecir ~cada 100 ms) y `MIN_SAMPLE_FRAMES` (~0.4 s) se derivan de la frecuencia. Si cambia la frecuencia: actualizar `SAMPLE_RATE_HZ`, volver a correr `prepare_data.py` y reentrenar.
- Muestras más cortas que la ventana se rellenan al inicio repitiendo la primera lectura (`src/data/windowing.py`).
- `SIGN_LABELS` incluye `reposo` y `transicion`; son obligatorias para que la máquina de estados funcione.
- Umbrales en vivo: `MIN_CONFIDENCE = 0.80`, `CONFIRM_WINDOWS = 3`, `COOLDOWN_MS = 500`.

## 3. Estructura

```
config/settings.py              # única fuente de verdad
src/capture/ble_reader.py       # GloveBLEReader(on_sample(t_ms, valores)); reconecta solo
src/capture/session.py          # SampleRecorder: start/stop -> guarda una muestra
src/data/io.py                  # data/raw/<persona>/<seña>/<sample_id>/{glove.csv, metadata.json}
src/data/validate.py            # ERROR invalida la muestra, WARNING solo avisa
src/data/split.py               # por persona: 3 personas -> test=1; 4+ -> val=1 y test=1
src/data/windowing.py           # make_windows (entrenar) y SlidingWindow (en vivo)
src/data/normalize.py           # Scaler con media/std de train
src/data/features.py            # window_stats: features estadísticas para Random Forest
src/data/build_dataset.py       # build_and_save / load_processed (data/processed)
src/models/model_io.py          # meta.json + load_model() con verificación de columnas/ventana
src/models/random_forest_model.py, src/models/cnn_bigru.py
src/training/evaluate.py        # métricas por ventana, muestra y persona; guarda experiments/
src/inference/predictor.py      # LivePredictor(model_dir).push(valores)
src/inference/state_machine.py  # REPOSO -> CANDIDATO -> BLOQUEADO
src/inference/ws_server.py      # SignBroadcaster (websockets), puerto WS_PORT
src/inference/tts.py            # pyttsx3
src/vision/                     # maestro de visión (MediaPipe): tracker, features, classifier, letter_rules
scripts/                        # un script por paso del pipeline (ver README)
```

Interfaz común de modelos del guante: `labels`, `predict_proba(X: (N, W, F)) -> (N, len(labels))`, `save(dir)`, `load(dir, meta)`.

## 4. Cómo correr

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # incluye tensorflow (~500 MB)

python scripts/collect_data.py --participant p01 --label hola   # Enter empieza/termina, d descarta, q sale
python scripts/validate_dataset.py
python scripts/prepare_data.py                                   # --val pXX --test pYY opcional
python scripts/train_rf.py
python scripts/train_cnn_bigru.py
python scripts/evaluate_models.py
python scripts/run_realtime.py [--model cnn_bigru] [--no-tts] [--verbose] [--replay data/raw/p02]
```

## 5. Flujo 2 computadoras (hardware desacoplado por BLE)

- Compañero (Arduino): sube código al ESP32 una vez con Arduino IDE, luego lo desconecta. El ESP32 se alimenta con batería y transmite solo.
- Este repo (lado ML): con `bleak` se escanea, conecta y reciben notificaciones en vivo. No requiere la laptop del compañero.
- Otras apps consumen las señas por WebSocket (`ws://<ip>:8765`, mensajes `hello` / `prediction` / `sign`).

## 6. Reglas de datos y ML

- Los datos crudos (`data/raw`) nunca se modifican. Todo lo derivado se regenera con `prepare_data.py`.
- `prepare_data.py` es la única fuente del preprocesamiento. Los scripts de entrenamiento leen `data/processed`; `run_realtime.py` usa `windowing.py` y el `scaler.json` guardado con el modelo.
- Evaluar siempre por persona (o, con menos de 3, por muestra), nunca mezclando ventanas de una misma muestra entre splits.
- El Scaler se calcula solo con train y se guarda con el modelo; nunca se recalcula en vivo.
- Random Forest es el baseline: si la CNN + BiGRU no lo supera, revisar datos antes que el modelo.
- Recolección: 10+ repeticiones por seña, varias personas, siempre `reposo` y `transicion`.

## 7. Qué NO hacer

- No cambiar `DEVICE_NAME`, `CHARACTERISTIC_UUID`, `FEATURE_NAMES` o `WINDOW_SIZE` sin actualizar el ESP32 y reentrenar (los modelos viejos se niegan a cargar).
- No cortar ventanas ni normalizar fuera de `src/data/`; no duplicar preprocesamiento en `run_realtime.py`.
- No interpolar/estirar muestras en el tiempo: en vivo la ventana es a la velocidad real.
- No versionar `data/raw`, `data/processed`, `models/` ni `experiments/`.
- No inventar UUIDs, puertos ni rutas fuera de `config/settings.py`.
