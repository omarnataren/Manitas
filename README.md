# LSM Smart Glove

Proyecto de hackathon para reconocer un conjunto definido de señas de la Lengua de Señas Mexicana mediante un guante con sensores.

## Objetivo del MVP

1. Capturar datos de sensores desde un ESP32.
2. Reconocer señas LSM en tiempo real con un modelo de Machine Learning.
3. Mostrar la seña reconocida como texto.
4. Convertir el texto reconocido a voz.

## Arquitectura

```
ESP32 (GuanteLSM) --BLE/CSV--> GloveBLEClient --> ventana (20x6) --> modelo --> texto --> voz
```

Cada muestra llega por BLE como CSV con 6 valores:
`[izq, der, arr, abj, giroIzq, giroDer]`, aprox. cada 20 ms.
Una ventana son 20 muestras (~400 ms) → 120 features al aplanarla.

### Estructura

```
shared/
  config.py         # UUIDs BLE, nombres de features, WINDOW_SIZE, rutas, lista de señas
  ble_client.py     # Conexión BLE + parseo CSV + ventana deslizante (usado por collect y realtime)
src/
  features.py       # ventana (20,6) -> vector (120,) para RandomForest
  dataset.py        # lee data/raw/<seña>/*.csv y lo corta en ventanas con solapamiento
  model.py          # SignClassifier: interfaz train/predict/save/load (backend intercambiable)
  tts.py            # texto a voz offline (pyttsx3)
scripts/
  collect_data.py   # graba datos crudos etiquetados desde el guante
  train_rf.py       # entrena baseline RandomForest + métricas para evaluar el dataset
  run_realtime.py   # inferencia en vivo con suavizado por votación + voz
data/raw/<seña>/    # grabaciones crudas (no se versionan)
artifacts/          # modelos entrenados .pkl (no se versionan)
```

Decisiones clave:

- **Se guardan grabaciones crudas, no ventanas.** El corte en ventanas ocurre al entrenar, así se puede cambiar `WINDOW_SIZE` o el solapamiento sin volver a recolectar.
- **`SignClassifier` aísla el modelo.** Pasar de RandomForest a BiLSTM solo cambia `src/model.py` (y `features.py` para no aplanar); la recolección, el entrenamiento y el loop en vivo se quedan igual.
- **Suavizado en tiempo real.** Una seña se reconoce solo si 5 ventanas seguidas coinciden con confianza ≥ 60%, para evitar parpadeos.

## Cómo correr

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Recolectar: una grabación por corrida, repite varias veces por seña
python scripts/collect_data.py

# 2. Entrenar baseline y revisar métricas
python scripts/train_rf.py

# 3. Reconocimiento en vivo
python scripts/run_realtime.py
```

### Tips de recolección

- Varias grabaciones cortas por seña (idealmente de varias personas) en vez de una larga.
- Incluye una clase `reposo` (mano relajada) para que el modelo no fuerce una seña cuando no hay ninguna.
- Empieza a grabar ya con la seña hecha; la transición desde reposo mete ruido a la etiqueta.

## RandomForest → BiLSTM: cuándo cambiar

`train_rf.py` imprime ventanas y grabaciones por clase, reporte de clasificación, matriz de confusión y accuracy con cross-validation. Train y test se separan **por grabación** (las ventanas solapadas de una misma grabación son casi iguales y mezclarlas infla el accuracy), así que se necesitan ≥3 grabaciones por seña. Úsalo para decidir:

- **CV alta y estable, sin confusiones fuertes:** los datos alcanzan; RF basta para el demo.
- **CV baja pero sube al agregar grabaciones:** falta data, no modelo. Recolecta más.
- **Se confunden señas que difieren en el movimiento (no en la postura):** ahí un modelo secuencial (BiLSTM) sobre la ventana `(20, 6)` sin aplanar sí debería ayudar.

## Estado actual

- [x] Repositorio creado
- [x] Protocolo de datos definido con electrónica (CSV por BLE, 6 valores)
- [x] Receptor de datos por BLE
- [ ] Definir lista final de señas (`shared/config.py`)
- [ ] Dataset recolectado
- [ ] Modelo baseline (RandomForest) entrenado
- [ ] Decidir si se pasa a BiLSTM
- [ ] Reconocimiento en tiempo real probado con el guante
- [ ] Salida de voz
