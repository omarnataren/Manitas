# LSM Smart Glove

Proyecto de hackathon para reconocer un conjunto definido de señas de la Lengua de Señas Mexicana mediante un guante con sensores.

## Objetivo del MVP

1. Capturar datos de sensores desde un ESP32.
2. Reconocer señas LSM en tiempo real con un modelo de Machine Learning.
3. Mostrar la seña reconocida como texto.
4. Convertir el texto reconocido a voz.

## Arquitectura

```
ESP32 (GuanteLSM) --BLE/CSV--> GloveBLEClient --> ventana (60x11) --> modelo --> texto --> voz
```

Cada muestra llega por BLE como CSV con 11 valores, aprox. cada 20 ms:

| Valores | Sensor | Rango |
|---|---|---|
| `izq, der, arr, abj, giroIzq, giroDer` | MPU (orientación e inclinación de la mano) | Continuo |
| `pulgar, indice, medio, anular, menique` | Sensores flex de cada dedo | 1 = muy flexionado, 3 = nada flexionado |

Una ventana son 60 muestras (~1.2 s) → 660 features al aplanarla. El formato y el tamaño de ventana se definen en `shared/config.py` (`FEATURE_NAMES`, `WINDOW_SIZE`); si electrónica cambia el número de valores, solo se actualiza ahí y se reentrena.

### Estructura

```
shared/
  config.py         # UUIDs BLE, nombres de features, WINDOW_SIZE, rutas, lista de señas
  ble_client.py     # Conexión BLE con reconexión automática + parseo CSV + ventana deslizante
src/
  features.py       # ventana (60,11) -> vector (660,) para RandomForest
  dataset.py        # lee data/raw/<seña>/*.csv y lo corta en ventanas con solapamiento
  model.py          # SignClassifier: interfaz train/predict/save/load (backend intercambiable)
  tts.py            # voz offline en español de México (say en macOS, pyttsx3 en otros), sin bloquear
scripts/
  collect_data.py   # graba datos crudos etiquetados desde el guante
  train_rf.py       # entrena baseline RandomForest + métricas para evaluar el dataset
  run_realtime.py   # inferencia en vivo con suavizado por votación + voz
data/raw/<seña>/    # grabaciones crudas del guante (no se versionan)
data/vision/        # puntos de la mano para el maestro de visión (no se versionan)
data/external/      # datasets descargados, p. ej. MSL-ABC (no se versionan)
artifacts/          # modelos entrenados .pkl y modelo de MediaPipe (no se versionan)
```

Los archivos de visión artificial se describen en [Visión artificial (MediaPipe)](#visión-artificial-mediapipe).

Decisiones clave:

- **Se guardan grabaciones crudas, no ventanas.** El corte en ventanas ocurre al entrenar, así se puede cambiar `WINDOW_SIZE` o el solapamiento sin volver a recolectar.
- **`SignClassifier` aísla el modelo.** Pasar de RandomForest a BiLSTM solo cambia `src/model.py` (y `features.py` para no aplanar); la recolección, el entrenamiento y el loop en vivo se quedan igual.
- **Suavizado en tiempo real.** Una seña se reconoce solo si 5 ventanas seguidas coinciden con confianza ≥ 60%, para evitar parpadeos. La misma seña se vuelve a decir si pasan 5 s o si hubo una racha de predicciones dudosas en medio.
- **Etiquetas controladas.** `collect_data.py` guarda la etiqueta en minúsculas y avisa si no está en `SIGN_LABELS`; `dataset.py` ignora al entrenar las carpetas que no están en esa lista y los archivos con columnas distintas o con menos de `WINDOW_SIZE` muestras.

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

### Voz

La seña o letra reconocida se muestra en pantalla y además se dice en voz alta en español de México, sin internet (`src/tts.py`):

- **macOS:** usa el comando `say` con la voz `Paulina` (es_MX). Prueba: `say -v Paulina "por favor"`. Para ver otras voces: `say -v '?' | grep es_MX`.
- **Windows / Linux:** usa `pyttsx3` y elige sola la primera voz `es-MX`, `es-419` o `es` instalada. En Windows instala el paquete de voz "Español (México)" en Configuración > Hora e idioma > Voz. En Linux: `sudo apt install espeak-ng`.
- La voz corre en un hilo aparte: no frena el BLE ni la cámara, y si llegan varias señas mientras habla solo dice la más reciente.
- Se configura en `shared/config.py`: `TTS_VOICE`, `TTS_RATE`, `SILENT_LABELS` (no se dicen, p. ej. `reposo`) y `SPOKEN_TEXT` (cómo se pronuncia cada etiqueta, p. ej. `por_favor` → "por favor", `y` → "i griega").
- En `vision_demo.py` se puede apagar con `--sin-voz`.

### Tips de recolección

- Varias grabaciones cortas por seña (idealmente de varias personas) en vez de una larga.
- Incluye una clase `reposo` (mano relajada) para que el modelo no fuerce una seña cuando no hay ninguna.
- Empieza a grabar ya con la seña hecha; la transición desde reposo mete ruido a la etiqueta.

## RandomForest → BiLSTM: cuándo cambiar

`train_rf.py` imprime ventanas y grabaciones por clase, reporte de clasificación, matriz de confusión y accuracy con cross-validation. Train y test se separan **por grabación** (las ventanas solapadas de una misma grabación son casi iguales y mezclarlas infla el accuracy), así que se necesitan ≥3 grabaciones por seña. Úsalo para decidir:

- **CV alta y estable, sin confusiones fuertes:** los datos alcanzan; RF basta para el demo.
- **CV baja pero sube al agregar grabaciones:** falta data, no modelo. Recolecta más.
- **Se confunden señas que difieren en el movimiento (no en la postura):** ahí un modelo secuencial (BiLSTM) sobre la ventana `(60, 11)` sin aplanar sí debería ayudar.

## Visión artificial (MediaPipe)

Modelo "maestro" que reconoce letras estáticas con la webcam, independiente del guante. Sirve para probar sin hardware y, más adelante, para etiquetar automáticamente los datos del guante.

```
src/vision.py              # HandTracker (MediaPipe Hand Landmarker) + dibujo de la mano
src/vision_features.py     # 21 puntos -> vector de 63 (sin normalizar rotación: la orientación distingue letras)
src/letter_rules.py        # reglas simples por dedos + orientación, para probar sin modelo
scripts/vision_demo.py     # demo en vivo: mano + letra en panel lateral
scripts/vision_teacher.py  # collect / train / live del clasificador de visión
scripts/import_image_dataset.py  # convierte un dataset de imágenes en puntos de la mano
```

### Datos usados para entrenar el maestro

**MSL-ABC — Mexican Sign Language Alphabet (static signs only)**
- Descarga: https://zenodo.org/records/10067509 (archivo `MSL-ABC.7z`, 4.8 GB)
- DOI: [10.5281/zenodo.10067509](https://doi.org/10.5281/zenodo.10067509)
- Contenido: 279,716 imágenes JPG de las 21 letras estáticas (A–I, L–P, R–U, W, Y), 20 personas, fondo verde, en 3 grupos con rotaciones crecientes de la mano. Los nombres `S<persona>-<letra>-...jpg` identifican a la persona.
- Licencia: CC-BY 4.0. Hay que citar a los autores si se usa o se redistribuye.
- Artículo: Lopez-Nava, I. H., Navarrete-López, J. A., Morfín-Chávez, R. F. (2026). *A comprehensive dataset of static and dynamic signs for the Mexican Sign Language alphabet*. Data in Brief. CICESE. https://pmc.ncbi.nlm.nih.gov/articles/PMC13241836/

El dataset **no se sube al repo** (`data/external/` y `data/vision/` están en `.gitignore`). Para reproducir:

```bash
# 1. Descargar MSL-ABC.7z de Zenodo y descomprimir en data/external/
brew install sevenzip && 7zz x data/external/MSL-ABC.7z -odata/external
# 2. Pasar las imágenes por MediaPipe (40 por persona en cada carpeta; --max-per-person 0 = todas)
python scripts/import_image_dataset.py data/external/MSL-ABC --name msl-abc
# 3. Entrenar el maestro -> artifacts/vision_teacher.pkl (compara modelos y guarda el mejor)
python scripts/vision_teacher.py train --compare
# 4. Probar en vivo
python scripts/vision_demo.py
```

Resultado con ~50,000 imágenes (40 por persona, letra y grupo), evaluando con personas que el modelo no vio: **95% de accuracy (± 0.9%)** con SVM. Comparación (`train --compare`): SVM 95.1%, Random Forest 94.8%, MLP 94.7%, Extra Trees 94.4%. Las confusiones que quedan son M↔N y R/U/V.

Features (96 por cuadro): coordenadas normalizadas, distancias entre puntas de dedos, distancias del pulgar a los otros dedos, ángulos de flexión por articulación y un indicador de índice y medio cruzados.

### Datos para letras con movimiento (pendiente)

Las letras dinámicas (J, K, Ñ, Q, X, Z) están en un dataset aparte, de los mismos autores y el mismo artículo:
- Descarga: https://doi.org/10.5281/zenodo.14689869
- Contenido: 1,200 videos MP4 (1280×720, 30 fps, ~1.8 s cada uno), 20 personas, 5 repeticiones por letra, vista frontal y de perfil (45°). Dos archivos: ~2.2 GB frontal y ~1.4 GB perfil.
- Licencia: CC-BY 4.0.

Todavía no se usa: requiere un modelo secuencial y un importador de video.

## Estado actual

- [x] Repositorio creado
- [x] Protocolo de datos definido con electrónica (CSV por BLE, 11 valores: MPU + 5 flex)
- [x] Receptor de datos por BLE
- [x] Maestro de visión (MediaPipe) entrenado con MSL-ABC: 21 letras estáticas, 95%
- [ ] Grabar guante + webcam juntos para que el maestro etiquete los datos del guante
- [ ] Definir lista final de señas (`shared/config.py`)
- [ ] Dataset recolectado
- [ ] Modelo baseline (RandomForest) entrenado
- [ ] Decidir si se pasa a BiLSTM
- [ ] Reconocimiento en tiempo real probado con el guante
- [ ] Salida de voz
