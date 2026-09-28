# LSM Smart Glove

Guante con sensores que reconoce señas de la Lengua de Señas Mexicana (LSM) en tiempo real y las convierte en texto y voz.

## Objetivo del MVP

1. Capturar datos de sensores desde un ESP32.
2. Reconocer señas LSM en tiempo real con un modelo de Machine Learning.
3. Mostrar la seña reconocida como texto y publicarla por WebSocket.
4. Convertir el texto reconocido a voz.

## Arquitectura

```
captura → validación → preparación → entrenamiento → evaluación → inferencia en vivo

ESP32 --BLE--> collect_data --> data/raw --> validate_dataset --> prepare_data --> data/processed
                                                                                     ├──> Random Forest (baseline)
                                                                                     └──> CNN 1D + BiGRU
                                                                                              ↓
ESP32 --BLE--> run_realtime: ventana deslizante → modelo → máquina de estados → texto / voz / WebSocket
```

### Datos del guante

Cada lectura llega por BLE como CSV con 11 valores, ~15 veces por segundo (medido):

| Valores | Sensor | Rango |
|---|---|---|
| `izq, der, arr, abj, giro_izq, giro_der` | MPU (orientación e inclinación de la mano) | Continuo |
| `pulgar, indice, medio, anular, menique` | Sensores flex de cada dedo | 1 = muy flexionado, 3 = nada flexionado |

El formato, la ventana, las señas y los umbrales se definen en un solo lugar: [config/settings.py](config/settings.py). Si electrónica cambia los valores que envía, se actualiza ahí y se reentrena; los modelos guardan qué columnas y ventana usaron y se niegan a cargar si no coinciden.

### Estructura

```
config/settings.py           # única fuente de verdad: BLE, columnas, ventana, señas, umbrales, rutas

src/
  capture/
    ble_reader.py            # conexión BLE con reconexión; entrega cada lectura con timestamp
    session.py               # SampleRecorder: una muestra = una repetición de la seña
  data/
    io.py                    # formato en disco de una muestra (glove.csv + metadata.json)
    validate.py              # columnas, NaN, timestamps, frecuencia, paquetes perdidos, rangos
    split.py                 # división por persona (train / val / test)
    windowing.py             # muestra -> ventanas (entrenar) y ventana deslizante (en vivo)
    normalize.py             # Scaler (media/std de train), se guarda con el modelo
    features.py              # features estadísticas por ventana para Random Forest
    build_dataset.py         # genera data/processed; única fuente del preprocesamiento
  models/
    model_io.py              # meta.json de cada modelo + carga con verificación de compatibilidad
    random_forest_model.py   # baseline
    cnn_bigru.py             # CNN 1D + BiGRU (Keras)
  training/
    evaluate.py              # métricas por ventana, por muestra y por persona + experiments/
  inference/
    predictor.py             # ventana deslizante + modelo
    state_machine.py         # REPOSO → CANDIDATO → BLOQUEADO: evita "HOLA HOLA HOLA"
    ws_server.py             # WebSocket que publica predicciones y señas confirmadas
    tts.py                   # voz offline en español de México, sin bloquear (say en macOS, pyttsx3 en otros)
  vision/                    # maestro de visión artificial (MediaPipe), ver abajo

scripts/
  collect_data.py            # grabar muestras del guante
  validate_dataset.py        # revisar data/raw antes de entrenar
  prepare_data.py            # data/raw -> data/processed (ventanas por split + scaler)
  train_rf.py                # entrenar Random Forest
  train_cnn_bigru.py         # entrenar CNN + BiGRU
  evaluate_models.py         # comparar modelos sobre el mismo test
  run_realtime.py            # reconocimiento en vivo (o --replay sin guante)
  vision_demo.py, vision_teacher.py, vision_dynamic.py,
  import_image_dataset.py, import_video_dataset.py             # visión artificial

data/raw/<persona>/<seña>/<sample_id>/   # glove.csv + metadata.json (no se versiona)
data/processed/                          # ventanas listas para entrenar (se regenera)
models/<modelo>/                         # modelo entrenado + meta.json (+ scaler.json)
experiments/<modelo>_<fecha>/            # métricas de cada entrenamiento
```

### Decisiones de diseño

- **Una muestra = una repetición de la seña.** Se graba con inicio y fin, junto con `metadata.json` (persona, seña, tiempos, frecuencia medida). Los datos crudos nunca se modifican; todo lo derivado se regenera con `prepare_data.py`.
- **Reconocimiento continuo, a la velocidad real.** Al entrenar y en vivo se usa la misma ventana de `WINDOW_SIZE` lecturas, sin estirar ni comprimir, para que el modelo vea los gestos a la misma velocidad. Las muestras más cortas que la ventana se rellenan al inicio con la mano quieta.
- **Mismo preprocesamiento en entrenamiento y en vivo.** `windowing.py` y el `scaler.json` guardado con el modelo se usan en ambos lados; `run_realtime.py` nunca recalcula nada.
- **Evaluación por persona.** Desde 3 personas, test es alguien que el modelo no vio al entrenar; desde 4, también validación. Con menos, se divide por muestra y `prepare_data.py` avisa que el resultado no mide personas nuevas.
- **Máquina de estados en vivo.** Una seña se confirma con 3 predicciones seguidas iguales y confianza ≥ 80%. Después no se repite hasta ver `reposo`, `transicion` o una racha de predicciones dudosas.
- **Random Forest como baseline.** Si la CNN + BiGRU no le gana, el problema suele estar en los datos (pocas personas, etiquetas, normalización), no en el modelo.

## Cómo correr

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Grabar: se conecta una vez; Enter empieza/termina cada repetición, d descarta, q sale
python scripts/collect_data.py --participant p01 --label hola
python scripts/collect_data.py --participant p01 --label reposo
python scripts/collect_data.py --participant p01 --label transicion

# 2. Revisar y preparar
python scripts/validate_dataset.py
python scripts/prepare_data.py              # o --val p04 --test p05 para elegir personas

# 3. Entrenar y comparar
python scripts/train_rf.py
python scripts/train_cnn_bigru.py
python scripts/evaluate_models.py

# 4. En vivo (WebSocket en ws://<ip-de-la-laptop>:8765)
python scripts/run_realtime.py                      # Random Forest
python scripts/run_realtime.py --model cnn_bigru
python scripts/run_realtime.py --replay data/raw/p02 --no-tts   # probar sin guante
```

### Tips de recolección

- Varias repeticiones por seña (10+) y de varias personas: la evaluación mide cómo funciona con personas nuevas, así que cada persona extra vale más que muchas repeticiones de la misma.
- Graba siempre `reposo` (mano relajada) y `transicion` (moviendo la mano entre señas): son las que le dicen al sistema cuándo no hay seña y cuándo termina una.
- Empieza a grabar justo antes de la seña y termina justo al acabar.
- `collect_data.py` muestra las lecturas por segundo; si no coincide con `SAMPLE_RATE_HZ` (hoy 15), revisa el guante o actualiza ese valor.
- Enter empieza a grabar y Enter otra vez guarda. Ctrl+C sale **sin** guardar la repetición en curso.

### WebSocket

`run_realtime.py` publica en `ws://<ip>:8765` mensajes JSON:

```json
{"type": "hello",      "labels": ["reposo", "transicion", "hola"], "model": "random_forest"}
{"type": "prediction", "label": "hola", "confidence": 0.91, "state": "CANDIDATO", "t_ms": 1790554273905}
{"type": "sign",       "label": "hola", "confidence": 0.93, "t_ms": 1790554274193}
```

`prediction` llega ~10 veces por segundo, para mostrar en vivo qué ve el modelo; `sign` llega solo cuando se confirma una seña.

### Voz

Las señas confirmadas (`run_realtime.py`) y las letras confirmadas (`vision_demo.py`) se muestran en pantalla y además se dicen en voz alta en español de México, sin internet (`src/inference/tts.py`). Se apaga con `--no-tts`.

- **macOS:** usa el comando `say` con la voz `Paulina` (es_MX). Prueba: `say -v Paulina "por favor"`. Otras voces: `say -v '?' | grep es_MX`.
- **Windows / Linux:** usa `pyttsx3` y elige sola la primera voz `es-MX`, `es-419` o `es` instalada. En Windows instala el paquete de voz "Español (México)" en Configuración > Hora e idioma > Voz. En Linux: `sudo apt install espeak-ng`.
- La voz corre en un hilo aparte: no frena el BLE ni la cámara, y si llegan varias señas mientras habla solo dice la más reciente.
- Se configura en `config/settings.py`: `TTS_VOICE`, `TTS_RATE` y `SPOKEN_TEXT` (cómo se pronuncia cada etiqueta, p. ej. `por_favor` → "por favor", `y` → "i griega"). Las de `NON_SIGN_LABELS` (`reposo`, `transicion`) nunca se dicen.

## Visión artificial (MediaPipe)

Modelo "maestro" que reconoce letras estáticas con la webcam, independiente del guante. Sirve para probar sin hardware y, más adelante, para etiquetar automáticamente los datos del guante.

```
src/vision/tracker.py         # HandTracker (MediaPipe Hand Landmarker) + dibujo de la mano
src/vision/features.py        # 21 puntos -> 96 features (sin normalizar rotación: la orientación distingue letras)
src/vision/classifier.py      # clasificador de scikit-learn con nombres de letra
src/vision/letter_rules.py    # reglas simples por dedos + orientación, para probar sin modelo
scripts/vision_demo.py        # demo en vivo: mano + letra en panel lateral
scripts/vision_teacher.py     # collect / train / live del clasificador de visión
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
# 3. Entrenar el maestro -> models/vision/teacher.pkl (compara modelos y guarda el mejor)
python scripts/vision_teacher.py train --compare
# 4. Probar en vivo
python scripts/vision_demo.py
```

Resultado con ~50,000 imágenes (40 por persona, letra y grupo), evaluando con personas que el modelo no vio: **95% de accuracy (± 0.9%)** con SVM. Comparación (`train --compare`): SVM 95.1%, Random Forest 94.8%, MLP 94.7%, Extra Trees 94.4%. Las confusiones que quedan son M↔N y R/U/V.

Features (96 por cuadro): coordenadas normalizadas, distancias entre puntas de dedos, distancias del pulgar a los otros dedos, ángulos de flexión por articulación y un indicador de índice y medio cruzados.

### Letras con movimiento (J, K, Ñ, Q, X, Z)

Una sola foto no basta para estas letras: el modelo mira los últimos 2 s de la mano. La secuencia se remuestrea por tiempo a 15 cuadros/s (da igual si la cámara va a 30 o 60 fps) y se resume en la forma de la mano al inicio, medio y final, cuánto cambia y la trayectoria de la muñeca. Además de las 6 letras aprende dos clases negativas (`estatica` y `transicion`), generadas a partir de las fotos de MSL-ABC, para no confundir cualquier movimiento con una letra.

```
src/vision/sequence.py            # remuestreo, ventanas, features y DynamicLetterDetector (en vivo)
src/vision/teacher_data.py        # carga las fotos estáticas para generar los negativos
scripts/import_video_dataset.py   # videos -> secuencias de puntos (data/vision_dynamic/)
scripts/vision_dynamic.py         # entrena y evalúa por persona -> models/vision/dynamic.pkl
```

**Datos: MSL dynamic signs**, de los mismos autores y el mismo artículo que MSL-ABC:
- Descarga: https://doi.org/10.5281/zenodo.14689869
- Contenido: videos MP4 (900×900, 60 fps, mediana 1.8 s), 20 personas, 5 repeticiones por letra, vista frontal y de perfil (45°). Dos archivos: ~2.2 GB frontal y ~1.4 GB perfil. Se usa solo la frontal (621 videos), que es como ve la webcam.
- Licencia: CC-BY 4.0.

```bash
# 1. Descomprimir el .7z frontal (en cualquier carpeta)
7zz x MSL-dynamic-signs-frontal-view.7z
# 2. Extraer los puntos de la mano de cada video (~10 min)
python scripts/import_video_dataset.py ~/Downloads/MSL-dynamic-signs --name msl-dyn --view frontal
# 3. Entrenar (necesita también data/vision de MSL-ABC para los negativos)
python scripts/vision_dynamic.py train --compare
# 4. El demo usa el modelo automáticamente: las letras con movimiento salen en naranja
python scripts/vision_demo.py
```

Resultado evaluando con personas que el modelo no vio: **93% (± 3%)** por ventana con Random Forest (Extra Trees 93%, SVM 92%, MLP 91%). Pasando los 60 videos de test por el demo completo: 51/60 bien; los fallos son casi siempre "no detectó nada" en lugar de otra letra. Ventana probada: 1.5 s → 94%, 2.0 s → 96%, 2.5 s → 97% (con los negativos anteriores); se eligió 2.0 s por estar cerca de la duración típica de la seña.

Los negativos incluyen la forma inicial y final de cada letra con movimiento sostenida sin moverse: sin eso, una mano quieta con forma de X se confundía con la X.

**Con tu cámara:** el dataset es de otras personas y otra cámara. Si en vivo no detecta bien, graba repeticiones propias; se suman al entrenamiento con 3 veces más peso:

```bash
python scripts/vision_dynamic.py collect --label J --person omar      # 10-15 por letra; ESPACIO empieza/termina, D descarta
python scripts/vision_dynamic.py collect --label transicion --person omar
python scripts/vision_dynamic.py collect --label estatica --person omar
python scripts/vision_dynamic.py check    # antes de reentrenar: qué detecta hoy en cada grabación
python scripts/vision_dynamic.py train
```

**Ciclo rápido (examen en vivo):** `quiz` pide letras al azar, dice si el detector acertó (mismo detector y máquina de estados que el demo) y guarda cada intento como grabación propia con la letra pedida. Probar y juntar datos es lo mismo; reentrenar tarda ~6 s.

```bash
python scripts/vision_dynamic.py quiz --person omar                    # 6 letras x 3 intentos; D descarta el último si lo hiciste mal
python scripts/vision_dynamic.py train                                 # aprende los intentos
python scripts/vision_dynamic.py quiz --person omar                    # ¿subió el %?
python scripts/vision_dynamic.py quiz --person omar --letters Q X transicion --rounds 5   # enfocarse en las que fallan
python scripts/vision_dynamic.py quiz --person omar --no-save          # medición final sin tocar los datos
```

Cada examen se agrega a `experiments/quiz_log.csv` (fecha, fecha del modelo, letra pedida, confirmada, segundos) para comparar entre entrenamientos. `transicion` pide mover la mano sin hacer letra; acierta si no se confirma nada (mide falsos positivos). Con guante: `--min-conf 0.3 --enhance ambos`.

Para probar sin webcam: `python scripts/vision_demo.py --video ~/Downloads/MSL-dynamic-signs/test` (o un video tuyo).

### Cuándo se confirma una letra en el demo

`src/vision/letter_state.py` decide cuándo una predicción cuenta, para que no sea instantáneo:

| Seña | Se confirma cuando | Tiempo |
|---|---|---|
| Letra estática | La mano está quieta y la misma letra aparece en ≥80% de los cuadros | Sostenerla 0.7 s (barra de progreso) |
| Letra con movimiento | Durante el gesto el detector vota (confianza ≥70%); al detenerse la mano 0.2 s gana la letra con más votos. 4 votos seguidos iguales confirman sin esperar | Al terminar el gesto, ~0.2–0.5 s después |
| Siguiente letra | Mínimo 0.5 s después de la anterior | — |

- Con la mano en movimiento no se confirman letras estáticas (la J empieza como I).
- La misma letra estática no se repite hasta mover la mano o sacarla de cuadro; otra letra distinta sí.
- Tras una letra con movimiento, queda bloqueado hasta que el detector deja de verla: no se repite y su forma final (la Z termina como D) no cuenta como estática.
- Si haces una pausa de más de 0.7 s en la forma inicial de una letra con movimiento, esa forma se confirma como su propia letra (p. ej. I antes de la J). Hazlas de corrido.

Los umbrales están al inicio de `letter_state.py` (`STATIC_HOLD_MS`, `MOTION_THRESHOLD`, etc.); el panel muestra el estado y el movimiento en palmas/s para calibrarlos.

Limitación: los negativos (`estatica`, `transicion`) son sintéticos; con movimientos reales entre letras puede haber falsos positivos. Si pasa, graba `transicion` y `estatica` reales con `vision_dynamic.py collect`.

## Estado actual

- [x] Protocolo de datos definido con electrónica (CSV por BLE, 11 valores: MPU + 5 flex)
- [x] Receptor BLE con reconexión
- [x] Estructura de pipeline: captura → validación → preparación → entrenamiento → evaluación → inferencia
- [x] Random Forest y CNN + BiGRU implementados (probados con datos sintéticos)
- [x] Tiempo real con máquina de estados, voz y WebSocket (probado con `--replay`)
- [x] Maestro de visión (MediaPipe) entrenado con MSL-ABC: 21 letras estáticas, 95%
- [x] Letras con movimiento (J, K, Ñ, Q, X, Z) con la cámara: 96%
- [ ] Definir lista final de señas (`config/settings.py`)
- [ ] Grabar dataset real con el guante (varias personas, incluyendo `reposo` y `transicion`)
- [ ] Confirmar la frecuencia real del guante (`SAMPLE_RATE_HZ`)
- [ ] Comparar Random Forest vs CNN + BiGRU con datos reales
- [ ] Probar el reconocimiento en vivo con el guante
- [ ] Grabar guante + webcam juntos para que el maestro etiquete los datos del guante
