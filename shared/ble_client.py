import asyncio
from collections import deque
from typing import Callable, Optional

import numpy as np
from bleak import BleakClient, BleakScanner

from . import config


class GloveBLEClient:
    """Conecta con el guante por BLE y mantiene una ventana deslizante de muestras.

    - on_sample(valores: list[float]) se dispara con cada muestra cruda que llega.
    - on_window(window: np.ndarray) se dispara cuando la ventana está llena,
      con shape (config.WINDOW_SIZE, config.NUM_FEATURES).

    Se reutiliza tanto en scripts/collect_data.py (solo necesita on_sample)
    como en scripts/run_realtime.py (solo necesita on_window).
    """

    def __init__(
        self,
        on_window: Optional[Callable[[np.ndarray], None]] = None,
        on_sample: Optional[Callable[[list], None]] = None,
    ):
        self._buffer = deque(maxlen=config.WINDOW_SIZE)
        self._on_window = on_window
        self._on_sample = on_sample

    def _handle_notification(self, _sender, data: bytearray) -> None:
        try:
            valores = [float(x) for x in data.decode("utf-8").strip().split(",")]
        except ValueError:
            return  # trama incompleta o ruido en la transmisión

        if len(valores) != config.NUM_FEATURES:
            return

        if self._on_sample:
            self._on_sample(valores)

        if self._on_window:
            self._buffer.append(valores)
            if len(self._buffer) == config.WINDOW_SIZE:
                self._on_window(np.array(self._buffer))

    async def run(self) -> None:
        print(f"Buscando dispositivo '{config.DEVICE_NAME}'...")
        device = await BleakScanner.find_device_by_name(config.DEVICE_NAME, timeout=10.0)

        if not device:
            raise RuntimeError(f"No se encontró '{config.DEVICE_NAME}'. Revisa el ESP32.")

        print(f"Conectado a {device.address}. Iniciando captura...")

        async with BleakClient(device.address) as client:
            await client.start_notify(config.CHARACTERISTIC_UUID, self._handle_notification)
            print("¡Conexión BLE activa!\n")
            while True:
                await asyncio.sleep(1)
