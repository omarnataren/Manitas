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
        self._notif_count = getattr(self, "_notif_count", 0) + 1
        try:
            raw = data.decode("utf-8").strip()
        except UnicodeDecodeError:
            print(f"[BLE #{self._notif_count}] trama no UTF-8, descartada: {bytes(data)!r}")
            return
        print(f"[BLE #{self._notif_count}] raw: {raw!r}")
        try:
            valores = [float(x) for x in raw.split(",")]
        except ValueError:
            print(f"[BLE #{self._notif_count}] no se pudo parsear a floats, descartada")
            return  # trama incompleta o ruido en la transmisión

        if len(valores) != config.NUM_FEATURES:
            print(
                f"[BLE #{self._notif_count}] se esperaban {config.NUM_FEATURES} valores "
                f"pero llegaron {len(valores)}: {valores} (descartada)"
            )
            return

        if self._on_sample:
            self._on_sample(valores)

        if self._on_window:
            self._buffer.append(valores)
            if len(self._buffer) == config.WINDOW_SIZE:
                self._on_window(np.array(self._buffer))

    async def run(self) -> None:
        """Escanea, conecta y captura hasta Ctrl+C.

        Reconecta solo ante caídas o si el ESP32 no aparece al inicio:
        la captura (y el archivo en collect_data) sobrevive a los cortes.
        """
        scan_fails = 0
        while True:
            try:
                print(f"Buscando dispositivo '{config.DEVICE_NAME}'...")
                device = await BleakScanner.find_device_by_name(config.DEVICE_NAME, timeout=10.0)

                if not device:
                    scan_fails += 1
                    print(f"No se encontró '{config.DEVICE_NAME}' (intento {scan_fails}).")
                    if scan_fails == 3:
                        print(
                            "Revisa: 1) ESP32 encendido y con batería, "
                            "2) cerca de esta laptop (<3m), "
                            "3) no conectado a otra laptop/celular, "
                            "4) reinicia el ESP32 con el botón RST."
                        )
                    await asyncio.sleep(2.0)
                    continue
                scan_fails = 0

                print(f"Encontrado ({device.address}). Conectando...")
                disconnected = asyncio.Event()
                try:
                    async with BleakClient(
                        device.address,
                        timeout=15.0,
                        disconnected_callback=lambda _c: disconnected.set(),
                    ) as client:
                        print("Conectado. Suscribiendo notificaciones...")
                        await client.start_notify(config.CHARACTERISTIC_UUID, self._handle_notification)
                        print("¡Conexión BLE activa! (Ctrl+C para detener)\n")
                        await disconnected.wait()
                        print("\nSe perdió la conexión con el ESP32, reconectando...")
                except (KeyboardInterrupt, asyncio.CancelledError):
                    raise
            except (KeyboardInterrupt, asyncio.CancelledError):
                raise
            except Exception as e:
                print(f"Error BLE ({type(e).__name__}): {e}. Reintentando en 3s...")
                await asyncio.sleep(3.0)
