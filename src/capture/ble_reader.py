import asyncio
import time
from typing import Callable

from bleak import BleakClient, BleakScanner

from config import settings

SampleCallback = Callable[[int, list[float]], None]  # (timestamp_ms, valores)


class GloveBLEReader:
    """Conecta con el guante por BLE y entrega cada lectura válida a on_sample.

    No arma ventanas ni guarda archivos: eso es trabajo de quien lo usa
    (captura de muestras o inferencia en vivo). Reconecta solo si se cae.
    """

    def __init__(self, on_sample: SampleCallback, verbose: bool = False):
        self._on_sample = on_sample
        self._verbose = verbose
        self._count = 0
        self.dropped = 0

    def _handle_notification(self, _sender, data: bytearray) -> None:
        self._count += 1
        t_ms = int(time.time() * 1000)
        try:
            raw = data.decode("utf-8").strip()
            valores = [float(x) for x in raw.split(",")]
        except (UnicodeDecodeError, ValueError):
            self.dropped += 1
            if self._verbose:
                print(f"[BLE #{self._count}] trama inválida, descartada: {bytes(data)!r}")
            return

        if len(valores) != settings.NUM_FEATURES:
            self.dropped += 1
            if self._verbose:
                print(f"[BLE #{self._count}] {len(valores)} valores (se esperaban {settings.NUM_FEATURES}): {raw!r}")
            return

        if self._verbose:
            print(f"[BLE #{self._count}] {raw}")
        self._on_sample(t_ms, valores)

    async def run(self) -> None:
        """Escanea, conecta y recibe hasta Ctrl+C. Reconecta ante caídas."""
        scan_fails = 0
        while True:
            try:
                print(f"Buscando dispositivo '{settings.DEVICE_NAME}'...")
                device = await BleakScanner.find_device_by_name(settings.DEVICE_NAME, timeout=10.0)
                if not device:
                    scan_fails += 1
                    print(f"No se encontró '{settings.DEVICE_NAME}' (intento {scan_fails}).")
                    if scan_fails == 3:
                        print(
                            "Revisa: 1) ESP32 encendido y con batería, 2) cerca de esta laptop (<3 m), "
                            "3) no conectado a otra laptop/celular, 4) reinicia el ESP32 con RST."
                        )
                    await asyncio.sleep(2.0)
                    continue
                scan_fails = 0

                print(f"Encontrado ({device.address}). Conectando...")
                disconnected = asyncio.Event()
                async with BleakClient(
                    device.address, timeout=15.0, disconnected_callback=lambda _c: disconnected.set()
                ) as client:
                    await client.start_notify(settings.CHARACTERISTIC_UUID, self._handle_notification)
                    print("Conexión BLE activa (Ctrl+C para detener).\n")
                    await disconnected.wait()
                    print("\nSe perdió la conexión con el ESP32, reconectando...")
            except (KeyboardInterrupt, asyncio.CancelledError):
                raise
            except Exception as e:
                print(f"Error BLE ({type(e).__name__}): {e}. Reintentando en 3 s...")
                await asyncio.sleep(3.0)
