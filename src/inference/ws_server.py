"""Servidor WebSocket que publica lo que reconoce el guante.

Cualquier cliente (app, página web, otro script) se conecta a
ws://<ip-de-la-laptop>:8765 y recibe mensajes JSON:

    {"type": "hello",      "labels": [...], "model": "random_forest"}
    {"type": "prediction", "label": "hola", "confidence": 0.91, "state": "CANDIDATO", "t_ms": ...}
    {"type": "sign",       "label": "hola", "confidence": 0.93, "t_ms": ...}

"prediction" llega ~10 veces por segundo (para mostrar en vivo);
"sign" solo cuando la máquina de estados confirma una seña.
"""

import json

from websockets.asyncio.server import ServerConnection, broadcast, serve

from config import settings


class SignBroadcaster:
    def __init__(self, hello: dict, host: str = settings.WS_HOST, port: int = settings.WS_PORT):
        self.host, self.port = host, port
        self._hello = json.dumps({"type": "hello", **hello}, ensure_ascii=False)
        self._clients: set[ServerConnection] = set()
        self._server = None

    async def start(self) -> None:
        self._server = await serve(self._handler, self.host, self.port)
        print(f"WebSocket escuchando en ws://{self.host}:{self.port}")

    async def _handler(self, ws: ServerConnection) -> None:
        self._clients.add(ws)
        try:
            await ws.send(self._hello)
            async for _ in ws:  # los mensajes del cliente se ignoran
                pass
        finally:
            self._clients.discard(ws)

    def send(self, message: dict) -> None:
        if self._clients:
            broadcast(self._clients, json.dumps(message, ensure_ascii=False))

    async def close(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
