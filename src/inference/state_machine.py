"""Convierte predicciones por ventana en señas confirmadas.

    REPOSO ──(seña con confianza)──> CANDIDATO ──(N ventanas iguales)──> se emite la seña
       ^                                                                     │
       └──(reposo, transición o N dudas)─── BLOQUEADO <──────────────────────┘

En BLOQUEADO la misma seña no se repite aunque la persona mantenga la postura.
"""

from dataclasses import dataclass

from config import settings


@dataclass
class SignEvent:
    label: str
    confidence: float
    t_ms: int


class SignStateMachine:
    def __init__(
        self,
        min_confidence: float = settings.MIN_CONFIDENCE,
        confirm_windows: int = settings.CONFIRM_WINDOWS,
        cooldown_ms: int = settings.COOLDOWN_MS,
        non_sign_labels: set[str] = settings.NON_SIGN_LABELS,
    ):
        self.min_confidence = min_confidence
        self.confirm_windows = confirm_windows
        self.cooldown_ms = cooldown_ms
        self.non_sign_labels = non_sign_labels
        self.state = "REPOSO"
        self._candidate: str | None = None
        self._streak = 0
        self._doubt_streak = 0
        self._last_emit_ms = -10**12

    def update(self, label: str, confidence: float, t_ms: int) -> SignEvent | None:
        if confidence < self.min_confidence:
            self._candidate, self._streak = None, 0
            self._doubt_streak += 1
            # Una racha de dudas equivale a que la persona cambió de postura: libera la seña.
            if self.state == "CANDIDATO" or self._doubt_streak >= self.confirm_windows:
                self.state = "REPOSO"
            return None
        self._doubt_streak = 0

        if label in self.non_sign_labels:
            self.state = "REPOSO"
            self._candidate, self._streak = None, 0
            return None

        if self.state == "BLOQUEADO":
            return None

        if label != self._candidate:
            self._candidate, self._streak = label, 0
        self._streak += 1
        self.state = "CANDIDATO"

        if self._streak >= self.confirm_windows and t_ms - self._last_emit_ms >= self.cooldown_ms:
            self.state = "BLOQUEADO"
            self._candidate, self._streak = None, 0
            self._last_emit_ms = t_ms
            return SignEvent(label, confidence, t_ms)
        return None
