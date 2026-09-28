"""Decide cuándo una letra de la cámara cuenta como confirmada.

    ESPERANDO ──(mano quieta, misma letra)──> SOSTENIENDO ──(STATIC_HOLD_MS)──> confirma letra estática
        │
        └──(el detector de movimiento vota letras)──> DECIDIENDO ──(la mano se detiene SETTLE_MS)──>
                                                       confirma la letra con más votos

- Letras con movimiento: no se decide con las primeras predicciones (a mitad del gesto el
  detector puede ver otra letra), sino al terminar el gesto, por mayoría de votos, y solo si
  la mano se movió. Si una letra sale DYNAMIC_FAST_CONFIRM veces seguidas, se confirma sin esperar.
- Con la mano en movimiento, o mientras se decide una letra con movimiento, no se confirman
  letras estáticas (la J empieza como I).
- Tras una letra estática: la misma no se repite hasta mover la mano (RELEASE_MOVING_MS)
  o sacarla de cuadro (RELEASE_ABSENT_MS); otra letra estática distinta sí puede confirmarse.
- Tras una letra con movimiento: queda bloqueado hasta que el detector deja de verla (su
  ventana de 2 s la sigue conteniendo) o la mano sale de cuadro. Así no se repite y su forma
  final (la Z termina como D) no se confirma como estática.
"""

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

STATIC_HOLD_MS = 700  # sostener la letra estática este tiempo
STATIC_MIN_CONFIDENCE = 0.6
STATIC_AGREEMENT = 0.8  # fracción de cuadros en STATIC_HOLD_MS que deben ser la misma letra
DYNAMIC_MIN_CONFIDENCE = 0.7  # una predicción de movimiento solo vota con al menos esta confianza
DYNAMIC_MIN_VOTES = 2  # votos mínimos de la letra ganadora (~150 ms entre predicciones)
DYNAMIC_FAST_CONFIRM = 4  # tantas predicciones seguidas iguales confirman sin esperar a que pare
DYNAMIC_VOTE_MS = 2500  # los votos más viejos que esto se descartan
DYNAMIC_RELEASE_CONFIDENCE = 0.5  # el detector debe ver otra cosa con al menos esta confianza
SETTLE_MS = 200  # la mano quieta este tiempo = el gesto terminó
MOTION_WINDOW_MS = 250
MOTION_THRESHOLD = 0.8  # tamaños de palma por segundo; por encima la mano "se mueve"
RELEASE_MOVING_MS = 150
RELEASE_ABSENT_MS = 400
COOLDOWN_MS = 500

TRACKED_POINTS = [0, 4, 8, 12, 16, 20]  # muñeca y puntas de los dedos


@dataclass
class LetterEvent:
    letter: str
    confidence: float
    dynamic: bool


class LetterStateMachine:
    def __init__(self, non_letter_labels: set[str]):
        self.non_letter_labels = non_letter_labels
        self.state = "ESPERANDO"
        self.progress = 0.0  # 0-1, cuánto falta para confirmar la letra estática
        self.candidate: str | None = None
        self.motion = 0.0
        self._static: deque = deque()  # (t_ms, letra o None)
        self._points: deque = deque()  # (t_ms, puntos (6, 2), palma)
        self._votes: deque = deque()  # (t_ms, letra, confianza) del detector de movimiento
        self._static_lock: str | None = None  # letra estática que no se debe repetir
        self._dyn_block: str | None = None  # letra con movimiento que el detector todavía ve
        self._moving_since: int | None = None
        self._still_since: int | None = None
        self._last_hand_ms = -10**9
        self._last_emit_ms = -10**9
        self._dyn_label: str | None = None
        self._dyn_streak = 0
        self._gesture_moved = False  # la mano se movió desde que empezaron los votos actuales

    def _update_motion(self, t_ms: int, pts: np.ndarray) -> float:
        palm = max(float(np.linalg.norm(pts[9, :2] - pts[0, :2])), 1e-6)
        self._points.append((t_ms, pts[TRACKED_POINTS, :2], palm))
        while self._points and t_ms - self._points[0][0] > MOTION_WINDOW_MS:
            self._points.popleft()
        t0, p0, _ = self._points[0]
        if t_ms - t0 < MOTION_WINDOW_MS / 2:
            return 0.0
        displacement = np.linalg.norm(pts[TRACKED_POINTS, :2] - p0, axis=1).mean() / palm
        return displacement / ((t_ms - t0) / 1000)

    def _emit(self, t_ms: int, letter: str, confidence: float, dynamic: bool) -> LetterEvent:
        self.state, self.progress = "BLOQUEADO", 0.0
        self._last_emit_ms = t_ms
        if dynamic:
            self._dyn_block, self._static_lock = letter, None
        else:
            self._static_lock = letter
        self._static.clear()
        self._votes.clear()
        self._dyn_label, self._dyn_streak = None, 0
        return LetterEvent(letter, confidence, dynamic)

    def _vote_winner(self) -> tuple[str, int, float] | None:
        counts = Counter(label for _, label, _ in self._votes if label != self._dyn_block)
        if not counts:
            return None
        label, n = counts.most_common(1)[0]
        conf = float(np.mean([c for _, lab, c in self._votes if lab == label]))
        return label, n, conf

    def update(self, t_ms: int, pts: np.ndarray | None, static_pred, dynamic_pred) -> LetterEvent | None:
        """pts: puntos de la mano o None. static_pred: (letra, confianza) del cuadro o None.
        dynamic_pred: (letra, confianza) si el modelo de movimiento predijo en este cuadro, o None."""
        if pts is None:
            self._points.clear()
            self._static.clear()
            self.motion, self.progress, self.candidate = 0.0, 0.0, None
            if t_ms - self._last_hand_ms > RELEASE_ABSENT_MS:
                self._static_lock, self._dyn_block = None, None
                self._votes.clear()
                self.state = "ESPERANDO"
            return None
        self._last_hand_ms = t_ms

        self.motion = self._update_motion(t_ms, pts)
        moving = self.motion > MOTION_THRESHOLD
        if moving:
            self._still_since = None
            self._moving_since = self._moving_since if self._moving_since is not None else t_ms
            if t_ms - self._moving_since >= RELEASE_MOVING_MS:
                self._static_lock = None
        else:
            self._moving_since = None
            self._still_since = self._still_since if self._still_since is not None else t_ms

        cooldown_ok = t_ms - self._last_emit_ms >= COOLDOWN_MS

        # Letras con movimiento: se acumulan votos durante el gesto.
        if dynamic_pred is not None:
            label, conf = dynamic_pred
            if self._dyn_block and label != self._dyn_block and conf >= DYNAMIC_RELEASE_CONFIDENCE:
                self._dyn_block = None
            if label not in self.non_letter_labels and conf >= DYNAMIC_MIN_CONFIDENCE:
                self._votes.append((t_ms, label, conf))
                self._dyn_streak = self._dyn_streak + 1 if label == self._dyn_label else 1
                self._dyn_label = label
                fast = self._dyn_streak >= DYNAMIC_FAST_CONFIRM and (self._gesture_moved or moving)
                if fast and label != self._dyn_block and cooldown_ok:
                    return self._emit(t_ms, label, conf, dynamic=True)
            else:
                self._dyn_label, self._dyn_streak = None, 0
        while self._votes and t_ms - self._votes[0][0] > DYNAMIC_VOTE_MS:
            self._votes.popleft()

        if not self._votes:
            self._gesture_moved = False
        elif moving:
            self._gesture_moved = True
        winner = self._vote_winner()
        if winner:
            settled = self._still_since is not None and t_ms - self._still_since >= SETTLE_MS
            if settled:
                label, n, conf = winner
                gesture_moved = self._gesture_moved
                self._votes.clear()
                # Votos con la mano siempre quieta (p. ej. la forma inicial antes de empezar) no cuentan.
                if gesture_moved and n >= DYNAMIC_MIN_VOTES and cooldown_ok:
                    return self._emit(t_ms, label, conf, dynamic=True)
            else:
                self.state, self.progress, self.candidate = "DECIDIENDO", 0.0, winner[0]
                self._static.clear()
                return None

        if self._dyn_block:
            self.state, self.progress, self.candidate = "BLOQUEADO", 0.0, None
            return None
        if moving:
            self._static.clear()
            self.state, self.progress, self.candidate = "MOVIMIENTO", 0.0, None
            return None

        letter = static_pred[0] if static_pred and static_pred[1] >= STATIC_MIN_CONFIDENCE else None
        self._static.append((t_ms, letter))
        while self._static and t_ms - self._static[0][0] > STATIC_HOLD_MS:
            self._static.popleft()
        counts: dict[str, int] = {}
        for _, s in self._static:
            if s:
                counts[s] = counts.get(s, 0) + 1
        if not counts:
            self.state = "BLOQUEADO" if self._static_lock else "ESPERANDO"
            self.progress, self.candidate = 0.0, None
            return None

        self.candidate = max(counts, key=counts.get)
        if self.candidate == self._static_lock:
            self.state, self.progress = "BLOQUEADO", 0.0
            return None

        agreement = counts[self.candidate] / len(self._static)
        span = t_ms - self._static[0][0]
        self.state = "SOSTENIENDO"
        self.progress = min(1.0, span / STATIC_HOLD_MS) if agreement >= STATIC_AGREEMENT else 0.0
        if span >= STATIC_HOLD_MS * 0.95 and agreement >= STATIC_AGREEMENT and cooldown_ok:
            conf = static_pred[1] if static_pred and static_pred[0] == self.candidate else STATIC_MIN_CONFIDENCE
            return self._emit(t_ms, self.candidate, conf, dynamic=False)
        return None
