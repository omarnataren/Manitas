"""Estimación de letras estáticas por reglas (dedos estirados + orientación de la mano).

Es una aproximación para probar sin modelo entrenado; no distingue letras que
comparten dedos y orientación (p. ej. A/E/S o U/R). Para eso está vision_teacher.py.
"""

from typing import Optional

import numpy as np

WRIST, THUMB_IP, THUMB_TIP, INDEX_MCP, MIDDLE_MCP = 0, 3, 4, 5, 9
FINGERS = {  # nombre: (PIP, TIP)
    "indice": (6, 8),
    "medio": (10, 12),
    "anular": (14, 16),
    "menique": (18, 20),
}


def finger_states(pts_px: np.ndarray) -> dict[str, bool]:
    """pts_px: (21, 2) en pixeles. Devuelve qué dedos están estirados."""
    dist = lambda a, b: float(np.linalg.norm(pts_px[a] - pts_px[b]))  # noqa: E731
    palm = dist(WRIST, MIDDLE_MCP) or 1.0

    states = {"pulgar": dist(THUMB_TIP, INDEX_MCP) > 0.7 * palm and dist(THUMB_TIP, WRIST) > dist(THUMB_IP, WRIST)}
    for name, (pip, tip) in FINGERS.items():
        states[name] = dist(tip, WRIST) > dist(pip, WRIST) * 1.15
    return states


def hand_orientation(pts_px: np.ndarray) -> tuple[str, float]:
    """Hacia dónde apunta la mano (muñeca -> nudillo del medio). Ángulo en grados, 90 = arriba."""
    dx, dy = pts_px[MIDDLE_MCP] - pts_px[WRIST]
    angle = float(np.degrees(np.arctan2(-dy, dx)))
    if 45 <= angle <= 135:
        return "arriba", angle
    if -135 <= angle <= -45:
        return "abajo", angle
    return "de lado", angle


# (dedos estirados, orientación) -> letra.
# Las de lado (G, H) siguen la forma común del alfabeto; validar con alguien que sepa LSM.
RULES = {
    ((), "arriba"): "A",
    (("pulgar",), "arriba"): "A",
    (("indice", "medio", "anular", "menique"), "arriba"): "B",
    (("indice",), "arriba"): "D",
    (("menique",), "arriba"): "I",
    (("pulgar", "indice"), "arriba"): "L",
    (("pulgar", "menique"), "arriba"): "Y",
    (("indice", "medio", "anular"), "arriba"): "W",
    (("pulgar", "indice"), "de lado"): "G",
    (("indice",), "de lado"): "G",
    (("indice", "medio"), "de lado"): "H",
}


def guess_letter(pts_px: np.ndarray) -> Optional[str]:
    s = finger_states(pts_px)
    up = tuple(name for name in ["pulgar", "indice", "medio", "anular", "menique"] if s[name])
    orientation, _ = hand_orientation(pts_px)

    if up == ("indice", "medio") and orientation == "arriba":
        palm = float(np.linalg.norm(pts_px[WRIST] - pts_px[MIDDLE_MCP])) or 1.0
        spread = float(np.linalg.norm(pts_px[8] - pts_px[12]))
        return "V" if spread > 0.35 * palm else "U"
    return RULES.get((up, orientation))
