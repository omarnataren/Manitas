import numpy as np

NUM_LANDMARKS = 21
LANDMARK_DIM = 3
WRIST = 0
MIDDLE_MCP = 9
TIPS = [4, 8, 12, 16, 20]
# Articulaciones del pulgar y los otros dedos que el pulgar toca o cubre en A/E/S/T/M/N.
THUMB_TARGETS = [5, 6, 9, 10, 13, 14, 17]
# Cadenas muñeca -> punta de cada dedo; se mide el ángulo en cada articulación intermedia.
FINGER_CHAINS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]

CSV_HEADER = ["t", "is_left"] + [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in "xyz"]


def _normalize(landmarks: np.ndarray, is_left: bool) -> np.ndarray:
    pts = landmarks.astype(np.float64).copy()
    if is_left:
        pts[:, 0] = -pts[:, 0]
    pts -= pts[WRIST]
    scale = np.linalg.norm(pts[MIDDLE_MCP])
    if scale > 0:
        pts /= scale
    return pts


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Ángulo en b (radianes) entre b->a y b->c. pi = recto, menor = doblado."""
    v1, v2 = a - b, c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)))


def landmarks_to_vector(landmarks: np.ndarray, is_left: bool = False) -> np.ndarray:
    """(21, 3) en coordenadas de imagen -> vector de features.

    - 63 coordenadas normalizadas (posición y tamaño de la mano no importan).
      La rotación NO se normaliza: en LSM la orientación distingue letras.
    - 10 distancias entre puntas de dedos (juntos/separados: U vs V).
    - 7 distancias de la punta del pulgar a nudillos y falanges (A/E/S/T/M/N).
    - 15 ángulos de flexión, 3 por dedo.
    - 1 indicador de índice y medio cruzados (R): negativo si las puntas
      quedan en orden inverso al de los nudillos.
    """
    pts = _normalize(landmarks, is_left)

    tip_dists = [np.linalg.norm(pts[a] - pts[b]) for i, a in enumerate(TIPS) for b in TIPS[i + 1:]]
    thumb_dists = [np.linalg.norm(pts[4] - pts[t]) for t in THUMB_TARGETS]
    angles = [_angle(pts[ch[j - 1]], pts[ch[j]], pts[ch[j + 1]]) for ch in FINGER_CHAINS for j in (1, 2, 3)]

    knuckles = pts[5, :2] - pts[9, :2]
    tips = pts[8, :2] - pts[12, :2]
    crossed = float(np.dot(tips, knuckles) / (np.dot(knuckles, knuckles) or 1.0))

    return np.concatenate([pts.flatten(), tip_dists, thumb_dists, angles, [crossed]])


NUM_VISION_FEATURES = NUM_LANDMARKS * LANDMARK_DIM + 10 + len(THUMB_TARGETS) + 3 * len(FINGER_CHAINS) + 1
