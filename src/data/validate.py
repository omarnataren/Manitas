from dataclasses import dataclass

import numpy as np

from config import settings
from src.data import io

FLEX_COLUMNS = ["pulgar", "indice", "medio", "anular", "menique"]
FLEX_RANGE = (0.9, 3.1)  # escala 1-3 con un poco de tolerancia
RATE_TOLERANCE = 0.3  # ±30% respecto a SAMPLE_RATE_HZ
GAP_FACTOR = 3  # un salto > 3 periodos entre lecturas = paquetes perdidos


@dataclass
class Issue:
    level: str  # "ERROR" invalida la muestra, "WARNING" solo avisa
    message: str


def check_sample(sample: io.Sample) -> list[Issue]:
    issues: list[Issue] = []
    n = len(sample.frames)

    if sample.label not in settings.SIGN_LABELS:
        issues.append(Issue("ERROR", f"etiqueta '{sample.label}' no está en SIGN_LABELS"))
    if n < settings.MIN_SAMPLE_FRAMES:
        issues.append(Issue("ERROR", f"solo {n} lecturas (mínimo {settings.MIN_SAMPLE_FRAMES})"))
        return issues
    if not np.isfinite(sample.frames).all() or not np.isfinite(sample.t_ms).all():
        issues.append(Issue("ERROR", "hay valores NaN o infinitos"))
    dt = np.diff(sample.t_ms)
    if (dt < 0).any():
        issues.append(Issue("ERROR", "timestamps desordenados"))

    meta = sample.metadata
    if meta and (meta.get("label") != sample.label or meta.get("participant_id") != sample.participant):
        issues.append(Issue("WARNING", "metadata.json no coincide con la carpeta (persona/seña)"))

    duration_s = (sample.t_ms[-1] - sample.t_ms[0]) / 1000
    if duration_s > 0:
        rate = (n - 1) / duration_s
        if abs(rate - settings.SAMPLE_RATE_HZ) > RATE_TOLERANCE * settings.SAMPLE_RATE_HZ:
            issues.append(Issue("WARNING", f"frecuencia {rate:.0f} Hz (se esperaban ~{settings.SAMPLE_RATE_HZ})"))
        period_ms = 1000 / settings.SAMPLE_RATE_HZ
        gaps = int((dt > GAP_FACTOR * period_ms).sum())
        if gaps:
            issues.append(Issue("WARNING", f"{gaps} saltos de tiempo (posible pérdida de paquetes)"))

    flex_idx = [settings.FEATURE_NAMES.index(c) for c in FLEX_COLUMNS if c in settings.FEATURE_NAMES]
    if flex_idx:
        flex = sample.frames[:, flex_idx]
        out = (flex < FLEX_RANGE[0]) | (flex > FLEX_RANGE[1])
        if out.any():
            issues.append(Issue("WARNING", f"{int(out.sum())} lecturas de dedos fuera del rango 1-3"))

    if (sample.frames.std(axis=0) == 0).all():
        issues.append(Issue("WARNING", "todos los sensores constantes (¿guante desconectado?)"))
    return issues


def validate_dataset(verbose: bool = True) -> tuple[list[io.Sample], int]:
    """Carga y revisa todas las muestras. Devuelve (muestras válidas, número de inválidas)."""
    valid, invalid = [], 0
    for path in io.find_sample_dirs():
        rel = path.relative_to(settings.RAW_DIR)
        try:
            sample = io.load_sample(path)
        except ValueError as e:
            invalid += 1
            if verbose:
                print(f"[ERROR]   {rel}: {e}")
            continue

        issues = check_sample(sample)
        errors = [i for i in issues if i.level == "ERROR"]
        if verbose:
            if not issues:
                print(f"[OK]      {rel}: {len(sample.frames)} lecturas")
            for issue in issues:
                print(f"[{issue.level}]{' ' * (8 - len(issue.level))}{rel}: {issue.message}")
        if errors:
            invalid += 1
        else:
            valid.append(sample)
    return valid, invalid
