from collections import defaultdict

from src.data.io import Sample

SPLITS = ("train", "val", "test")


def assign_splits(
    samples: list[Sample], val: list[str] | None = None, test: list[str] | None = None
) -> tuple[dict[str, str], bool]:
    """Devuelve ({sample_id: split}, por_persona).

    Por persona: todas las muestras de una persona van al mismo split, así test
    mide si el modelo funciona con alguien que no vio al entrenar. Con menos de
    3 personas no se puede, y se reparte por muestra dentro de cada seña.
    """
    participants = sorted({s.participant for s in samples})
    if val is None and test is None:
        if len(participants) >= 4:
            test, val = [participants[-1]], [participants[-2]]
        elif len(participants) == 3:
            test, val = [participants[-1]], []  # entrenar con 2 personas vale más que tener val

    if val is not None or test is not None:
        val, test = set(val or []), set(test or [])
        unknown = (val | test) - set(participants)
        if unknown:
            raise ValueError(f"Personas desconocidas: {sorted(unknown)}. Hay: {participants}")
        mapping = {
            s.sample_id: "test" if s.participant in test else "val" if s.participant in val else "train"
            for s in samples
        }
        return mapping, True

    by_label: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_label[s.label].append(s)
    mapping = {}
    for label_samples in by_label.values():
        label_samples.sort(key=lambda s: s.sample_id)
        n = len(label_samples)
        n_holdout = max(1, round(n * 0.2)) if n >= 3 else 0
        for i, s in enumerate(label_samples):
            if i < n_holdout:
                mapping[s.sample_id] = "test"
            elif i < 2 * n_holdout:
                mapping[s.sample_id] = "val"
            else:
                mapping[s.sample_id] = "train"
    return mapping, False
