import csv
import json
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from config import settings


def participant_of(sample_id: str) -> str:
    return sample_id.split("_", 1)[0]


def evaluate(model, X: np.ndarray, y: np.ndarray, ids: np.ndarray, labels: list[str]) -> dict:
    """Métricas por ventana, por muestra (voto de sus ventanas) y por persona."""
    t0 = time.perf_counter()
    proba = model.predict_proba(X)
    latency_ms = (time.perf_counter() - t0) / max(len(X), 1) * 1000
    pred = proba.argmax(axis=1)
    idx = list(range(len(labels)))

    votes: dict[str, Counter] = defaultdict(Counter)
    truth = {}
    for sid, p, t in zip(ids, pred, y):
        votes[sid][p] += 1
        truth[sid] = t
    sample_ok = {sid: votes[sid].most_common(1)[0][0] == truth[sid] for sid in votes}

    by_participant: dict[str, list[bool]] = defaultdict(list)
    for sid, ok in sample_ok.items():
        by_participant[participant_of(sid)].append(ok)

    return {
        "windows": int(len(y)),
        "samples": len(sample_ok),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=idx, average="macro", zero_division=0)),
        "sample_accuracy": float(np.mean(list(sample_ok.values()))) if sample_ok else 0.0,
        "per_participant_sample_accuracy": {p: float(np.mean(v)) for p, v in sorted(by_participant.items())},
        "latency_ms_per_window": round(latency_ms, 3),
        "report": classification_report(y, pred, labels=idx, target_names=labels, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=idx).tolist(),
    }


def print_metrics(name: str, m: dict) -> None:
    print(f"\n=== {name}: {m['windows']} ventanas de {m['samples']} muestras ===")
    print(m["report"])
    print(f"Accuracy por ventana: {m['accuracy']:.2%} | macro F1: {m['macro_f1']:.2%}")
    print(f"Accuracy por muestra (voto de sus ventanas): {m['sample_accuracy']:.2%}")
    for p, acc in m["per_participant_sample_accuracy"].items():
        print(f"  {p}: {acc:.2%}")


def save_experiment(model_name: str, labels: list[str], results: dict, config: dict, history: dict | None = None) -> Path:
    """Guarda métricas y configuración en experiments/<modelo>_<fecha>/."""
    out = settings.EXPERIMENTS_DIR / f"{model_name}_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    with open(out / "config.json", "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    summary = {}
    for split, m in results.items():
        summary[split] = {k: v for k, v in m.items() if k not in ("report", "confusion_matrix")}
        (out / f"classification_report_{split}.txt").write_text(m["report"])
        with open(out / f"confusion_matrix_{split}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["real \\ predicho", *labels])
            for label, row in zip(labels, m["confusion_matrix"]):
                writer.writerow([label, *row])
    with open(out / "metrics.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if history:
        with open(out / "history.json", "w") as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)
    return out
