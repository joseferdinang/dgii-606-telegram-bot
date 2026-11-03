"""
Data Flywheel for Active Learning & Continuous Model Improvement.

Persists real verified and user-corrected Dominican invoices into an active learning corpus
at data/training_corpus/confirmed_samples.jsonl and provides automated re-training capabilities.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CORPUS_DIR = Path("data/training_corpus")
SAMPLES_FILE = CORPUS_DIR / "confirmed_samples.jsonl"


def record_training_sample(
    data: Dict[str, Any],
    *,
    chat_id: int,
    was_corrected: bool = False,
    raw_text: Optional[str] = None,
) -> None:
    """
    Saves a confirmed invoice into the active learning JSONL corpus.
    """
    try:
        CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        
        sample = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chat_id": chat_id,
            "was_corrected": was_corrected,
            "rnc_cedula": data.get("rnc_cedula"),
            "ncf": data.get("ncf"),
            "fecha_comprobante": data.get("fecha_comprobante"),
            "monto_total": data.get("monto_total"),
            "itbis_facturado": data.get("itbis_facturado"),
            "tipo_bienes_servicios": data.get("tipo_bienes_servicios"),
            "forma_pago": data.get("forma_pago"),
            "monto_servicios": data.get("monto_servicios"),
            "monto_bienes": data.get("monto_bienes"),
            "raw_text": raw_text or data.get("_raw_text") or "",
        }
        
        with SAMPLES_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
        logger.info(
            "Training sample recorded for chat_id=%s (corrected=%s, rnc=%s, ncf=%s)",
            chat_id, was_corrected, data.get("rnc_cedula"), data.get("ncf")
        )
    except Exception:
        logger.exception("Failed to record training sample in data flywheel")


def get_sample_count() -> int:
    """Retorna la cantidad de muestras reales confirmadas en el corpus de aprendizaje."""
    if not SAMPLES_FILE.exists():
        return 0
    try:
        with SAMPLES_FILE.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def load_training_corpus() -> List[Dict[str, Any]]:
    """Carga todas las muestras registradas en el corpus de aprendizaje."""
    if not SAMPLES_FILE.exists():
        return []
    samples = []
    try:
        with SAMPLES_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    except Exception:
        logger.exception("Error al leer el corpus de entrenamiento")
    return samples


def retrain_model_from_corpus(model_path: Optional[Path] = None, min_samples: int = 1) -> dict:
    """Re-entrena el clasificador local integrando las muestras reales del corpus de aprendizaje."""
    import joblib
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from ml.feature_engineering import tokenize, build_features
    from ml.synthetic_generator import generate_dataset

    samples = load_training_corpus()
    if len(samples) < min_samples:
        return {"status": "insufficient_samples", "count": len(samples)}

    target_path = model_path or (Path(__file__).parent / "ml" / "models" / "field_classifier.joblib")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Base sintética balanceada
    texts, labels_list = generate_dataset(n=1000)

    # 2. Incorporar muestras reales anotadas desde el corpus con peso reforzado
    real_count = 0
    for s in samples:
        text = s.get("raw_text") or ""
        if not text.strip():
            continue
        tokens = tokenize(text)
        if not tokens:
            continue

        rnc = str(s.get("rnc_cedula") or "").replace("-", "").strip()
        ncf = str(s.get("ncf") or "").upper().replace(" ", "").strip()
        total = str(s.get("monto_total") or "").strip()
        itbis = str(s.get("itbis_facturado") or "").strip()
        fecha = str(s.get("fecha_comprobante") or "").strip()

        labels = []
        for t in tokens:
            t_clean = t["text"].replace("-", "").replace(".", "").replace(",", "")
            t_upper = t["text"].upper().replace(" ", "")
            if rnc and (rnc in t_clean or t_clean in rnc) and len(t_clean) >= 8:
                labels.append({"label": "rnc_cedula"})
            elif ncf and (ncf == t_upper or t_upper in ncf) and len(t_upper) >= 8:
                labels.append({"label": "ncf"})
            elif total and (total in t["text"] or t["text"] in total) and len(t["text"]) >= 2:
                labels.append({"label": "monto_total"})
            elif itbis and (itbis in t["text"] or t["text"] in itbis) and len(t["text"]) >= 2:
                labels.append({"label": "itbis_facturado"})
            elif fecha and (fecha[:4] in t["text"] or fecha in t_clean):
                labels.append({"label": "fecha_comprobante"})
            else:
                labels.append({"label": "irrelevant"})

        # Ponderación: repetir las muestras reales 5 veces para reforzar el aprendizaje
        for _ in range(5):
            texts.append(text)
            labels_list.append(labels)
        real_count += 1

    # Construir matrices de características
    X_list = []
    y_list = []
    for text, labels in zip(texts, labels_list):
        tokens = tokenize(text)
        if len(tokens) != len(labels):
            continue
        df = build_features(tokens)
        X_list.append(df)
        y_list.extend([l["label"] for l in labels])

    if not X_list:
        return {"status": "error", "message": "No fue posible extraer características"}

    X = pd.concat(X_list, ignore_index=True)
    y = y_list

    clf = HistGradientBoostingClassifier(
        max_iter=80,
        learning_rate=0.1,
        max_leaf_nodes=31,
        random_state=42,
    )
    clf.fit(X, y)
    joblib.dump(clf, target_path)

    logger.info("Modelo ML re-entrenado exitosamente con %d muestras reales (%d tokens)", real_count, len(y))
    return {"status": "success", "real_samples": real_count, "tokens": len(y)}
