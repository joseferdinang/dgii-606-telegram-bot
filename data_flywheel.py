"""
Data Flywheel for Active Learning & Continuous Model Improvement.

Persists real verified and user-corrected Dominican invoices into an active learning corpus
at data/training_corpus/confirmed_samples.jsonl and provides automated re-training capabilities.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CORPUS_DIR = Path("data/training_corpus")
SAMPLES_FILE = CORPUS_DIR / "confirmed_samples.jsonl"
_RETRAIN_LOCK = threading.Lock()
_LAST_RETRAIN_TIME = 0.0


def synthesize_raw_invoice_text(data: Dict[str, Any]) -> str:
    """Genera una representación de texto verosímil para facturas confirmadas si no hubo OCR crudo."""
    rnc = data.get("rnc_cedula", "")
    ncf = data.get("ncf", "")
    fecha = data.get("fecha_comprobante", "")
    if fecha and len(fecha) == 8:
        fecha_fmt = f"{fecha[6:8]}/{fecha[4:6]}/{fecha[:4]}"
    else:
        fecha_fmt = fecha
    total = data.get("monto_total", "0.00")
    itbis = data.get("itbis_facturado", "0.00")
    bienes = data.get("monto_bienes", "0.00")
    servicios = data.get("monto_servicios", "0.00")

    return (
        f"COMPROBANTE FISCAL\n"
        f"RNC: {rnc}\n"
        f"NCF: {ncf}\n"
        f"FECHA EMISION: {fecha_fmt}\n"
        f"SUBTOTAL BIENES: {bienes}\n"
        f"SUBTOTAL SERVICIOS: {servicios}\n"
        f"ITBIS FACTURADO: {itbis}\n"
        f"TOTAL A PAGAR: {total}"
    )


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
        
        effective_raw_text = raw_text or data.get("_raw_text") or ""
        if not effective_raw_text.strip():
            effective_raw_text = synthesize_raw_invoice_text(data)

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
            "raw_text": effective_raw_text,
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


def retrain_model_from_corpus(
    model_path: Optional[Path] = None,
    min_samples: int = 1,
    n_synthetic: int = 150,
    force: bool = False,
) -> dict:
    """Re-entrena el clasificador local integrando las muestras reales del corpus con sincronización atómica y debounce."""
    global _LAST_RETRAIN_TIME

    if not _RETRAIN_LOCK.acquire(blocking=False):
        logger.info("Re-entrenamiento ML omitido: ya hay un proceso de entrenamiento en ejecución.")
        return {"status": "already_running"}

    try:
        now = time.time()
        if not force and min_samples == 1 and (now - _LAST_RETRAIN_TIME) < 60:
            return {"status": "debounced", "message": "Re-entrenamiento aplazado por debounce"}

        try:
            import joblib
            import pandas as pd
            from sklearn.ensemble import HistGradientBoostingClassifier
            from ml.feature_engineering import tokenize, build_features, parse_numeric
            from ml.synthetic_generator import generate_dataset
        except ImportError as e:
            logger.warning("ML libraries not installed: %s", e)
            return {"status": "dependency_missing", "message": str(e)}

        samples = load_training_corpus()
        if len(samples) < min_samples:
            return {"status": "insufficient_samples", "count": len(samples)}

        target_path = model_path or (Path(__file__).parent / "ml" / "models" / "field_classifier.joblib")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Base sintética balanceada y rápida
        texts, labels_list = generate_dataset(n=n_synthetic)

        # 2. Incorporar muestras reales anotadas desde el corpus con peso reforzado
        real_count = 0
        for s in samples:
            text = s.get("raw_text") or ""
            if not text.strip():
                text = synthesize_raw_invoice_text(s)
            tokens = tokenize(text)
            if not tokens:
                continue

            rnc = str(s.get("rnc_cedula") or "").replace("-", "").strip()
            ncf = str(s.get("ncf") or "").upper().replace(" ", "").strip()
            
            try:
                total_num = float(str(s.get("monto_total") or "0").replace(",", ""))
            except ValueError:
                total_num = None

            try:
                itbis_num = float(str(s.get("itbis_facturado") or "0").replace(",", ""))
            except ValueError:
                itbis_num = None

            fecha = str(s.get("fecha_comprobante") or "").strip()

            labels = []
            for t in tokens:
                t_clean = t["text"].replace("-", "").replace(".", "").replace(",", "")
                t_upper = t["text"].upper().replace(" ", "")
                val_num = parse_numeric(t["text"])

                if rnc and (rnc == t_clean or (len(t_clean) >= 9 and rnc in t_clean)):
                    labels.append({"label": "rnc_cedula"})
                elif ncf and (ncf == t_upper or (len(t_upper) >= 11 and ncf in t_upper)):
                    labels.append({"label": "ncf"})
                elif total_num is not None and abs(val_num - total_num) < 0.01 and total_num > 0:
                    labels.append({"label": "monto_total"})
                elif itbis_num is not None and abs(val_num - itbis_num) < 0.01:
                    labels.append({"label": "itbis_facturado"})
                elif fecha and (fecha == t_clean or (len(t_clean) == 8 and t_clean == fecha)):
                    labels.append({"label": "fecha_comprobante"})
                else:
                    labels.append({"label": "irrelevant"})

            # Ponderación reforzada
            weight = 15 if s.get("was_corrected") else 8
            for _ in range(weight):
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
            max_iter=60,
            learning_rate=0.1,
            max_leaf_nodes=31,
            random_state=42,
        )
        clf.fit(X, y)

        # Guardado atómico con reemplazo para evitar archivos corruptos en lecturas concurrentes
        tmp_path = target_path.with_suffix(".tmp")
        joblib.dump(clf, tmp_path)
        os.replace(tmp_path, target_path)
        _LAST_RETRAIN_TIME = time.time()

        logger.info("Modelo ML re-entrenado exitosamente con %d muestras reales (%d tokens)", real_count, len(y))
        return {"status": "success", "real_samples": real_count, "tokens": len(y)}
    finally:
        _RETRAIN_LOCK.release()
