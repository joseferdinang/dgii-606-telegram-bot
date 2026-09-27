"""
Data Flywheel for Active Learning & Continuous Model Improvement.

Persists real verified and user-corrected Dominican invoices into an active learning corpus
at data/training_corpus/confirmed_samples.jsonl.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

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
