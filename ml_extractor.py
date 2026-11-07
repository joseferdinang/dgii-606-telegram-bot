"""ML-based field extractor that complements regex extraction and supports continuous active learning."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

try:
    import joblib
except ImportError:
    joblib = None

try:
    import pandas as pd
except ImportError:
    pd = None

from ml.feature_engineering import tokenize, build_features, parse_numeric
import re

logger = logging.getLogger(__name__)


def normalize_predicted_token(field_name: str, raw_text: str) -> str:
    """Normaliza un token extraído por ML al formato estándar del Formato 606."""
    t = raw_text.strip()
    if field_name == "rnc_cedula":
        digits = re.sub(r"\D", "", t)
        return digits if len(digits) in (9, 11) else t
    if field_name == "ncf":
        clean = re.sub(r"[^A-Za-z0-9]", "", t).upper()
        if clean and clean[0] in ("B", "E") and len(clean) >= 11:
            tail = (
                clean[1:]
                .replace("O", "0")
                .replace("I", "1")
                .replace("L", "1")
                .replace("S", "5")
            )
            return clean[0] + tail
        return clean or t
    if field_name == "fecha_comprobante":
        cleaned = re.sub(r"(?<=\d)[iIl](?=\d)", "1", t)
        cleaned = re.sub(r"(?<=[/-])[iIl](?=\d)", "1", cleaned)
        cleaned = re.sub(r"(?<=\d)[oO](?=\d)", "0", cleaned)
        m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})", cleaned)
        if m:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1 <= mo <= 12 and 1 <= d <= 31:
                return f"{y:04d}{mo:02d}{d:02d}"
        m2 = re.match(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})", cleaned)
        if m2:
            y, mo, d = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
            if 1 <= mo <= 12 and 1 <= d <= 31:
                return f"{y:04d}{mo:02d}{d:02d}"
        digits = re.sub(r"\D", "", cleaned)
        if len(digits) == 8:
            if 1990 <= int(digits[:4]) <= 2100:
                return digits
            if 1990 <= int(digits[4:]) <= 2100:
                return f"{digits[4:]}{digits[2:4]}{digits[:2]}"
        return t
    if field_name in ("monto_total", "itbis_facturado"):
        val = parse_numeric(t)
        return f"{val:.2f}"
    return t


class MLFieldExtractor:
    def __init__(self, model_path: Path):
        self.available = False
        self.model_path = model_path
        self.model = None
        self.reload(model_path)

    def reload(self, model_path: Optional[Path] = None) -> bool:
        """Recarga el modelo entrenado en memoria sin necesidad de reiniciar el bot."""
        if joblib is None:
            logger.debug("joblib no disponible para MLFieldExtractor")
            self.available = False
            return False

        path = model_path or self.model_path
        if not path or not path.exists():
            logger.warning("ML model not found at %s", path)
            self.available = False
            return False
            
        try:
            self.model = joblib.load(path)
            self.model_path = path
            self.available = True
            logger.info("Modelo ML cargado/recargado exitosamente desde %s", path)
            return True
        except Exception as e:
            logger.error("Failed to load/reload ML model: %s", e)
            self.available = False
            return False

    def extract(self, ocr_text: str) -> dict[str, tuple[str, float]]:
        if not self.available or not self.model or pd is None:
            return {}
            
        tokens = tokenize(ocr_text)
        if not tokens:
            return {}
            
        features = build_features(tokens)
        if features.empty:
            return {}
            
        probabilities = self.model.predict_proba(features)
        classes = [str(c) for c in self.model.classes_]
        
        results = {}
        for i, token_data in enumerate(tokens):
            probs = probabilities[i]
            max_prob = max(probs)
            max_idx = probs.argmax()
            predicted_class = classes[max_idx]
            
            if predicted_class != 'irrelevant' and max_prob > 0.5:
                if predicted_class not in results or max_prob > results[predicted_class][1]:
                    norm_val = normalize_predicted_token(predicted_class, token_data['text'])
                    results[predicted_class] = (norm_val, float(max_prob))
                    
        return results

    def merge_with_regex(self, regex_fields: dict, ml_fields: dict) -> dict:
        merged = dict(regex_fields)
        for field, (val, conf) in ml_fields.items():
            if field not in merged or not merged[field]:
                merged[field] = val
        return merged
