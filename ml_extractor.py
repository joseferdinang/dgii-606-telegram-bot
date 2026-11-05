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

from ml.feature_engineering import tokenize, build_features

logger = logging.getLogger(__name__)


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
                    results[predicted_class] = (token_data['text'], float(max_prob))
                    
        return results

    def merge_with_regex(self, regex_fields: dict, ml_fields: dict) -> dict:
        merged = dict(regex_fields)
        for field, (val, conf) in ml_fields.items():
            if field not in merged or not merged[field]:
                merged[field] = val
        return merged
