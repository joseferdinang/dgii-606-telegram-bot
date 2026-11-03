"""ML-based field extractor that complements regex extraction."""
import joblib
import logging
from pathlib import Path
import pandas as pd

from ml.feature_engineering import tokenize, build_features

logger = logging.getLogger(__name__)

class MLFieldExtractor:
    def __init__(self, model_path: Path):
        self.available = False
        if not model_path.exists():
            logger.warning(f"ML model not found at {model_path}")
            return
            
        try:
            self.model = joblib.load(model_path)
            self.available = True
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")

    def extract(self, ocr_text: str) -> dict[str, tuple[str, float]]:
        if not self.available:
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
