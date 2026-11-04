"""Unit tests for Data Flywheel and Continuous Active Learning."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_flywheel import (
    get_sample_count,
    load_training_corpus,
    record_training_sample,
    retrain_model_from_corpus,
    synthesize_raw_invoice_text,
)


class TestDataFlywheel(unittest.TestCase):
    def test_record_training_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_corpus = Path(tmpdir) / "confirmed_samples.jsonl"
            with patch("data_flywheel.SAMPLES_FILE", test_corpus), \
                 patch("data_flywheel.CORPUS_DIR", Path(tmpdir)):
                
                invoice_data = {
                    "rnc_cedula": "101068744",
                    "ncf": "E310000084920",
                    "fecha_comprobante": "20260924",
                    "monto_total": "3646.28",
                    "itbis_facturado": "0.00",
                    "tipo_bienes_servicios": "04",
                    "forma_pago": "03",
                    "monto_bienes": "3646.28",
                    "monto_servicios": "0.00",
                }
                
                record_training_sample(
                    invoice_data,
                    chat_id=12345,
                    was_corrected=True,
                    raw_text="TOTALENERGIES CHURCHILL GASOIL OPTIMO TOTAL 3646.28",
                )
                
                self.assertTrue(test_corpus.exists())
                with test_corpus.open("r", encoding="utf-8") as f:
                    lines = [json.loads(line) for line in f if line.strip()]
                    
                self.assertEqual(len(lines), 1)
                sample = lines[0]
                self.assertEqual(sample["chat_id"], 12345)
                self.assertTrue(sample["was_corrected"])
                self.assertEqual(sample["rnc_cedula"], "101068744")
                self.assertEqual(sample["ncf"], "E310000084920")
                self.assertEqual(sample["monto_total"], "3646.28")
                self.assertEqual(sample["tipo_bienes_servicios"], "04")
                self.assertIn("GASOIL OPTIMO", sample["raw_text"])

    def test_synthesizes_text_when_empty_raw_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_corpus = Path(tmpdir) / "confirmed_samples.jsonl"
            with patch("data_flywheel.SAMPLES_FILE", test_corpus), \
                 patch("data_flywheel.CORPUS_DIR", Path(tmpdir)):
                
                invoice_data = {
                    "rnc_cedula": "101068744",
                    "ncf": "E310000084920",
                    "fecha_comprobante": "20260924",
                    "monto_total": "1180.00",
                    "itbis_facturado": "180.00",
                    "monto_bienes": "1000.00",
                }
                record_training_sample(invoice_data, chat_id=999, was_corrected=False, raw_text="")
                with test_corpus.open("r", encoding="utf-8") as f:
                    sample = json.loads(f.readline())
                self.assertIn("101068744", sample["raw_text"])
                self.assertIn("E310000084920", sample["raw_text"])
                self.assertIn("1180.00", sample["raw_text"])

    def test_sample_count_and_retrain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_corpus = Path(tmpdir) / "confirmed_samples.jsonl"
            test_model = Path(tmpdir) / "test_model.joblib"
            with patch("data_flywheel.SAMPLES_FILE", test_corpus), \
                 patch("data_flywheel.CORPUS_DIR", Path(tmpdir)):
                
                self.assertEqual(get_sample_count(), 0)
                invoice_data = {
                    "rnc_cedula": "101068744",
                    "ncf": "E310000084920",
                    "fecha_comprobante": "20260924",
                    "monto_total": "3646.28",
                    "itbis_facturado": "0.00",
                }
                record_training_sample(
                    invoice_data,
                    chat_id=1,
                    was_corrected=False,
                    raw_text="TOTAL RNC 101068744 NCF E310000084920 TOTAL 3646.28",
                )
                self.assertEqual(get_sample_count(), 1)
                self.assertEqual(len(load_training_corpus()), 1)
                
                # Test retraining with this real sample
                res = retrain_model_from_corpus(model_path=test_model, min_samples=1)
                self.assertIn(res["status"], ["success", "dependency_missing"])
                if res["status"] == "success":
                    self.assertEqual(res["real_samples"], 1)
                    self.assertTrue(test_model.exists())


if __name__ == "__main__":
    unittest.main()
