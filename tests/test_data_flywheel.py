"""Unit tests for Data Flywheel."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_flywheel import record_training_sample


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


if __name__ == "__main__":
    unittest.main()
