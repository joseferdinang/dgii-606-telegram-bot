"""Unit tests for MLFieldExtractor on Dominican invoice formats."""
import unittest
from pathlib import Path

from ml_extractor import MLFieldExtractor
from ml.synthetic_generator import (
    generate_combustible_invoice,
    generate_hardware_invoice,
    generate_tech_invoice,
)


class TestMLExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = Path(__file__).parent.parent / "ml" / "models" / "field_classifier.joblib"
        cls.extractor = MLFieldExtractor(model_path)

    def test_model_is_available(self):
        self.assertTrue(self.extractor.available, "ML model should be available and loaded")

    def test_fuel_invoice_extraction(self):
        """Test extraction on a real Dominican combustible receipt (Shell/Total/Propagas)."""
        text, _ = generate_combustible_invoice(noise_level=0.0)
        extracted = self.extractor.extract(text)
        
        self.assertIn("rnc_cedula", extracted)
        self.assertIn("ncf", extracted)
        self.assertIn("fecha_comprobante", extracted)
        self.assertIn("monto_total", extracted)
        
        rnc_val, rnc_conf = extracted["rnc_cedula"]
        self.assertGreater(rnc_conf, 0.5)
        self.assertTrue(len(rnc_val.replace("-", "")) in (9, 11))

    def test_hardware_invoice_extraction(self):
        """Test extraction on a real Dominican hardware receipt (Ochoa/Bellón/Americana)."""
        text, _ = generate_hardware_invoice(noise_level=0.0)
        extracted = self.extractor.extract(text)
        
        self.assertIn("rnc_cedula", extracted)
        self.assertIn("ncf", extracted)
        self.assertIn("monto_total", extracted)
        self.assertIn("itbis_facturado", extracted)

    def test_tech_invoice_extraction(self):
        """Test extraction on a real Dominican computer & tech receipt (Cecomsa/Omega Tech)."""
        text, _ = generate_tech_invoice(noise_level=0.0)
        extracted = self.extractor.extract(text)
        
        self.assertIn("rnc_cedula", extracted)
        self.assertIn("ncf", extracted)
        self.assertIn("monto_total", extracted)
        self.assertIn("itbis_facturado", extracted)

    def test_merge_with_regex_preserves_regex_and_fills_gaps(self):
        regex_fields = {
            "rnc_cedula": "101068744",
            "ncf": "B0100000001",
            "fecha_comprobante": "",  # missing
            "monto_total": None,      # missing
        }
        ml_fields = {
            "rnc_cedula": ("101999999", 0.95),  # should NOT overwrite regex
            "fecha_comprobante": ("20260924", 0.98),
            "monto_total": ("3646.28", 0.92),
            "itbis_facturado": ("0.00", 0.90),
        }
        merged = self.extractor.merge_with_regex(regex_fields, ml_fields)
        
        # Regex field preserved
        self.assertEqual(merged["rnc_cedula"], "101068744")
        self.assertEqual(merged["ncf"], "B0100000001")
        # ML filled the gaps
        self.assertEqual(merged["fecha_comprobante"], "20260924")
        self.assertEqual(merged["monto_total"], "3646.28")
        self.assertEqual(merged["itbis_facturado"], "0.00")


if __name__ == "__main__":
    unittest.main()
