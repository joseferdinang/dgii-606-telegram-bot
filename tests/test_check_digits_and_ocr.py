"""Unit tests for the new reliability improvements and continuous ML learning."""
import unittest
from unittest.mock import patch

from fiscal_validators import (
    validate_rnc_check_digit,
    validate_cedula_check_digit,
    normalize_identifier,
)
from ocr_engine import InvoiceOCREngine
from data_flywheel import synthesize_raw_invoice_text


class TestImprovements(unittest.TestCase):

    def test_dgii_rnc_modulo_11(self):
        # RNCs válidos reales de empresas dominicanas
        self.assertTrue(validate_rnc_check_digit("101068744"))  # TotalEnergies
        self.assertTrue(validate_rnc_check_digit("101008172"))  # Shell / Isla Dominicana
        self.assertTrue(validate_rnc_check_digit("131000002"))  # Valid 9-digit RNC
        # RNCs con dígito alterado (fallan Módulo 11)
        self.assertFalse(validate_rnc_check_digit("101068740"))
        self.assertFalse(validate_rnc_check_digit("101068749"))

    def test_dgii_cedula_modulo_10(self):
        # Cédula 11 dígitos con dígito verificador correcto
        self.assertTrue(validate_cedula_check_digit("00100000009"))
        # Cédula con dígito alterado (esperaba 9, tiene 8)
        self.assertFalse(validate_cedula_check_digit("00100000008"))

    def test_normalize_identifier_check_digit(self):
        # Normaliza y valida dígito cuando se solicita
        rnc = normalize_identifier("101-06874-4", validate_check_digit=True)
        self.assertEqual(rnc, "101068744")
        with self.assertRaises(ValueError):
            normalize_identifier("101-06874-9", validate_check_digit=True)

    def test_ocr_avoids_buyer_rnc_when_provided(self):
        ocr_text = """
        EMPRESA COMPRADORA (CLIENTE):
        RNC: 101008172
        
        EMPRESA PROVEEDORA (EMISOR):
        TOTALENERGIES DOMINICANA
        RNC: 101-06874-4
        NCF: E310000000001
        """
        # Debe descartar el RNC del comprador 101008172 y extraer el del emisor 101068744
        res = InvoiceOCREngine._extract_rnc_cedula(ocr_text, buyer_rnc="101008172")
        self.assertIsNotNone(res)
        self.assertEqual(res.value, "101068744")

    def test_ocr_pdf_pages_extracts_qr_with_priority(self):
        engine = InvoiceOCREngine()
        # Mocking qr_reader to return DGII fields on page 2
        mock_qr_fields = {
            "rnc_cedula": "101068744",
            "ncf": "E310000099999",
            "fecha_comprobante": "20261103",
            "monto_total": "5500.00",
            "itbis_facturado": "838.98",
        }
        with patch("qr_reader.extract_qr_from_image_bytes") as mock_qr:
            mock_qr.side_effect = [(None, None), ("fake_raw", mock_qr_fields)]
            res = engine.analyze_pdf_pages([b"page_1_bytes", b"page_2_bytes"])
            self.assertEqual(res.fields.get("ncf"), "E310000099999")
            self.assertEqual(res.fields.get("rnc_cedula"), "101068744")
            self.assertEqual(res.fields.get("monto_total"), "5500.00")
            self.assertEqual(res.confidence.get("ncf"), 1.0)
            self.assertEqual(res.confidence.get("rnc_cedula"), 1.0)

    def test_data_flywheel_synthesize_text_on_empty_ocr(self):
        data = {
            "rnc_cedula": "101068744",
            "ncf": "E310000084920",
            "fecha_comprobante": "20260924",
            "monto_total": "3646.28",
            "itbis_facturado": "0.00",
        }
        text = synthesize_raw_invoice_text(data)
        self.assertIn("101068744", text)
        self.assertIn("E310000084920", text)
        self.assertIn("3646.28", text)


if __name__ == "__main__":
    unittest.main()
