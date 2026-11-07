"""Unit tests verifying edge-case hardening, adversarial inputs and quality checks."""

import unittest
from decimal import Decimal
import io
from PIL import Image, ImageFilter
from pathlib import Path

from fiscal_validators import (
    normalize_date,
    normalize_amount,
    normalize_identifier,
    normalize_ncf,
)
from dgii_txt import Record606, generate_606_txt
from state_manager import StateManager, DuplicateInvoiceError
from ocr_engine import InvoiceOCREngine


class TestBreakHardening(unittest.TestCase):
    def test_normalize_date_iso_and_short(self):
        self.assertEqual(normalize_date("2026-10-05"), "20261005")
        self.assertEqual(normalize_date("2026/10/05"), "20261005")
        self.assertEqual(normalize_date("05-10-26"), "20261005")
        self.assertEqual(normalize_date("5/10/26"), "20261005")
        self.assertEqual(normalize_date("05/10/2026"), "20261005")

    def test_normalize_amount_adversarial(self):
        # NaN and Infinity must raise ValueError
        with self.assertRaises(ValueError):
            normalize_amount("NaN")
        with self.assertRaises(ValueError):
            normalize_amount("Infinity")
        with self.assertRaises(ValueError):
            normalize_amount("-10.00")
        # Fullwidth digits
        self.assertEqual(normalize_amount("１２３.４５"), Decimal("123.45"))
        # Valid numbers
        self.assertEqual(normalize_amount("1,234.56"), Decimal("1234.56"))
        self.assertEqual(normalize_amount("1.234,56"), Decimal("1234.56"))

    def test_normalize_identifier_fullwidth(self):
        # Fullwidth unicode digits for RNC 131878271
        self.assertEqual(normalize_identifier("１３１８７８２７１"), "131878271")

    def test_record606_missing_fields_friendly_error(self):
        with self.assertRaises(ValueError) as ctx:
            Record606.from_dict({"ncf": "B0100000001", "rnc_cedula": "131878271"})
        self.assertIn("fecha_comprobante", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            Record606.from_dict({"fecha_comprobante": "20261005", "rnc_cedula": "131878271"})
        self.assertIn("NCF", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            Record606.from_dict({"ncf": "B0100000001", "fecha_comprobante": "20261005"})
        self.assertIn("RNC o Cédula", str(ctx.exception))

    def test_state_manager_case_insensitive_duplicate(self):
        import tempfile
        d = tempfile.mkdtemp()
        sm = StateManager(Path(d) / "test.db")
        inv1 = {
            "rnc_cedula": "131878271",
            "ncf": "B0100000001",
            "fecha_comprobante": "20261005",
            "monto_total": "100.00",
        }
        sm.add_confirmed_invoice(1, inv1, rnc="131878271")
        self.assertTrue(sm.has_invoice(1, {"ncf": "b0100000001"}, rnc="131878271"))
        with self.assertRaises(DuplicateInvoiceError):
            sm.add_confirmed_invoice(1, {**inv1, "ncf": "b0100000001"}, rnc="131878271")

    def test_state_manager_mixed_period_rejection(self):
        import tempfile
        d = tempfile.mkdtemp()
        sm = StateManager(Path(d) / "test.db")
        inv_oct = {
            "rnc_cedula": "131878271",
            "ncf": "B0100000001",
            "fecha_comprobante": "20261005",
            "monto_total": "100.00",
        }
        inv_jan = {
            "rnc_cedula": "131878271",
            "ncf": "B0100000002",
            "fecha_comprobante": "20260115",
            "monto_total": "100.00",
        }
        sm.add_confirmed_invoice(1, inv_oct, rnc="131878271")
        with self.assertRaises(ValueError) as ctx:
            sm.add_confirmed_invoice(1, inv_jan, rnc="131878271")
        self.assertIn("período fiscal", str(ctx.exception))

    def test_ocr_amount_none_safe(self):
        self.assertIsNone(InvoiceOCREngine._normalize_amount(None))
        self.assertIsNone(InvoiceOCREngine._normalize_amount(""))
        self.assertIsNone(InvoiceOCREngine._normalize_amount("   "))

    def test_ocr_image_quality_check(self):
        # Empty or tiny
        is_ok, reason = InvoiceOCREngine.check_image_quality(b"")
        self.assertFalse(is_ok)

        # Tiny image 50x50
        buf_tiny = io.BytesIO()
        Image.new("RGB", (50, 50), "white").save(buf_tiny, "JPEG")
        is_ok, reason = InvoiceOCREngine.check_image_quality(buf_tiny.getvalue())
        self.assertFalse(is_ok)
        self.assertIn("reducida", reason)

        # Extremely blurred image
        buf_blur = io.BytesIO()
        im = Image.new("RGB", (400, 400), "white")
        im = im.filter(ImageFilter.GaussianBlur(10))
        im.save(buf_blur, "JPEG")
        is_ok, reason = InvoiceOCREngine.check_image_quality(buf_blur.getvalue())
        self.assertFalse(is_ok)
        self.assertIn("borrosa", reason)


if __name__ == "__main__":
    unittest.main()
