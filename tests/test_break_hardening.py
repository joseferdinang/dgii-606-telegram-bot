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

    def test_itbis_rates_18_16_10_and_multirate(self):
        from fiscal_validators import validate_invoice
        from handlers.confirmation_handler import _effective_itbis_label
        eng = InvoiceOCREngine()

        # 18% ITBIS
        t18 = "RNC: 101068744\nNCF: E310000000101\nFECHA: 15/10/2026\nSUBTOTAL: 10,000.00\nITBIS (18%): 1,800.00\nTOTAL A PAGAR: 11,800.00"
        f18, _ = eng._parse_fields(t18)
        self.assertEqual(f18.get("itbis_facturado"), "1800.00")
        self.assertEqual(f18.get("monto_total"), "11800.00")
        inv18 = {**f18, "monto_bienes": "10000.00", "monto_servicios": "0.00", "tipo_bienes_servicios": "02", "forma_pago": "01"}
        self.assertEqual(validate_invoice(inv18), [])
        self.assertEqual(_effective_itbis_label(inv18), " (18%)")

        # 16% ITBIS (tasa reducida)
        t16 = "RNC: 101068744\nNCF: E310000000102\nFECHA: 15/10/2026\nSUBTOTAL: 25,000.00\nITBIS 16%: 4,000.00\nTOTAL A PAGAR: 29,000.00"
        f16, _ = eng._parse_fields(t16)
        self.assertEqual(f16.get("itbis_facturado"), "4000.00")
        self.assertEqual(f16.get("monto_total"), "29000.00")
        inv16 = {**f16, "monto_bienes": "25000.00", "monto_servicios": "0.00", "tipo_bienes_servicios": "09", "forma_pago": "02"}
        self.assertEqual(validate_invoice(inv16), [])
        self.assertEqual(_effective_itbis_label(inv16), " (16% - Tasa reducida)")

        # 10% ITBIS
        t10 = "RNC: 101068744\nNCF: E310000000103\nFECHA: 15/10/2026\nSUBTOTAL: 15,000.00\nITBIS 10%: 1,500.00\nTOTAL A PAGAR: 16,500.00"
        f10, _ = eng._parse_fields(t10)
        self.assertEqual(f10.get("itbis_facturado"), "1500.00")
        self.assertEqual(f10.get("monto_total"), "16500.00")
        inv10 = {**f10, "monto_bienes": "15000.00", "monto_servicios": "0.00", "tipo_bienes_servicios": "09", "forma_pago": "03"}
        self.assertEqual(validate_invoice(inv10), [])
        self.assertEqual(_effective_itbis_label(inv10), " (10%)")

        # Multi-tasa (18% + 16% + 10% en la misma factura)
        t_multi = (
            "RNC: 101068744\nNCF: E310000000104\nFECHA: 15/10/2026\n"
            "SUBTOTAL: 50,000.00\nITBIS 18%: 3,600.00\nITBIS 16%: 3,200.00\nITBIS 10%: 1,000.00\nTOTAL: 57,800.00"
        )
        f_multi, _ = eng._parse_fields(t_multi)
        self.assertEqual(f_multi.get("itbis_facturado"), "7800.00")
        self.assertEqual(f_multi.get("monto_total"), "57800.00")

    def test_large_amounts_and_many_goods_txt_excel(self):
        import tempfile, openpyxl
        from excel_exporter import Excel606Exporter
        eng = InvoiceOCREngine()

        # Factura con 45 bienes/artículos y montos grandes
        item_lines = "\n".join(
            f"{i:02d}  ARTICULO INDUSTRIAL LOTE #{i:03d}   CANT: 10   PRECIO: 10,000.00   ITBIS: 18,000.00   VALOR TOTAL: 118,000.00"
            for i in range(1, 46)
        )
        text_many = (
            "PROVEEDOR INDUSTRIAL DOMINICANO SRL\n"
            "RNC: 101068744\n"
            "e-NCF: E310000000201\n"
            "FECHA: 20/10/2026\n"
            "CANT | DESCRIPCION | PRECIO | ITBIS | VALOR TOTAL\n"
            f"{item_lines}\n"
            "SUBTOTAL: RD$ 4,500,000.00\n"
            "ITBIS 18%: RD$ 810,000.00\n"
            "TOTAL A PAGAR: RD$ 5,310,000.00\n"
        )
        fields_many, _ = eng._parse_fields(text_many)
        self.assertEqual(fields_many.get("rnc_cedula"), "101068744")
        self.assertEqual(fields_many.get("ncf"), "E310000000201")
        self.assertEqual(fields_many.get("itbis_facturado"), "810000.00")
        self.assertEqual(fields_many.get("monto_total"), "5310000.00")

        # Exportar lote con 18%, 16%, 10%, multi-tasa, 45 bienes y monto multimillonario (147.5M)
        batch = [
            {
                "rnc_cedula": "101068744", "tipo_id": "1", "ncf": "E310000000201", "fecha_comprobante": "20261020",
                "monto_servicios": "0.00", "monto_bienes": "4500000.00", "monto_total": "5310000.00",
                "itbis_facturado": "810000.00", "tipo_bienes_servicios": "09", "forma_pago": "02",
            },
            {
                "rnc_cedula": "101008172", "tipo_id": "1", "ncf": "E310000000202", "fecha_comprobante": "20261021",
                "monto_servicios": "0.00", "monto_bienes": "125000000.00", "monto_total": "147500000.00",
                "itbis_facturado": "22500000.00", "tipo_bienes_servicios": "09", "forma_pago": "02",
            },
            {
                "rnc_cedula": "131000002", "tipo_id": "1", "ncf": "B0100000101", "fecha_comprobante": "20261022",
                "monto_servicios": "0.00", "monto_bienes": "25000.00", "monto_total": "29000.00",
                "itbis_facturado": "4000.00", "tipo_bienes_servicios": "02", "forma_pago": "03",
            },
            {
                "rnc_cedula": "131240674", "tipo_id": "1", "ncf": "B0100000102", "fecha_comprobante": "20261023",
                "monto_servicios": "0.00", "monto_bienes": "15000.00", "monto_total": "16500.00",
                "itbis_facturado": "1500.00", "tipo_bienes_servicios": "02", "forma_pago": "01",
            },
        ]
        out_dir = tempfile.mkdtemp()
        txt_path = generate_606_txt(batch, reporting_rnc="03100308018", output_dir=out_dir)
        raw_bytes = txt_path.read_bytes()
        self.assertTrue(raw_bytes.endswith(b"\r\n"))
        lines = raw_bytes.decode("cp1252").split("\r\n")[:-1]
        self.assertEqual(lines[0], "606|03100308018|202610|4")
        for line in lines[1:]:
            self.assertEqual(line.count("|"), 22)
        self.assertIn("|125000000.00|125000000.00|22500000.00|", lines[2])

        xlsx_path = Excel606Exporter(out_dir).export(batch, chat_id=1, reporting_rnc="03100308018")
        wb = openpyxl.load_workbook(xlsx_path)
        ws = wb.worksheets[0]
        self.assertEqual(ws.max_column, 26)
        self.assertEqual(ws.max_row, 5)  # 1 header + 4 rows


if __name__ == "__main__":
    unittest.main()
