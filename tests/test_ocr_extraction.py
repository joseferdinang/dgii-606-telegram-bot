import unittest
from ocr_engine import InvoiceOCREngine

class TestOCRExtraction(unittest.TestCase):
    
    def test_extract_rnc_with_label(self):
        self.assertEqual(getattr(InvoiceOCREngine._extract_rnc_cedula("RNC: 131-00000-1"), 'value', None), "131000001")

    def test_extract_rnc_without_label(self):
        self.assertEqual(getattr(InvoiceOCREngine._extract_rnc_cedula("some text 131000001 here"), 'value', None), "131000001")

    def test_extract_cedula_11_digits(self):
        self.assertEqual(getattr(InvoiceOCREngine._extract_rnc_cedula("Cédula: 001-1234567-8"), 'value', None), "00112345678")

    def test_extract_rnc_ignores_phone(self):
        self.assertNotEqual(InvoiceOCREngine._extract_rnc_cedula("Tel: 809-555-1234"), "809555123")

    def test_extract_ncf_standard(self):
        self.assertEqual(getattr(InvoiceOCREngine._extract_ncf("NCF: B0100000001"), 'value', None), "B0100000001")

    def test_extract_ncf_lowercase(self):
        self.assertEqual(getattr(InvoiceOCREngine._extract_ncf("ncf: b0100000001"), 'value', None), "B0100000001")

    def test_extract_ecf(self):
        self.assertEqual(getattr(InvoiceOCREngine._extract_ncf("e-CF: E310000000001"), 'value', None), "E310000000001")

    def test_extract_ncf_with_spaces(self):
        self.assertEqual(getattr(InvoiceOCREngine._extract_ncf("B01 00000001"), 'value', None), "B0100000001")

    def test_extract_date_dd_mm_yyyy(self):
        self.assertEqual(getattr(InvoiceOCREngine._extract_fecha("Fecha: 15/03/2026"), 'value', None), "20260315")

    def test_extract_date_iso(self):
        self.assertEqual(getattr(InvoiceOCREngine._extract_fecha("2026-03-15"), 'value', None), "20260315")

    def test_extract_date_textual_spanish(self):
        self.assertEqual(getattr(InvoiceOCREngine._extract_fecha("15 de marzo de 2026"), 'value', None), "20260315")

    def test_extract_date_ignores_card_expiry(self):
        self.assertNotEqual(InvoiceOCREngine._extract_fecha("VENCE: 03/28"), "20280301")

    def test_extract_date_with_label(self):
        self.assertEqual(getattr(InvoiceOCREngine._extract_fecha("F. COMP.: 15/03/2026"), 'value', None), "20260315")

    def test_extract_monto_with_label(self):
        self.assertEqual(getattr(InvoiceOCREngine._extract_monto("TOTAL A PAGAR: RD$ 1,250.50"), 'value', None), "1250.50")

    def test_extract_monto_total_general(self):
        self.assertEqual(getattr(InvoiceOCREngine._extract_monto("TOTAL GENERAL: 5000.00"), 'value', None), "5000.00")

    def test_extract_monto_ignores_subtotal(self):
        text = "SUBTOTAL: 1000.00\nTOTAL: 1180.00"
        self.assertEqual(getattr(InvoiceOCREngine._extract_monto(text), 'value', None), "1180.00")

    def test_extract_monto_multiline(self):
        text = "TOTAL\n1250.50"
        self.assertEqual(getattr(InvoiceOCREngine._extract_monto(text), 'value', None), "1250.50")

    def test_extract_itbis_explicit(self):
        self.assertEqual(getattr(InvoiceOCREngine._extract_itbis("ITBIS FACTURADO: 270.00"), 'value', None), "270.00")

    def test_extract_itbis_with_percent(self):
        self.assertEqual(getattr(InvoiceOCREngine._extract_itbis("ITBIS (18%): 270.00"), 'value', None), "270.00")

    def test_extract_itbis_ignores_retained(self):
        self.assertNotEqual(InvoiceOCREngine._extract_itbis("ITBIS RETENIDO: 50.00"), "50.00")

    def test_extract_itbis_iva_synonym(self):
        self.assertEqual(getattr(InvoiceOCREngine._extract_itbis("IVA: 270.00"), 'value', None), "270.00")

    def test_currency_dop_explicit(self):
        self.assertEqual(getattr(InvoiceOCREngine._extract_currency("DOP"), 'value', None), "DOP")

    def test_currency_rd_dollar(self):
        self.assertEqual(getattr(InvoiceOCREngine._extract_currency("RD$"), 'value', None), "DOP")

    def test_currency_usd(self):
        self.assertEqual(getattr(InvoiceOCREngine._extract_currency("USD"), 'value', None), "USD")

    def test_currency_default_dop(self):
        self.assertEqual(getattr(InvoiceOCREngine._extract_currency("1000.00"), 'value', None), "DOP")

    def test_normalize_dominican_format(self):
        self.assertEqual(InvoiceOCREngine._normalize_amount("1,250.50"), "1250.50")

    def test_normalize_european_with_thousands(self):
        self.assertEqual(InvoiceOCREngine._normalize_amount("1.250,50"), "1250.50")

    def test_normalize_plain_number(self):
        self.assertEqual(InvoiceOCREngine._normalize_amount("1250.50"), "1250.50")

if __name__ == '__main__':
    unittest.main()
