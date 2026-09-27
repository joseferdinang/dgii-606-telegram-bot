from __future__ import annotations

import unittest

from fiscal_validators import normalize_date, normalize_identifier, normalize_ncf, validate_invoice


def valid_invoice() -> dict:
    return {
        "rnc_cedula": "131000001",
        "ncf": "B0100000001",
        "fecha_comprobante": "20260801",
        "monto_total": "1180.00",
        "itbis_facturado": "180.00",
        "monto_servicios": "0.00",
        "monto_bienes": "1000.00",
        "tipo_bienes_servicios": "02",
        "forma_pago": "01",
    }


class FiscalValidatorTests(unittest.TestCase):
    def test_identifier_and_ncf_structures(self) -> None:
        self.assertEqual(normalize_identifier("131-00000-1"), "131000001")
        self.assertEqual(normalize_identifier("001-1234567-8"), "00112345678")
        self.assertEqual(normalize_ncf("b01-00000001"), "B0100000001")
        self.assertEqual(normalize_ncf("e31 0000000001"), "E310000000001")
        with self.assertRaises(ValueError):
            normalize_ncf("B9900000001")

    def test_real_calendar_date_is_required(self) -> None:
        self.assertEqual(normalize_date("01/08/2026"), "20260801")
        with self.assertRaises(ValueError):
            normalize_date("31/02/2026")

    def test_valid_invoice_has_no_blocking_issues(self) -> None:
        self.assertEqual(validate_invoice(valid_invoice()), [])

    def test_itbis_cannot_exceed_eighteen_percent(self) -> None:
        invoice = valid_invoice()
        invoice["itbis_facturado"] = "181.00"
        invoice["monto_total"] = "1181.00"
        issues = validate_invoice(invoice)
        self.assertTrue(any(issue.field == "itbis_facturado" for issue in issues))

    def test_credit_note_requires_modified_ncf(self) -> None:
        invoice = valid_invoice()
        invoice["ncf"] = "B0400000001"
        issues = validate_invoice(invoice)
        self.assertTrue(any(issue.field == "ncf_modificado" for issue in issues))
