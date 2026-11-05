from __future__ import annotations

from datetime import date, timedelta
import unittest

from dgii_txt import Record606
from fiscal_validators import (
    calc_rnc_check_digit,
    is_valid_cedula_checksum,
    is_valid_identifier_checksum,
    is_valid_rnc_checksum,
    normalize_date,
    normalize_identifier,
    normalize_ncf,
    validate_invoice,
)
from ocr_engine import InvoiceOCREngine


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

    def test_rnc_and_cedula_checksum_algorithms(self) -> None:
        # Módulo 11 (RNC oficial de 9 dígitos)
        check_digit = calc_rnc_check_digit("10185004")
        self.assertEqual(check_digit, "3")
        self.assertTrue(is_valid_rnc_checksum("101850043"))
        self.assertFalse(is_valid_rnc_checksum("101850044"))
        self.assertTrue(is_valid_identifier_checksum("101-85004-3"))

        # Módulo 10 / Luhn (Cédula oficial de 11 dígitos)
        self.assertTrue(is_valid_cedula_checksum("00113918205"))
        self.assertFalse(is_valid_cedula_checksum("00113918206"))
        self.assertTrue(is_valid_identifier_checksum("001-1391820-5"))

        # Strict checksum mode in normalize_identifier and validate_invoice
        self.assertEqual(normalize_identifier("101-85004-3", strict_checksum=True), "101850043")
        with self.assertRaises(ValueError):
            normalize_identifier("101-85004-4", strict_checksum=True)

        inv = valid_invoice()
        inv["rnc_cedula"] = "101850044"  # Dígito verificador inválido
        self.assertEqual(validate_invoice(inv, strict_checksum=False), [])
        strict_issues = validate_invoice(inv, strict_checksum=True)
        self.assertTrue(any(issue.field == "rnc_cedula" for issue in strict_issues))

    def test_non_deductible_606_ncfs_are_blocked(self) -> None:
        # B02 / E32 (Consumo) no son deducibles en el 606
        for bad_ncf in ("B0200000001", "E320000000001", "B1500000001", "B1600000001", "E450000000001"):
            with self.subTest(ncf=bad_ncf):
                with self.assertRaises(ValueError):
                    normalize_ncf(bad_ncf)

    def test_real_calendar_date_and_no_future_dates(self) -> None:
        self.assertEqual(normalize_date("01/08/2026"), "20260801")
        with self.assertRaises(ValueError):
            normalize_date("31/02/2026")

        tomorrow = (date.today() + timedelta(days=1)).strftime("%Y%m%d")
        with self.assertRaises(ValueError):
            normalize_date(tomorrow)

    def test_valid_invoice_has_no_blocking_issues(self) -> None:
        self.assertEqual(validate_invoice(valid_invoice()), [])

    def test_itbis_rounding_tolerance_and_eighteen_percent_cap(self) -> None:
        # Tolerancia de redondeo multicentavo (hasta RD$ 0.05 sobre el 18%)
        invoice = valid_invoice()
        invoice["itbis_facturado"] = "180.04"
        invoice["monto_total"] = "1180.04"
        self.assertEqual(validate_invoice(invoice), [])

        # Exceso real (> 18% + 0.05) se bloquea
        invoice["itbis_facturado"] = "181.00"
        invoice["monto_total"] = "1181.00"
        issues = validate_invoice(invoice)
        self.assertTrue(any(issue.field == "itbis_facturado" for issue in issues))

    def test_legal_tip_ten_percent_cap(self) -> None:
        invoice = valid_invoice()
        invoice["monto_propina_legal"] = "100.00"  # Exactamente 10% de 1000.00
        invoice["monto_total"] = "1280.00"
        self.assertEqual(validate_invoice(invoice), [])

        invoice["monto_propina_legal"] = "150.00"  # 15% > 10% permitido
        invoice["monto_total"] = "1330.00"
        issues = validate_invoice(invoice)
        self.assertTrue(any(issue.field == "monto_propina_legal" for issue in issues))

    def test_itbis_cost_plus_proportionality_cannot_exceed_billed_itbis(self) -> None:
        invoice = valid_invoice()
        invoice["itbis_llevado_costo"] = "100.00"
        invoice["itbis_proporcionalidad"] = "90.00"  # 100 + 90 = 190 > 180
        issues = validate_invoice(invoice)
        self.assertTrue(any(issue.field == "itbis_llevado_costo" for issue in issues))

    def test_ncf_specific_rules_b11_b13_b14_b17(self) -> None:
        # B13 / E43 (Gastos Menores): no permite ITBIS
        b13 = valid_invoice()
        b13["ncf"] = "B1300000001"
        issues = validate_invoice(b13)
        self.assertTrue(any(issue.field == "itbis_facturado" for issue in issues))
        b13["itbis_facturado"] = "0.00"
        b13["monto_total"] = "1000.00"
        self.assertEqual(validate_invoice(b13), [])

        # B14 / E44 (Regímenes Especiales): exento de ITBIS
        b14 = valid_invoice()
        b14["ncf"] = "E440000000001"
        issues = validate_invoice(b14)
        self.assertTrue(any(issue.field == "itbis_facturado" for issue in issues))

        # B17 / E47 (Pagos al Exterior): no permite ITBIS y reemplaza RNC por el de la empresa en TXT
        e47 = valid_invoice()
        e47["ncf"] = "E470000000001"
        e47["itbis_facturado"] = "0.00"
        e47["monto_total"] = "1000.00"
        self.assertEqual(validate_invoice(e47), [])
        rec_e47 = Record606.from_dict(e47, reporting_rnc="101850043")
        self.assertEqual(rec_e47.rnc_cedula, "101850043")

        # B11 / E41 (Compras Informales): exige 100% de retención de ITBIS si hay ITBIS facturado
        b11 = valid_invoice()
        b11["ncf"] = "B1100000001"
        issues = validate_invoice(b11)
        self.assertTrue(any(issue.field == "itbis_retenido" for issue in issues))
        b11["itbis_retenido"] = "180.00"
        b11["fecha_pago"] = "20260801"
        self.assertEqual(validate_invoice(b11), [])

    def test_isr_retention_rules_for_legal_entities_vs_individuals(self) -> None:
        # Persona Jurídica (RNC 9 dígitos) no admite retención ISR por Honorarios (02)
        invoice = valid_invoice()
        invoice["rnc_cedula"] = "131000001"  # 9 dígitos -> tipo_id 1
        invoice["tipo_retencion_isr"] = "02"
        invoice["isr_retenido"] = "100.00"
        invoice["fecha_pago"] = "20260801"
        issues = validate_invoice(invoice)
        self.assertTrue(any(issue.field == "tipo_retencion_isr" for issue in issues))

        # Persona Física (Cédula 11 dígitos) sí admite retención ISR por Honorarios (02)
        invoice["rnc_cedula"] = "00113918205"  # 11 dígitos -> tipo_id 2
        self.assertEqual(validate_invoice(invoice), [])

    def test_credit_note_requires_modified_ncf(self) -> None:
        invoice = valid_invoice()
        invoice["ncf"] = "B0400000001"
        issues = validate_invoice(invoice)
        self.assertTrue(any(issue.field == "ncf_modificado" for issue in issues))

    def test_ocr_extracts_propina_isc_and_ignores_buyer_rnc(self) -> None:
        sample_text = (
            "RESTAURANTE EL CONUCO SRL\n"
            "RNC Cliente: 131-00000-1\n"
            "RNC: 101-85004-3\n"
            "NCF: B0100000045\n"
            "FECHA: 15/08/2026\n"
            "SUBTOTAL: RD$ 1,000.00\n"
            "ITBIS 18%: RD$ 180.00\n"
            "PROPINA LEGAL 10%: RD$ 100.00\n"
            "ISC: RD$ 20.00\n"
            "TOTAL A PAGAR: RD$ 1,300.00\n"
        )
        engine = InvoiceOCREngine.__new__(InvoiceOCREngine)
        fields, confs = engine._parse_fields(sample_text)
        self.assertEqual(fields.get("rnc_cedula"), "101850043")
        self.assertEqual(fields.get("rnc_comprador"), "131000001")
        self.assertEqual(fields.get("monto_propina_legal"), "100.00")
        self.assertEqual(fields.get("impuesto_selectivo_consumo"), "20.00")
        self.assertEqual(fields.get("monto_total"), "1300.00")
        self.assertEqual(fields.get("itbis_facturado"), "180.00")

    def test_allocation_inference_and_preview_tax_breakdown(self) -> None:
        from handlers.confirmation_handler import (
            _effective_itbis_label,
            _format_preview_text,
            _infer_allocation_mode,
        )

        # Inferencia por categoría (09 -> goods, 03 -> services)
        self.assertEqual(_infer_allocation_mode({"tipo_bienes_servicios": "09"}), "goods")
        self.assertEqual(_infer_allocation_mode({"tipo_bienes_servicios": "03"}), "services")
        # Inferencia por memoria del proveedor
        self.assertEqual(
            _infer_allocation_mode({"tipo_bienes_servicios": "02"}, {"tipo_asignacion": "services"}),
            "services",
        )

        # Tasa efectiva de ITBIS (18%, 16% Ley 253-12, Exento)
        inv18 = valid_invoice()
        self.assertEqual(_effective_itbis_label(inv18), " (18%)")
        inv16 = dict(inv18, itbis_facturado="160.00", monto_total="1160.00")
        self.assertEqual(_effective_itbis_label(inv16), " (16% - Tasa reducida)")

        # Vista previa incluye alerta de RNC comprador y propina legal
        inv_prev = dict(
            inv18,
            monto_propina_legal="100.00",
            monto_total="1280.00",
            _buyer_rnc_mismatch="131999999 (Empresa activa: 101850043)",
        )
        preview = _format_preview_text(inv_prev)
        self.assertIn("Alerta RNC Comprador", preview)
        self.assertIn("Propina legal (10%): RD$ 100.00", preview)
        self.assertIn("ITBIS (18%): RD$ 180.00", preview)

    def test_usd_conversion_converts_all_fields_without_rounding_drift(self) -> None:
        from decimal import Decimal
        from handlers.exchange_rate_handler import apply_usd_exchange_rate

        usd_invoice = {
            "rnc_cedula": "101850043",
            "ncf": "B0100000001",
            "fecha_comprobante": "20260801",
            "monto_total": "128.00",
            "itbis_facturado": "18.00",
            "monto_propina_legal": "10.00",
            "monto_bienes": "100.00",
            "monto_servicios": "0.00",
            "tipo_bienes_servicios": "09",
            "forma_pago": "03",
        }
        apply_usd_exchange_rate(usd_invoice, Decimal("61.35"))
        self.assertEqual(usd_invoice["monto_propina_legal"], "613.50")
        self.assertEqual(validate_invoice(usd_invoice), [])

    def test_excel_exporter_replaces_rnc_on_self_issued_ncfs(self) -> None:
        from pathlib import Path
        import tempfile
        from excel_exporter import Excel606Exporter

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = Excel606Exporter(Path(tmpdir))
            inv_b13 = valid_invoice()
            inv_b13["ncf"] = "B1300000001"
            inv_b13["itbis_facturado"] = "0.00"
            inv_b13["monto_total"] = "1000.00"
            row = exporter._row(1, inv_b13, reporting_rnc="101850043")
            self.assertEqual(row[1], "101850043")
