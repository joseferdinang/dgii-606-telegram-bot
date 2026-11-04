import unittest
import os
import tempfile
from decimal import Decimal
try:
    import openpyxl
except ImportError:
    openpyxl = None
from pathlib import Path

from fiscal_validators import validate_invoice, ValidationIssue, normalize_amount
from state_manager import StateManager, DuplicateInvoiceError, UserState
from excel_exporter import Excel606Exporter

def valid_invoice() -> dict:
    return {
        "rnc_cedula": "131000001",
        "tipo_id": "1",
        "ncf": "B0100000001",
        "fecha_comprobante": "20260801",
        "monto_total": "1180.00",
        "itbis_facturado": "180.00",
        "monto_bienes": "1000.00",
        "monto_servicios": "0.00",
        "tipo_bienes_servicios": "02",
        "forma_pago": "01",
    }

class TestErrorScenarios(unittest.TestCase):

    def test_valid_invoice_no_issues(self):
        self.assertEqual(validate_invoice(valid_invoice()), [])

    def test_rnc_wrong_length_8_digits(self):
        inv = valid_invoice()
        inv['rnc_cedula'] = "13100000"
        issues = validate_invoice(inv)
        self.assertTrue(any("RNC" in i.message or "cédula" in i.message or "longitud" in i.message for i in issues))
        
    def test_rnc_wrong_length_10_digits(self):
        inv = valid_invoice()
        inv['rnc_cedula'] = "1310000012"
        issues = validate_invoice(inv)
        self.assertTrue(any("RNC" in i.message or "cédula" in i.message for i in issues))

    def test_ncf_invalid_type_B99(self):
        inv = valid_invoice()
        inv['ncf'] = "B9900000001"
        issues = validate_invoice(inv)
        self.assertTrue(any("NCF" in i.message for i in issues))

    def test_ncf_too_short(self):
        inv = valid_invoice()
        inv['ncf'] = "B010000001"
        issues = validate_invoice(inv)
        self.assertTrue(any("NCF" in i.message for i in issues))

    def test_date_february_31(self):
        inv = valid_invoice()
        inv['fecha_comprobante'] = "20260231"
        issues = validate_invoice(inv)
        self.assertTrue(any("Fecha" in i.message or "fecha" in i.message for i in issues))

    def test_date_year_1999(self):
        inv = valid_invoice()
        inv['fecha_comprobante'] = "19991231"
        issues = validate_invoice(inv)
        self.assertTrue(any("año" in i.message or "Fecha" in i.message for i in issues))

    def test_date_far_future(self):
        inv = valid_invoice()
        inv['fecha_comprobante'] = "20990101"
        issues = validate_invoice(inv)
        self.assertTrue(any("año" in i.message or "Fecha" in i.message for i in issues))

    def test_itbis_exceeds_18_percent(self):
        inv = valid_invoice()
        inv['itbis_facturado'] = "181.02"
        issues = validate_invoice(inv)
        self.assertTrue(any("ITBIS" in i.message or "18%" in i.message or "itbis" in i.field.lower() for i in issues))

    def test_itbis_exactly_18_percent_passes(self):
        inv = valid_invoice()
        issues = validate_invoice(inv)
        self.assertEqual(issues, [])

    def test_total_arithmetic_mismatch(self):
        inv = valid_invoice()
        inv['monto_total'] = "1181.02"
        issues = validate_invoice(inv)
        self.assertTrue(any("monto_total" in i.field or "Total" in i.message or "monto" in i.field.lower() for i in issues))

    def test_credit_note_needs_modified_ncf(self):
        inv = valid_invoice()
        inv['ncf'] = "B0400000001"
        issues = validate_invoice(inv)
        self.assertTrue(any("NCF" in i.message or "modificado" in i.message for i in issues))

    def test_debit_note_needs_modified_ncf(self):
        inv = valid_invoice()
        inv['ncf'] = "B0300000001"
        issues = validate_invoice(inv)
        self.assertTrue(any("NCF" in i.message or "modificado" in i.message for i in issues))

    def test_ecf_credit_note_needs_modified(self):
        inv = valid_invoice()
        inv['ncf'] = "E340000000001"
        issues = validate_invoice(inv)
        self.assertTrue(any("NCF" in i.message or "modificado" in i.message for i in issues))

    def test_retention_requires_payment_date(self):
        inv = valid_invoice()
        inv['itbis_retenido'] = "50.00"
        issues = validate_invoice(inv)
        self.assertTrue(any("fecha" in i.message.lower() for i in issues))

    def test_payment_date_before_invoice_date(self):
        inv = valid_invoice()
        inv['fecha_pago'] = "20260701"
        issues = validate_invoice(inv)
        self.assertTrue(any("fecha_pago" in i.field for i in issues))

    def test_invalid_category_code(self):
        inv = valid_invoice()
        inv['tipo_bienes_servicios'] = "99"
        issues = validate_invoice(inv)
        self.assertTrue(any("tipo_bienes_servicios" in i.field for i in issues))

    def test_invalid_payment_method(self):
        inv = valid_invoice()
        inv['forma_pago'] = "99"
        issues = validate_invoice(inv)
        self.assertTrue(any("forma_pago" in i.field for i in issues))

    def test_itbis_retenido_exceeds_facturado(self):
        inv = valid_invoice()
        inv['itbis_retenido'] = "200.00"
        inv['fecha_pago'] = "20260801"
        issues = validate_invoice(inv)
        self.assertTrue(any("itbis_retenido" in i.field or "retenido" in i.message.lower() for i in issues))

    def test_itbis_proporcionalidad_exceeds(self):
        inv = valid_invoice()
        inv['itbis_proporcionalidad'] = "200.00"
        issues = validate_invoice(inv)
        self.assertTrue(any("itbis_proporcionalidad" in i.field or "proporcionalidad" in i.message.lower() for i in issues))

    def test_itbis_costo_exceeds_facturado(self):
        inv = valid_invoice()
        inv['itbis_llevado_costo'] = "200.00"
        issues = validate_invoice(inv)
        self.assertTrue(any("itbis_llevado_costo" in i.field or "costo" in i.message.lower() for i in issues))

    def test_isr_without_valid_retention_type(self):
        inv = valid_invoice()
        inv['isr_retenido'] = "50.00"
        inv['tipo_retencion_isr'] = "99"
        inv['fecha_pago'] = "20260801"
        issues = validate_invoice(inv)
        self.assertTrue(any("isr" in i.field.lower() or "isr" in i.message.lower() or "tipo_retencion" in i.field for i in issues))

    def test_amount_negative_rejected(self):
        with self.assertRaises(ValueError):
            normalize_amount("-100.00")

    def test_amount_exceeds_max(self):
        with self.assertRaises(ValueError):
            normalize_amount("1000000000000.00")

    def test_amount_empty_allowed(self):
        self.assertIsNone(normalize_amount("", allow_empty=True))

    def test_amount_empty_rejected(self):
        with self.assertRaises(ValueError):
            normalize_amount("")

    # === State Manager Errors ===
    def test_duplicate_invoice_raises_error(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name
        sm = StateManager(db_path)
        try:
            sm.add_confirmed_invoice(1, valid_invoice())
            with self.assertRaises(DuplicateInvoiceError):
                sm.add_confirmed_invoice(1, valid_invoice())
        finally:
            os.remove(db_path)

    def test_session_default_is_idle(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name
        sm = StateManager(db_path)
        try:
            self.assertEqual(sm.get_session(1).state, UserState.IDLE)
        finally:
            os.remove(db_path)

    def test_batch_empty_initially(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name
        sm = StateManager(db_path)
        try:
            self.assertEqual(len(sm.get_batch(1)), 0)
        finally:
            os.remove(db_path)

    def test_clear_session_resets_state(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name
        sm = StateManager(db_path)
        try:
            sm.set_state(1, UserState.PROCESSING_OCR)
            sm.clear_session(1)
            self.assertEqual(sm.get_session(1).state, UserState.IDLE)
        finally:
            os.remove(db_path)

    def test_clear_batch_removes_all(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name
        sm = StateManager(db_path)
        try:
            sm.add_confirmed_invoice(1, valid_invoice())
            sm.clear_batch_and_session(1)
            self.assertEqual(len(sm.get_batch(1)), 0)
            self.assertEqual(sm.get_session(1).state, UserState.IDLE)
        finally:
            os.remove(db_path)

    def test_supplier_memory_roundtrip(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name
        sm = StateManager(db_path)
        try:
            sm.add_confirmed_invoice(1, valid_invoice())
            sm.save_supplier_memory("131000001", "02", "01")
            mem = sm.get_supplier_memory("131000001")
            self.assertTrue(mem.get("tipo_bienes_servicios") == "02")
        finally:
            os.remove(db_path)

    def test_batch_period_from_first_invoice(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name
        sm = StateManager(db_path)
        try:
            sm.add_confirmed_invoice(1, valid_invoice())
            self.assertEqual(sm.batch_period(1), "202608")
        finally:
            os.remove(db_path)

    # === Excel Export Errors ===
    def test_export_empty_list_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = Excel606Exporter(Path(tmpdir))
            with self.assertRaises(ValueError):
                exporter.export([], chat_id=1, reporting_rnc="131000001")

    @unittest.skipIf(openpyxl is None, "openpyxl not installed")
    def test_export_produces_26_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = Excel606Exporter(Path(tmpdir))
            export_path = exporter.export([valid_invoice()], chat_id=1, reporting_rnc="131000001")
            wb = openpyxl.load_workbook(export_path)
            ws = wb.active
            self.assertGreaterEqual(ws.max_column, 23)

    @unittest.skipIf(openpyxl is None, "openpyxl not installed")
    def test_export_date_split_correctly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = Excel606Exporter(Path(tmpdir))
            export_path = exporter.export([valid_invoice()], chat_id=1, reporting_rnc="131000001")
            self.assertTrue(os.path.exists(export_path))

    @unittest.skipIf(openpyxl is None, "openpyxl not installed")
    def test_export_instructions_sheet_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = Excel606Exporter(Path(tmpdir))
            export_path = exporter.export([valid_invoice()], chat_id=1, reporting_rnc="131000001")
            wb = openpyxl.load_workbook(export_path)
            self.assertIn("Instrucciones", wb.sheetnames)

    @unittest.skipIf(openpyxl is None, "openpyxl not installed")
    def test_export_total_calculation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = Excel606Exporter(Path(tmpdir))
            export_path = exporter.export([valid_invoice()], chat_id=1, reporting_rnc="131000001")
            self.assertTrue(os.path.exists(export_path))

    @unittest.skipIf(openpyxl is None, "openpyxl not installed")
    def test_export_file_created_and_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = Excel606Exporter(Path(tmpdir))
            export_path = exporter.export([valid_invoice()], chat_id=1, reporting_rnc="131000001")
            self.assertTrue(os.path.exists(export_path))

if __name__ == '__main__':
    unittest.main()
