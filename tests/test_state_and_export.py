from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

try:
    from openpyxl import load_workbook
except ImportError:
    load_workbook = None

from excel_exporter import Excel606Exporter, HEADERS
from state_manager import StateManager, UserState


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


class StateAndExportTests(unittest.TestCase):
    def test_state_and_batch_survive_a_new_repository_instance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "state.sqlite3"
            first = StateManager(database)
            first.save_extracted_data(42, {"ncf": "B0100000001"}, [])
            first.add_confirmed_invoice(42, valid_invoice())

            restarted = StateManager(database)
            self.assertEqual(restarted.get_session(42).state, UserState.WAITING_CATEGORY)
            self.assertEqual(restarted.batch_count(42), 1)
            self.assertTrue(restarted.has_invoice(42, valid_invoice()))

    @unittest.skipIf(load_workbook is None, "openpyxl not installed")
    def test_export_uses_all_26_dgii_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exporter = Excel606Exporter(Path(directory))
            output = exporter.export([valid_invoice()], chat_id=42, reporting_rnc="101010101")
            workbook = load_workbook(output, data_only=True)
            sheet = workbook["606"]
            self.assertEqual(tuple(cell.value for cell in sheet[1]), HEADERS)
            self.assertEqual(sheet.max_column, 26)
            self.assertEqual(sheet["B2"].value, "131000001")
            self.assertEqual(sheet["G2"].value, "202608")
            self.assertEqual(sheet["H2"].value, "01")
            self.assertEqual(float(sheet["M2"].value), 1000.0)
            self.assertEqual(float(sheet["R2"].value), 180.0)
