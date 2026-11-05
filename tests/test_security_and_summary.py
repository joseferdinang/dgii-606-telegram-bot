"""Tests for new reliability, security, multi-company isolation, and summary features."""
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from state_manager import StateManager, UserState, _hash_pin, _verify_pin


class NewFeaturesAndSecurityTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_bot606.sqlite3"
        self.sm = StateManager(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_pbkdf2_pin_hashing_and_legacy_compatibility(self):
        salt = "testsalt123456"
        pin = "9876"
        pbkdf2_hash = _hash_pin(pin, salt)

        # Verify PBKDF2 matches
        self.assertTrue(_verify_pin(pin, salt, pbkdf2_hash))
        self.assertFalse(_verify_pin("wrong", salt, pbkdf2_hash))

        # Test legacy SHA-256 hash backward compatibility
        import hashlib
        legacy_sha256 = hashlib.sha256(f"{salt}:{pin}".encode("utf-8")).hexdigest()
        self.assertTrue(_verify_pin(pin, salt, legacy_sha256))
        self.assertFalse(_verify_pin("wrong", salt, legacy_sha256))

    def test_login_rate_limiting_brute_force_protection(self):
        chat_id = 99991
        rnc = "131000001"
        pin = "12345"

        self.sm.register_company(chat_id, rnc, "Empresa Seguridad SRL", pin)

        # Attempt 1 to 4 with incorrect pin should fail with attempt count
        for attempt in range(1, 5):
            with self.assertRaises(ValueError) as ctx:
                self.sm.login_company(chat_id, rnc, "0000")
            self.assertIn(f"Intento {attempt} de 5", str(ctx.exception))

        # 5th failed attempt should trigger lockout
        with self.assertRaises(ValueError) as ctx:
            self.sm.login_company(chat_id, rnc, "0000")
        self.assertIn("bloqueado temporalmente por 15 minutos", str(ctx.exception))

        # Further attempts (even with correct PIN) should be blocked during lockout
        with self.assertRaises(ValueError) as ctx:
            self.sm.login_company(chat_id, rnc, pin)
        self.assertIn("Acceso bloqueado temporalmente", str(ctx.exception))

    def test_per_company_isolated_supplier_memory(self):
        vendor_rnc = "101999999"
        company_a = "131000001"
        company_b = "131000002"

        # Company A categorizes as 02 (Servicios) and 01 (Efectivo)
        self.sm.save_supplier_memory(vendor_rnc, "02", "01", company_rnc=company_a)

        # Company B categorizes as 09 (Costo de venta) and 03 (Tarjeta)
        self.sm.save_supplier_memory(vendor_rnc, "09", "03", company_rnc=company_b)

        # Verify company A gets its own categorization
        mem_a = self.sm.get_supplier_memory(vendor_rnc, company_rnc=company_a)
        self.assertEqual(mem_a.get("tipo_bienes_servicios"), "02")
        self.assertEqual(mem_a.get("forma_pago"), "01")

        # Verify company B gets its own categorization
        mem_b = self.sm.get_supplier_memory(vendor_rnc, company_rnc=company_b)
        self.assertEqual(mem_b.get("tipo_bienes_servicios"), "09")
        self.assertEqual(mem_b.get("forma_pago"), "03")

    def test_reset_stale_processing_sessions(self):
        chat_id_1 = 101
        chat_id_2 = 102

        self.sm.set_state(chat_id_1, UserState.PROCESSING_OCR)
        self.sm.set_state(chat_id_2, UserState.WAITING_CONFIRMATION)

        reset_count = self.sm.reset_stale_processing_sessions()
        self.assertEqual(reset_count, 1)

        # Chat 1 should now be IDLE
        self.assertEqual(self.sm.get_session(chat_id_1).state, UserState.IDLE)
        # Chat 2 should remain WAITING_CONFIRMATION
        self.assertEqual(self.sm.get_session(chat_id_2).state, UserState.WAITING_CONFIRMATION)

    def test_batch_summary_and_delete_invoice(self):
        chat_id = 8888
        rnc = "131000001"

        inv1 = {
            "rnc_cedula": "101000001",
            "ncf": "B0100000001",
            "fecha_comprobante": "20261015",
            "monto_total": "1180.00",
            "itbis_facturado": "180.00",
            "monto_servicios": "1000.00",
            "monto_bienes": "0.00",
            "tipo_bienes_servicios": "02",
            "forma_pago": "01",
        }
        inv2 = {
            "rnc_cedula": "101000002",
            "ncf": "B0100000002",
            "fecha_comprobante": "20261020",
            "monto_total": "590.00",
            "itbis_facturado": "90.00",
            "monto_servicios": "0.00",
            "monto_bienes": "500.00",
            "tipo_bienes_servicios": "09",
            "forma_pago": "03",
        }

        self.sm.add_confirmed_invoice(chat_id, inv1, rnc=rnc)
        self.sm.add_confirmed_invoice(chat_id, inv2, rnc=rnc)

        summary = self.sm.get_batch_summary(chat_id, rnc=rnc)
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["total_monto"], Decimal("1770.00"))
        self.assertEqual(summary["total_itbis"], Decimal("270.00"))
        self.assertEqual(summary["total_servicios"], Decimal("1000.00"))
        self.assertEqual(summary["total_bienes"], Decimal("500.00"))
        self.assertEqual(len(summary["items"]), 2)

        # Test deleting invoice 1
        deleted = self.sm.delete_invoice_from_batch(chat_id, "B0100000001", rnc=rnc)
        self.assertTrue(deleted)
        self.assertEqual(self.sm.batch_count(chat_id, rnc=rnc), 1)

        # Verify updated summary
        new_summary = self.sm.get_batch_summary(chat_id, rnc=rnc)
        self.assertEqual(new_summary["count"], 1)
        self.assertEqual(new_summary["total_monto"], Decimal("590.00"))

        # Deleting non-existent invoice should return False
        self.assertFalse(self.sm.delete_invoice_from_batch(chat_id, "B0199999999", rnc=rnc))

    def test_archive_batch_and_duplicate_export_detection(self):
        chat_id = 7777
        rnc = "131000001"
        period = "202610"

        inv = {
            "rnc_cedula": "101000001",
            "ncf": "E310000000001",
            "fecha_comprobante": "20261005",
            "monto_total": "2360.00",
            "itbis_facturado": "360.00",
            "monto_servicios": "2000.00",
            "monto_bienes": "0.00",
            "tipo_bienes_servicios": "02",
            "forma_pago": "02",
        }
        self.sm.add_confirmed_invoice(chat_id, inv, rnc=rnc)

        # Prior to archiving, it shouldn't show as previously exported
        self.assertIsNone(self.sm.is_invoice_previously_exported(rnc, "E310000000001"))

        # Archive batch
        archived_count = self.sm.archive_batch(chat_id, rnc, period)
        self.assertEqual(archived_count, 1)

        # Now it should be detected as previously exported in 202610
        prev = self.sm.is_invoice_previously_exported(rnc, "E310000000001")
        self.assertIsNotNone(prev)
        self.assertEqual(prev["period"], "202610")


if __name__ == "__main__":
    unittest.main()
