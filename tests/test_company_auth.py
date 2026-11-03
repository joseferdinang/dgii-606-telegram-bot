"""Pruebas unitarias de autenticación, control de acceso y aislamiento multi-empresa."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from state_manager import StateManager, normalize_rnc


def sample_invoice(ncf: str) -> dict:
    return {
        "rnc_cedula": "131000001",
        "ncf": ncf,
        "fecha_comprobante": "20260801",
        "monto_total": "1180.00",
        "itbis_facturado": "180.00",
        "monto_servicios": "0.00",
        "monto_bienes": "1000.00",
        "tipo_bienes_servicios": "02",
        "forma_pago": "01",
    }


class TestCompanyAuthAndIsolation(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_auth.sqlite3"
        self.sm = StateManager(self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_rnc_normalization(self) -> None:
        self.assertEqual(normalize_rnc("1-31-00000-1"), "131000001")
        self.assertEqual(normalize_rnc(" 001-1234567-8 "), "00112345678")
        self.assertEqual(normalize_rnc("101.222.333"), "101222333")

    def test_register_company_success(self) -> None:
        company = self.sm.register_company(
            chat_id=101,
            rnc="1-31-00000-1",
            razon_social="Constructora Gómez SRL",
            pin="1234",
        )
        self.assertEqual(company.rnc, "131000001")
        self.assertEqual(company.razon_social, "Constructora Gómez SRL")
        self.assertTrue(company.is_active)

        active = self.sm.get_active_company(101)
        self.assertIsNotNone(active)
        self.assertEqual(active.rnc, "131000001")

    def test_register_invalid_inputs(self) -> None:
        # RNC con longitud inválida
        with self.assertRaises(ValueError):
            self.sm.register_company(101, "12345678", "Empresa", "1234")  # 8 dígitos
        with self.assertRaises(ValueError):
            self.sm.register_company(101, "1234567890", "Empresa", "1234")  # 10 dígitos

        # PIN demasiado corto
        with self.assertRaises(ValueError):
            self.sm.register_company(101, "131000001", "Empresa", "12")

        # Razón social vacía
        with self.assertRaises(ValueError):
            self.sm.register_company(101, "131000001", "   ", "1234")

    def test_duplicate_registration_rejected(self) -> None:
        self.sm.register_company(101, "131000001", "Empresa Uno", "1234")
        with self.assertRaises(ValueError) as ctx:
            self.sm.register_company(102, "131000001", "Empresa Dos", "9999")
        self.assertIn("ya está registrado", str(ctx.exception))

    def test_login_success_and_failure(self) -> None:
        self.sm.register_company(101, "131000001", "Empresa Uno", "secret123")

        # Login correcto por otro usuario autorizado (ej: colaborador con chat_id 102)
        login_comp = self.sm.login_company(102, "131000001", "secret123")
        self.assertEqual(login_comp.rnc, "131000001")
        self.assertTrue(login_comp.is_active)

        # Login con PIN incorrecto
        with self.assertRaises(ValueError) as ctx:
            self.sm.login_company(103, "131000001", "wrong_pin")
        self.assertIn("RNC o PIN incorrecto", str(ctx.exception))

        # Login con RNC que no existe
        with self.assertRaises(ValueError) as ctx:
            self.sm.login_company(103, "999999999", "secret123")
        self.assertIn("RNC o PIN incorrecto", str(ctx.exception))

    def test_tenant_isolation_users_cannot_see_other_companies(self) -> None:
        """Una persona no puede ver ni listar el RNC o la empresa de otro usuario."""
        # Usuario 1 registra Empresa A
        self.sm.register_company(chat_id=1, rnc="131000001", razon_social="Empresa A", pin="1111")
        # Usuario 2 registra Empresa B
        self.sm.register_company(chat_id=2, rnc="101222222", razon_social="Empresa B", pin="2222")

        # Usuario 1 solo ve Empresa A
        comps_user1 = self.sm.get_user_companies(chat_id=1)
        self.assertEqual(len(comps_user1), 1)
        self.assertEqual(comps_user1[0].rnc, "131000001")
        self.assertEqual(comps_user1[0].razon_social, "Empresa A")

        # Usuario 2 solo ve Empresa B
        comps_user2 = self.sm.get_user_companies(chat_id=2)
        self.assertEqual(len(comps_user2), 1)
        self.assertEqual(comps_user2[0].rnc, "101222222")
        self.assertEqual(comps_user2[0].razon_social, "Empresa B")

        # Usuario 1 NO PUEDE cambiar a Empresa B porque no está autorizado
        with self.assertRaises(PermissionError):
            self.sm.switch_active_company(chat_id=1, rnc="101222222")

    def test_multi_company_switching_and_batch_isolation(self) -> None:
        """Un contador que maneja dos empresas puede alternar entre ellas sin mezclar facturas."""
        # Contador (chat_id=50) registra o se loguea en dos empresas distintas
        self.sm.register_company(chat_id=50, rnc="131000001", razon_social="Cliente Alfa", pin="1111")
        self.sm.register_company(chat_id=50, rnc="101222222", razon_social="Cliente Beta", pin="2222")

        # La última registrada (Beta) es la activa
        self.assertEqual(self.sm.get_active_company(50).rnc, "101222222")

        # Agrega factura al lote de Cliente Beta
        self.sm.add_confirmed_invoice(50, sample_invoice("B0100000001"))
        self.assertEqual(self.sm.batch_count(50), 1)
        self.assertEqual(self.sm.get_batch(50)[0]["ncf"], "B0100000001")

        # Cambia a Cliente Alfa
        self.sm.switch_active_company(50, "131000001")
        self.assertEqual(self.sm.get_active_company(50).rnc, "131000001")
        # El lote de Alfa está inicialmente vacío
        self.assertEqual(self.sm.batch_count(50), 0)

        # Agrega factura al lote de Cliente Alfa
        self.sm.add_confirmed_invoice(50, sample_invoice("B0100000002"))
        self.assertEqual(self.sm.batch_count(50), 1)
        self.assertEqual(self.sm.get_batch(50)[0]["ncf"], "B0100000002")

        # Cambia nuevamente a Beta: su factura B0100000001 sigue intacta
        self.sm.switch_active_company(50, "101222222")
        self.assertEqual(self.sm.batch_count(50), 1)
        self.assertEqual(self.sm.get_batch(50)[0]["ncf"], "B0100000001")

    def test_logout_behavior(self) -> None:
        self.sm.register_company(chat_id=99, rnc="131000001", razon_social="Mi Empresa", pin="1234")
        self.assertIsNotNone(self.sm.get_active_company(99))

        self.sm.logout_company(chat_id=99)
        self.assertIsNone(self.sm.get_active_company(99))
        self.assertEqual(len(self.sm.get_user_companies(99)), 0)
