from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from dgii_txt import (
    FormaPago,
    Record606,
    TipoBienesServicios,
    TipoId,
    TipoRetencionISR,
    generate_606_txt,
    MAX_REGISTROS,
)


class TestDgiiTxtGenerator(unittest.TestCase):
    def setUp(self):
        self.reporting_rnc = "131000001"
        self.sample_invoices = [
            {
                "rnc_cedula": "101123456",
                "tipo_id": "1",
                "tipo_bienes_servicios": "02",
                "ncf": "B0100000001",
                "ncf_modificado": "",
                "fecha_comprobante": "20261015",
                "fecha_pago": "",
                "monto_servicios": "0.00",
                "monto_bienes": "1000.00",
                "itbis_facturado": "180.00",
                "itbis_retenido": "0.00",
                "itbis_proporcionalidad": "0.00",
                "itbis_llevado_costo": "0.00",
                "itbis_percibido_compras": "0.00",
                "tipo_retencion_isr": "",
                "isr_retenido": "0.00",
                "isr_percibido_compras": "0.00",
                "impuesto_selectivo_consumo": "0.00",
                "otros_impuestos_tasas": "0.00",
                "monto_propina_legal": "0.00",
                "forma_pago": "03",
            },
            {
                "rnc_cedula": "00100000002",
                "tipo_id": "2",
                "tipo_bienes_servicios": "01",
                "ncf": "B0100000002",
                "ncf_modificado": "",
                "fecha_comprobante": "20261020",
                "fecha_pago": "20261025",
                "monto_servicios": "5000.00",
                "monto_bienes": "0.00",
                "itbis_facturado": "900.00",
                "itbis_retenido": "900.00",
                "itbis_proporcionalidad": "0.00",
                "itbis_llevado_costo": "0.00",
                "itbis_percibido_compras": "0.00",
                "tipo_retencion_isr": "02",
                "isr_retenido": "500.00",
                "isr_percibido_compras": "0.00",
                "impuesto_selectivo_consumo": "0.00",
                "otros_impuestos_tasas": "0.00",
                "monto_propina_legal": "0.00",
                "forma_pago": "02",
            },
            {
                "rnc_cedula": "101123456",
                "tipo_id": "1",
                "tipo_bienes_servicios": "02",
                "ncf": "B0400000001",
                "ncf_modificado": "B0100000001",
                "fecha_comprobante": "20261028",
                "fecha_pago": "",
                "monto_servicios": "0.00",
                "monto_bienes": "200.00",
                "itbis_facturado": "36.00",
                "itbis_retenido": "0.00",
                "itbis_proporcionalidad": "0.00",
                "itbis_llevado_costo": "0.00",
                "itbis_percibido_compras": "0.00",
                "tipo_retencion_isr": "",
                "isr_retenido": "0.00",
                "isr_percibido_compras": "0.00",
                "impuesto_selectivo_consumo": "0.00",
                "otros_impuestos_tasas": "0.00",
                "monto_propina_legal": "0.00",
                "forma_pago": "06",
            },
            {
                "rnc_cedula": "999999999",  # Foreign/arbitrary, must be replaced by reporting RNC
                "tipo_id": "1",
                "tipo_bienes_servicios": "02",
                "ncf": "B1700000001",
                "ncf_modificado": "",
                "fecha_comprobante": "20261029",
                "fecha_pago": "",
                "monto_servicios": "12000.00",
                "monto_bienes": "0.00",
                "itbis_facturado": "0.00",
                "itbis_retenido": "500.00",  # Should be omitted for B17 in 606
                "itbis_proporcionalidad": "0.00",
                "itbis_llevado_costo": "0.00",
                "itbis_percibido_compras": "0.00",
                "tipo_retencion_isr": "02",  # Should be omitted for B17 in 606
                "isr_retenido": "3240.00",   # Should be omitted for B17 in 606
                "isr_percibido_compras": "0.00",
                "impuesto_selectivo_consumo": "0.00",
                "otros_impuestos_tasas": "0.00",
                "monto_propina_legal": "0.00",
                "forma_pago": "02",
            },
        ]

    def test_record_calculations_and_types(self):
        inv = self.sample_invoices[0]
        rec = Record606.from_dict(inv, reporting_rnc=self.reporting_rnc)
        
        # Verify calculated fields
        self.assertEqual(rec.total_monto_facturado, Decimal("1000.00"))
        self.assertEqual(rec.itbis_por_adelantar, Decimal("180.00"))
        self.assertEqual(rec.tipo_id, TipoId.RNC)
        self.assertEqual(rec.tipo_bienes_servicios, TipoBienesServicios.GASTOS_TRABAJO_SUMINISTRO_SERVICIOS)
        self.assertEqual(rec.forma_pago, FormaPago.TARJETA_CREDITO_DEBITO)

    def test_line_has_exact_23_fields(self):
        for inv in self.sample_invoices:
            rec = Record606.from_dict(inv, reporting_rnc=self.reporting_rnc)
            line = rec.to_line()
            parts = line.split("|")
            self.assertEqual(len(parts), 23, f"Line does not have 23 fields: {line}")

    def test_b17_special_rules(self):
        b17_inv = self.sample_invoices[3]
        rec = Record606.from_dict(b17_inv, reporting_rnc=self.reporting_rnc)
        
        # Supplier RNC must be reporting RNC
        self.assertEqual(rec.rnc_cedula, self.reporting_rnc)
        # Retentions and optional taxes omitted/zeroed for 606
        self.assertEqual(rec.itbis_retenido, Decimal("0.00"))
        self.assertEqual(rec.isr_retenido, Decimal("0.00"))
        self.assertIsNone(rec.tipo_retencion_isr)

    def test_golden_reference_comparison(self):
        fixtures_path = Path(__file__).parent / "fixtures" / "sample_606_reference.txt"
        self.assertTrue(fixtures_path.exists(), "Reference fixture does not exist")

        with tempfile.TemporaryDirectory() as tmp_dir:
            generated_file = generate_606_txt(
                self.sample_invoices,
                reporting_rnc=self.reporting_rnc,
                output_dir=tmp_dir,
                period="202610",
            )
            self.assertEqual(generated_file.name, f"DGII_F_606_{self.reporting_rnc}_202610.TXT")

            with open(generated_file, "r", encoding="windows-1252") as f:
                generated_content = f.read()

            with open(fixtures_path, "r", encoding="windows-1252") as f:
                reference_content = f.read().replace("\r\n", "\n")

            # Normalizing CRLF for comparison
            self.assertEqual(
                generated_content.replace("\r\n", "\n").strip(),
                reference_content.strip()
            )

    def test_exceeding_max_records_raises_error(self):
        too_many = [self.sample_invoices[0]] * (MAX_REGISTROS + 1)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError) as ctx:
                generate_606_txt(
                    too_many,
                    reporting_rnc=self.reporting_rnc,
                    output_dir=tmp_dir,
                )
            self.assertIn("excediendo el límite máximo", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
