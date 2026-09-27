"""Unit tests for DGII e-CF QR code reader and parser."""
import unittest
from qr_reader import parse_dgii_qr_text, _parse_dgii_params


class TestQRReader(unittest.TestCase):
    def test_parse_dgii_url_standard(self):
        url = (
            "https://ecf.dgii.gov.do/consultatimbre?"
            "RncEmisor=101068744&"
            "RncComprador=130891234&"
            "eNCF=E310000084920&"
            "FechaEmision=20260924&"
            "MontoTotal=3646.28&"
            "MontoITBIS=0.00&"
            "CodigoSeguridad=7B9xA2"
        )
        fields = parse_dgii_qr_text(url)
        self.assertIsNotNone(fields)
        self.assertEqual(fields["rnc_cedula"], "101068744")
        self.assertEqual(fields["ncf"], "E310000084920")
        self.assertEqual(fields["fecha_comprobante"], "20260924")
        self.assertEqual(fields["monto_total"], "3646.28")
        self.assertEqual(fields["itbis_facturado"], "0.00")
        self.assertEqual(fields["codigo_seguridad"], "7B9xA2")
        self.assertEqual(fields["rnc_comprador"], "130891234")

    def test_parse_dgii_url_lowercase_params(self):
        url = (
            "https://ecf.dgii.gov.do/consultatimbre?"
            "rncemisor=101008172&"
            "encf=E320000049182&"
            "fechaemision=20260815&"
            "montototal=1500.00&"
            "montoitbis=270.00"
        )
        fields = parse_dgii_qr_text(url)
        self.assertIsNotNone(fields)
        self.assertEqual(fields["rnc_cedula"], "101008172")
        self.assertEqual(fields["ncf"], "E320000049182")
        self.assertEqual(fields["fecha_comprobante"], "20260815")
        self.assertEqual(fields["monto_total"], "1500.00")
        self.assertEqual(fields["itbis_facturado"], "270.00")

    def test_parse_pipe_delimited_qr(self):
        text = "101068744|130891234|E310000084920|20260924|3646.28|0.00|7B9xA2"
        fields = parse_dgii_qr_text(text)
        self.assertIsNotNone(fields)
        self.assertEqual(fields["rnc_cedula"], "101068744")
        self.assertEqual(fields["ncf"], "E310000084920")
        self.assertEqual(fields["fecha_comprobante"], "20260924")
        self.assertEqual(fields["monto_total"], "3646.28")
        self.assertEqual(fields["itbis_facturado"], "0.00")

    def test_non_dgii_qr_ignored(self):
        self.assertIsNone(parse_dgii_qr_text("https://example.com/menu"))
        self.assertIsNone(parse_dgii_qr_text("WIFI:S:MyNetwork;T:WPA;P:Password;;"))
        self.assertIsNone(parse_dgii_qr_text(""))


if __name__ == "__main__":
    unittest.main()
