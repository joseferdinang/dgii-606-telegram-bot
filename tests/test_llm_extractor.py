import json
from unittest.mock import MagicMock, patch
import unittest

from llm_extractor import LLMFieldExtractor


class TestLLMFieldExtractor(unittest.TestCase):
    def test_initialization_without_key(self):
        extractor = LLMFieldExtractor(api_key=None)
        self.assertFalse(extractor.is_available)
        self.assertEqual(extractor.extract_from_text("Factura de prueba"), {})
        self.assertEqual(extractor.extract_from_image(b"fake_image_bytes"), {})

    def test_normalization_logic(self):
        raw_llm_output = {
            "rnc_cedula": "101-12345-6",
            "ncf": "b0100000001",
            "ncf_modificado": None,
            "fecha_comprobante": "20261015",
            "fecha_pago": None,
            "monto_total": 1180.0,
            "monto_servicios": 0.0,
            "monto_bienes": 1000.0,
            "itbis_facturado": 180.0,
            "itbis_retenido": 0.0,
            "tipo_bienes_servicios": "2",
            "forma_pago": "3",
            "monto_propina_legal": 0.0,
            "impuesto_selectivo_consumo": 0.0,
            "otros_impuestos_tasas": 0.0,
            "moneda": "DOP",
            "tasa_cambio": None,
        }

        normalized = LLMFieldExtractor._normalize_extracted_dict(raw_llm_output)

        self.assertEqual(normalized["rnc_cedula"], "101123456")
        self.assertEqual(normalized["ncf"], "B0100000001")
        self.assertEqual(normalized["fecha_comprobante"], "20261015")
        self.assertEqual(normalized["monto_total"], "1180.00")
        self.assertEqual(normalized["monto_bienes"], "1000.00")
        self.assertEqual(normalized["monto_servicios"], "0.00")
        self.assertEqual(normalized["itbis_facturado"], "180.00")
        self.assertEqual(normalized["tipo_bienes_servicios"], "02")
        self.assertEqual(normalized["forma_pago"], "03")
        self.assertEqual(normalized["moneda"], "DOP")

    @patch("openai.OpenAI")
    def test_extract_from_text_mocked(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "rnc_cedula": "131000001",
            "ncf": "B0100000005",
            "ncf_modificado": None,
            "fecha_comprobante": "20261025",
            "fecha_pago": None,
            "monto_total": 2360.0,
            "monto_servicios": 2000.0,
            "monto_bienes": 0.0,
            "itbis_facturado": 360.0,
            "itbis_retenido": 0.0,
            "tipo_bienes_servicios": "02",
            "forma_pago": "02",
            "monto_propina_legal": 0.0,
            "impuesto_selectivo_consumo": 0.0,
            "otros_impuestos_tasas": 0.0,
            "moneda": "DOP",
            "tasa_cambio": None,
        })
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        extractor = LLMFieldExtractor(api_key="fake-sk-key-for-test", model="gpt-4o-mini")
        self.assertTrue(extractor.is_available)

        sample_ocr = "COMERCIAL X SRL\nRNC: 131-00000-1\nNCF: B0100000005\nSUBTOTAL: 2,000.00\nITBIS: 360.00\nTOTAL: 2,360.00"
        result = extractor.extract_from_text(sample_ocr)

        self.assertEqual(result.get("rnc_cedula"), "131000001")
        self.assertEqual(result.get("ncf"), "B0100000005")
        self.assertEqual(result.get("monto_total"), "2360.00")
        self.assertEqual(result.get("itbis_facturado"), "360.00")
        self.assertEqual(result.get("monto_servicios"), "2000.00")
        self.assertEqual(result.get("tipo_bienes_servicios"), "02")
        self.assertEqual(result.get("forma_pago"), "02")

    @patch("openai.OpenAI")
    def test_extract_handles_api_exception_gracefully(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("OpenAI API rate limit or network error")

        extractor = LLMFieldExtractor(api_key="fake-sk-key-for-test")
        result = extractor.extract_from_text("Factura texto")
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
