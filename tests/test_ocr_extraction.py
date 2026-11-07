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


    def test_extract_rnc_excludes_client_rnc(self):
        text = 'DATOS DEL CLIENTE:\nCLIENTE: JUAN PEREZ\nRNC: 131-00000-1\n\nDATOS DEL EMISOR:\nTOTALENERGIES\nRNC: 101-06874-4'
        res = InvoiceOCREngine._extract_rnc_cedula(text)
        self.assertIsNotNone(res)
        self.assertEqual(res.value, '101068744')

    def test_extract_rnc_discards_buyer_rnc(self):
        text = 'FACTURA CON RNC: 101-00817-2 Y RNC: 101-06874-4'
        res = InvoiceOCREngine._extract_rnc_cedula(text, buyer_rnc='101008172')
        self.assertIsNotNone(res)
        self.assertEqual(res.value, '101068744')

    def test_extract_rnc_prefers_valid_check_digit(self):
        text = '101068749 101068744'
        res = InvoiceOCREngine._extract_rnc_cedula(text)
        self.assertIsNotNone(res)
        self.assertEqual(res.value, '101068744')

    def test_real_glp_receipt_ncf_and_fields(self):
        text = (
            "CARDNET\nCOMERCIO: ESTACION SIGLO XXI\nTOTAL : RD$1,200.00\n"
            "SiGLO XXI FACTURA DE CREDITO\nRNC:131240674 NCF:E310000036276\n"
            "Fecha: 11/10/2026 Valida Hasta: 31/12/2026\n"
            "RNC CLIENTE: 03100308018\nSubtotal 1,200.00\nIT8IS: 0.00\nTotal: 1.200.00"
        )
        eng = InvoiceOCREngine()
        fields, _ = eng._parse_fields(text, buyer_rnc="03100308018")
        self.assertEqual(fields.get("rnc_cedula"), "131240674")
        self.assertEqual(fields.get("rnc_comprador"), "03100308018")
        self.assertEqual(fields.get("ncf"), "E310000036276")
        self.assertEqual(fields.get("fecha_comprobante"), "20261011")
        self.assertEqual(fields.get("monto_total"), "1200.00")
        self.assertEqual(fields.get("itbis_facturado"), "0.00")
        self.assertEqual(fields.get("forma_pago"), "03")

    def test_single_field_correction_callback_and_usd_validation(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from handlers.confirmation_handler import handle_confirmation_callback
        from state_manager import state_manager, UserState
        from fiscal_validators import validate_invoice

        # 1. Verify USD converted invoice rounding tolerance passes validation
        usd_inv = {
            "rnc_cedula": "102613923",
            "ncf": "E310000015210",
            "fecha_comprobante": "20261015",
            "monto_total": "1169.26",
            "itbis_facturado": "178.60",
            "monto_bienes": "990.66",
            "monto_servicios": "0.00",
            "tipo_bienes_servicios": "09",
            "forma_pago": "04",
            "tasa_cambio": "61.80",
            "_usd_converted": True,
        }
        self.assertEqual(validate_invoice(usd_inv), [])

        # 2. Verify field-specific correction menu and single-field prompt
        chat_id = 999123
        state_manager.clear_session(chat_id)
        sess = state_manager.get_session(chat_id)
        sess.state = UserState.WAITING_CONFIRMATION
        sess.extracted_data = dict(usd_inv)
        state_manager.update_session(chat_id, sess)

        update = MagicMock()
        update.message = None
        update.effective_chat.id = chat_id
        update.callback_query.data = "correct_invoice"
        update.callback_query.answer = AsyncMock()
        update.callback_query.edit_message_text = AsyncMock()
        update.callback_query.message.reply_text = AsyncMock()
        update.effective_message = update.callback_query.message
        context = MagicMock()

        asyncio.run(handle_confirmation_callback(update, context))
        # State remains WAITING_CONFIRMATION while showing the field menu
        self.assertEqual(state_manager.get_session(chat_id).state, UserState.WAITING_CONFIRMATION)
        update.callback_query.edit_message_text.assert_called_once()
        _, kwargs = update.callback_query.edit_message_text.call_args
        self.assertIsNotNone(kwargs.get("reply_markup"))

        # Now select only NCF to correct
        update.callback_query.data = "correct_field:ncf"
        asyncio.run(handle_confirmation_callback(update, context))
        sess_after = state_manager.get_session(chat_id)
        self.assertEqual(sess_after.state, UserState.WAITING_MANUAL_FIELD)
        self.assertEqual(sess_after.field_being_corrected, "ncf")
        self.assertEqual(sess_after.missing_fields, ["ncf"])
        # Allocation fields are preserved when correcting NCF
        self.assertEqual(sess_after.extracted_data.get("monto_bienes"), "990.66")
        state_manager.clear_session(chat_id)

if __name__ == '__main__':
    unittest.main()
