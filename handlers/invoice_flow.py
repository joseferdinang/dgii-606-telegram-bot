"""Shared post-OCR routing for photos and PDF invoices."""

from __future__ import annotations

import asyncio
import logging

from telegram import Update
from telegram.ext import ContextTypes

from fiscal_validators import normalize_extracted_fields
from handlers.confirmation_handler import send_missing_field_prompt
from ocr_engine import REQUIRED_FIELDS
from state_manager import UserState, state_manager

logger = logging.getLogger(__name__)


async def dispatch_ocr_result(update: Update, context: ContextTypes.DEFAULT_TYPE, result) -> None:
    chat_id = update.effective_chat.id
    target_msg = update.effective_message or (update.callback_query.message if update.callback_query else None)
    
    raw_fields = dict(getattr(result, "fields", {}))
    raw_text = getattr(result, "raw_text", "")
    image_bytes = getattr(result, "image_bytes", None)

    active_company = state_manager.get_active_company(chat_id)
    buyer_rnc = active_company.rnc if active_company else None

    # 1. Extractor LLM (OpenAI ChatGPT)
    llm_extractor = context.bot_data.get("llm_extractor") if context and hasattr(context, "bot_data") else None
    if llm_extractor and llm_extractor.is_available:
        try:
            llm_extracted = {}
            # Comprobar si QR o Vision ya obtuvieron todos los campos esenciales (RNC, NCF, Monto Total)
            has_essential_fields = bool(raw_fields.get("rnc_cedula") and raw_fields.get("ncf") and raw_fields.get("monto_total"))

            # Si NO hay texto de OCR, o si faltan campos esenciales pero disponemos de la imagen / página PDF:
            # ¡Activamos el modelo de visión multimodal de GPT directamente sobre la imagen!
            if (not raw_text.strip() or not has_essential_fields) and image_bytes:
                logger.info("Activando inspección visual directa con GPT Multimodal para chat_id=%s", chat_id)
                llm_extracted = await asyncio.to_thread(llm_extractor.extract_from_image, image_bytes, buyer_rnc=buyer_rnc)

            # Si la inspección de imagen no se activó o no arrojó campos, pero sí hay texto de OCR:
            if not llm_extracted and raw_text.strip():
                logger.info("Extrayendo campos con GPT vía texto OCR para chat_id=%s", chat_id)
                llm_extracted = await asyncio.to_thread(llm_extractor.extract_from_text, raw_text, buyer_rnc=buyer_rnc)

            if llm_extracted:
                logger.info("LLM extraction completed for chat_id=%s. Extracted fields: %s", chat_id, list(llm_extracted.keys()))
                # Preservar campos que vinieron del código QR (confianza 1.0)
                qr_fields = {
                    k: v for k, v in raw_fields.items()
                    if getattr(result, "confidence", {}).get(k) == 1.0
                }
                # Los campos de ChatGPT tienen alta prioridad sobre regex / heurística
                for k, v in llm_extracted.items():
                    if v not in (None, ""):
                        raw_fields[k] = v
                # Re-aplicar QR para asegurar integridad absoluta
                raw_fields.update(qr_fields)
        except Exception:
            logger.exception("Error running LLM extractor for chat_id=%s", chat_id)

    # 2. Extractor ML secundario (si está disponible y hay campos faltantes)
    ml_extractor = context.bot_data.get("ml_extractor") if context and hasattr(context, "bot_data") else None
    if ml_extractor and ml_extractor.available and raw_text:
        try:
            ml_extracted = ml_extractor.extract(raw_text)
            raw_fields = ml_extractor.merge_with_regex(raw_fields, ml_extracted)
            logger.info("ML extraction merged for chat_id=%s. Fields: %s", chat_id, list(ml_extracted.keys()))
        except Exception:
            logger.exception("Error running ML extractor on OCR text for chat_id=%s", chat_id)

    # Evitar que el RNC de la empresa declarante se asigne como emisor/proveedor
    if buyer_rnc:
        import re
        clean_buyer = re.sub(r"\D", "", buyer_rnc)
        clean_vendor = re.sub(r"\D", "", str(raw_fields.get("rnc_cedula") or ""))
        if clean_vendor and clean_vendor == clean_buyer:
            logger.info("El RNC extraído (%s) coincide con la empresa compradora. Removiéndolo de vendor RNC", clean_vendor)
            raw_fields.pop("rnc_cedula", None)
            if raw_text:
                from ocr_engine import InvoiceOCREngine
                alt_cand = InvoiceOCREngine._extract_rnc_cedula(raw_text, buyer_rnc=buyer_rnc)
                if alt_cand:
                    raw_fields["rnc_cedula"] = alt_cand.value

    fields, invalid_fields = normalize_extracted_fields(raw_fields)
    missing = list(dict.fromkeys([
        field for field in REQUIRED_FIELDS if not fields.get(field)
    ] + invalid_fields))

    existing_period = state_manager.batch_period(chat_id, rnc=buyer_rnc)
    new_period = str(fields.get("fecha_comprobante", ""))[:6]
    if existing_period and new_period and existing_period != new_period:
        state_manager.clear_session(chat_id)
        if target_msg:
            await target_msg.reply_text(
                f"⚠️ El lote actual corresponde al período fiscal {existing_period}, pero esta factura es del período {new_period}.\n"
                "Usa /terminar para exportar el lote actual antes de comenzar facturas de otro período.",
            )
        from handlers.queue_handler import process_next_queued_media
        await process_next_queued_media(update, context)
        return

    # Advertencia si la factura ya fue reportada en un lote exportado con anterioridad
    if buyer_rnc and fields.get("ncf"):
        prev_export = state_manager.is_invoice_previously_exported(buyer_rnc, fields["ncf"])
        if prev_export and target_msg:
            await target_msg.reply_text(
                f"⚠️ *Atención:* El NCF `{fields['ncf']}` ya fue reportado previamente en el período {prev_export['period']}.\n"
                "Verifica si se trata de un duplicado accidental.",
                parse_mode="Markdown",
            )

    if raw_text:
        fields["_raw_text"] = raw_text

    state_manager.save_extracted_data(chat_id, fields, missing, company_rnc=buyer_rnc or "")
    if missing:
        logger.info("Campos OCR pendientes de corrección para chat_id=%s: %s", chat_id, missing)
        await send_missing_field_prompt(update, context, missing)
        return

    if fields.get("moneda") == "USD" and not fields.get("tasa_cambio"):
        state_manager.set_state(chat_id, UserState.WAITING_EXCHANGE_RATE)
        if target_msg:
            await target_msg.reply_text(
                "💵 La factura está en USD. Envía la tasa de cambio RD$/USD a utilizar (por ejemplo: 61.50):"
            )
        return
        
    from handlers.confirmation_handler import proceed_to_next_step
    await proceed_to_next_step(update, context)
