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
    message = update.message
    
    raw_fields = dict(getattr(result, "fields", {}))
    raw_text = getattr(result, "raw_text", "")
    image_bytes = getattr(result, "image_bytes", None)

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
                llm_extracted = await asyncio.to_thread(llm_extractor.extract_from_image, image_bytes)

            # Si la inspección de imagen no se activó o no arrojó campos, pero sí hay texto de OCR:
            if not llm_extracted and raw_text.strip():
                logger.info("Extrayendo campos con GPT vía texto OCR para chat_id=%s", chat_id)
                llm_extracted = await asyncio.to_thread(llm_extractor.extract_from_text, raw_text)

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

    fields, invalid_fields = normalize_extracted_fields(raw_fields)
    missing = list(dict.fromkeys([
        field for field in REQUIRED_FIELDS if not fields.get(field)
    ] + invalid_fields))

    existing_period = state_manager.batch_period(chat_id)
    new_period = str(fields.get("fecha_comprobante", ""))[:6]
    if existing_period and new_period and existing_period != new_period:
        await message.reply_text(
            f"El lote actual corresponde al período {existing_period}. "
            "Escribe Terminar para exportarlo antes de iniciar facturas de otro período."
        )
        return

    if raw_text:
        fields["_raw_text"] = raw_text

    state_manager.save_extracted_data(chat_id, fields, missing)
    if missing:
        logger.info("Campos OCR pendientes de corrección para chat_id=%s: %s", chat_id, missing)
        await send_missing_field_prompt(update, context, missing)
        return

    if fields.get("moneda") == "USD" and not fields.get("tasa_cambio"):
        state_manager.set_state(chat_id, UserState.WAITING_EXCHANGE_RATE)
        await message.reply_text(
            "La factura está en USD. Envía la tasa RD$/USD a utilizar, por ejemplo 61.50."
        )
        return
        
    from handlers.confirmation_handler import proceed_to_next_step
    await proceed_to_next_step(update, context)
