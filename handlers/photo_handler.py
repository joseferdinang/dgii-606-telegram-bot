"""Photo reception and asynchronous OCR / multimodal vision execution with batch queue support."""

from __future__ import annotations

import asyncio
import logging

from telegram import Update
from telegram.ext import ContextTypes

from handlers.invoice_flow import dispatch_ocr_result
from ocr_engine import InvoiceOCREngine, OCRResult, REQUIRED_FIELDS
from state_manager import UserState, state_manager

logger = logging.getLogger(__name__)


async def handle_invoice_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    chat_id = update.effective_chat.id

    # Verificación de empresa activa para aislamiento y seguridad
    active_company = state_manager.get_active_company(chat_id)
    settings = context.bot_data.get("settings") if context and hasattr(context, "bot_data") else None
    if not active_company and not (settings and getattr(settings, "reporting_rnc", None)):
        await message.reply_text(
            "⚠️ *Acceso restringido: no has iniciado sesión en ninguna empresa.*\n\n"
            "Para comenzar a declarar facturas en el 606, identifícate con tu empresa:\n"
            "👉 `/login <RNC> <PIN>`\n\n"
            "O si tu empresa es nueva en el sistema, regístrala con:\n"
            "👉 `/registro <RNC> <PIN> <Nombre de la Empresa>`",
            parse_mode="Markdown",
        )
        return

    photo = message.photo[-1]
    file_id = photo.file_id
    rnc_scope = active_company.rnc if active_company else None

    # Si ya hay una factura en revisión o procesándose, encolamos esta imagen
    if state_manager.get_session(chat_id).state != UserState.IDLE:
        count = state_manager.enqueue_media(
            chat_id,
            file_id,
            "photo",
            media_group_id=message.media_group_id,
            rnc=rnc_scope,
        )
        await message.reply_text(
            f"📥 Foto añadida a la cola (Facturas en espera: {count}).\n"
            "Se procesará automáticamente en cuanto confirmes o descartes la factura actual."
        )
        return

    state_manager.set_state(chat_id, UserState.PROCESSING_OCR)
    await message.reply_text("Foto recibida. Analizando la factura...")
    
    image_bytes = None
    try:
        telegram_file = await context.bot.get_file(file_id)
        image_bytes = bytes(await telegram_file.download_as_bytearray())
        engine: InvoiceOCREngine = context.bot_data["ocr_engine"]
        buyer_rnc = active_company.rnc if active_company else None
        result = await asyncio.to_thread(engine.analyze, image_bytes, buyer_rnc=buyer_rnc)
    except Exception:
        logger.warning(
            "Google Vision OCR no disponible o falló para chat_id=%s. Activando fallback directo a GPT Multimodal",
            chat_id,
            exc_info=True,
        )
        if not image_bytes:
            state_manager.clear_session(chat_id)
            await message.reply_text("No pude descargar la imagen de Telegram. Por favor intenta enviarla de nuevo.")
            return
        result = OCRResult(raw_text="", image_bytes=image_bytes, missing_fields=list(REQUIRED_FIELDS))

    await dispatch_ocr_result(update, context, result)
