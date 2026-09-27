"""Photo reception and asynchronous OCR execution."""

from __future__ import annotations

import asyncio
import logging

from telegram import Update
from telegram.ext import ContextTypes

from handlers.invoice_flow import dispatch_ocr_result
from ocr_engine import InvoiceOCREngine
from state_manager import UserState, state_manager

logger = logging.getLogger(__name__)


async def handle_invoice_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    chat_id = update.effective_chat.id
    if state_manager.get_session(chat_id).state != UserState.IDLE:
        await message.reply_text("Ya hay una factura en proceso. Confírmala, corrígela o usa /cancelar antes de enviar otra.")
        return
    state_manager.set_state(chat_id, UserState.PROCESSING_OCR)
    photo = message.photo[-1]
    await message.reply_text("Foto recibida. Analizando la factura.")
    try:
        telegram_file = await context.bot.get_file(photo.file_id)
        image_bytes = bytes(await telegram_file.download_as_bytearray())
        engine: InvoiceOCREngine = context.bot_data["ocr_engine"]
        result = await asyncio.to_thread(engine.analyze, image_bytes)
    except Exception:
        logger.exception("Error procesando foto", extra={"chat_id": chat_id})
        state_manager.clear_session(chat_id)
        await message.reply_text("No pude procesar la imagen. Envía una foto nítida de la factura e intenta de nuevo.")
        return
    await dispatch_ocr_result(update, context, result)
