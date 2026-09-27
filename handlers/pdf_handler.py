"""PDF reception, page rendering and asynchronous OCR execution."""

from __future__ import annotations

import asyncio
import io
import logging

from pdf2image import convert_from_bytes
from telegram import Update
from telegram.ext import ContextTypes

from handlers.invoice_flow import dispatch_ocr_result
from ocr_engine import InvoiceOCREngine
from state_manager import UserState, state_manager

logger = logging.getLogger(__name__)
PDF_RENDER_DPI = 200
MAX_PDF_SIZE_MB = 20
MAX_PDF_PAGES = 10


def _pdf_to_images(pdf_bytes: bytes) -> list[bytes]:
    images = convert_from_bytes(pdf_bytes, dpi=PDF_RENDER_DPI, fmt="jpeg", first_page=1, last_page=MAX_PDF_PAGES)
    pages: list[bytes] = []
    for image in images:
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95)
        pages.append(output.getvalue())
    return pages


async def handle_invoice_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    chat_id = update.effective_chat.id
    if state_manager.get_session(chat_id).state != UserState.IDLE:
        await message.reply_text("Ya hay una factura en proceso. Confírmala, corrígela o usa /cancelar antes de enviar otra.")
        return
    document = message.document
    if document.file_size and document.file_size > MAX_PDF_SIZE_MB * 1024 * 1024:
        await message.reply_text(f"El PDF supera {MAX_PDF_SIZE_MB} MB. Envía un archivo más pequeño o una foto.")
        return
    state_manager.set_state(chat_id, UserState.PROCESSING_OCR)
    await message.reply_text("PDF recibido. Convirtiendo y analizando páginas.")
    try:
        telegram_file = await context.bot.get_file(document.file_id)
        pdf_bytes = bytes(await telegram_file.download_as_bytearray())
        pages = await asyncio.to_thread(_pdf_to_images, pdf_bytes)
        if not pages:
            raise ValueError("PDF sin páginas")
        engine: InvoiceOCREngine = context.bot_data["ocr_engine"]
        result = await asyncio.to_thread(engine.analyze_pdf_pages, pages)
    except Exception:
        logger.exception("Error procesando PDF", extra={"chat_id": chat_id})
        state_manager.clear_session(chat_id)
        await message.reply_text("No pude procesar el PDF. Verifica que no tenga contraseña y vuelve a intentarlo.")
        return
    await dispatch_ocr_result(update, context, result)
