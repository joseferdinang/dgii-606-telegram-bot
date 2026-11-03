"""PDF reception, page rendering and asynchronous OCR / multimodal vision execution."""

from __future__ import annotations

import asyncio
import io
import logging

from pdf2image import convert_from_bytes
from telegram import Update
from telegram.ext import ContextTypes

from handlers.invoice_flow import dispatch_ocr_result
from ocr_engine import InvoiceOCREngine, OCRResult, REQUIRED_FIELDS
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

    if state_manager.get_session(chat_id).state != UserState.IDLE:
        await message.reply_text("Ya hay una factura en proceso. Confírmala, corrígela o usa /cancelar antes de enviar otra.")
        return
    document = message.document
    if document.file_size and document.file_size > MAX_PDF_SIZE_MB * 1024 * 1024:
        await message.reply_text(f"El PDF supera {MAX_PDF_SIZE_MB} MB. Envía un archivo más pequeño o una foto.")
        return
    state_manager.set_state(chat_id, UserState.PROCESSING_OCR)
    await message.reply_text("PDF recibido. Convirtiendo y analizando páginas.")
    
    pages = []
    try:
        telegram_file = await context.bot.get_file(document.file_id)
        pdf_bytes = bytes(await telegram_file.download_as_bytearray())
        pages = await asyncio.to_thread(_pdf_to_images, pdf_bytes)
        if not pages:
            raise ValueError("PDF sin páginas")
        engine: InvoiceOCREngine = context.bot_data["ocr_engine"]
        result = await asyncio.to_thread(engine.analyze_pdf_pages, pages)
    except Exception:
        logger.warning(
            "Google Vision OCR falló o no disponible para PDF en chat_id=%s. Activando fallback directo a GPT Multimodal",
            chat_id,
            exc_info=True,
        )
        first_page = pages[0] if pages else None
        if not first_page:
            state_manager.clear_session(chat_id)
            await message.reply_text("No pude procesar el archivo PDF. Asegúrate de que no esté dañado o protegido con contraseña.")
            return
        result = OCRResult(raw_text="", image_bytes=first_page, missing_fields=list(REQUIRED_FIELDS))

    await dispatch_ocr_result(update, context, result)
