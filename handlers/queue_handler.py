"""Queue processing for batch photo and PDF invoice ingestion."""
from __future__ import annotations

import asyncio
import logging

from telegram import Update
from telegram.ext import ContextTypes

from handlers.invoice_flow import dispatch_ocr_result
from ocr_engine import InvoiceOCREngine, OCRResult, REQUIRED_FIELDS
from state_manager import UserState, state_manager

logger = logging.getLogger(__name__)


async def process_next_queued_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Pops and starts processing the next media item in queue. Returns True if an item was processed."""
    chat_id = update.effective_chat.id
    active_company = state_manager.get_active_company(chat_id)
    rnc_scope = active_company.rnc if active_company else None

    # Verificar si hay elementos pendientes en la cola
    item = state_manager.pop_next_queued_media(chat_id, rnc=rnc_scope)
    if not item:
        return False

    remaining = state_manager.get_queue_count(chat_id, rnc=rnc_scope)
    media_type = item["media_type"]
    file_name = item.get("file_name") or ("foto" if media_type == "photo" else "factura.pdf")

    target_msg = update.effective_message or (update.callback_query.message if update.callback_query else None)
    rem_info = f" (Quedan {remaining} en cola)" if remaining > 0 else ""
    if target_msg:
        await target_msg.reply_text(
            f"📄 Procesando siguiente {media_type.upper()}: *{file_name}*{rem_info}...",
            parse_mode="Markdown",
        )

    state_manager.set_state(chat_id, UserState.PROCESSING_OCR)

    file_bytes = None
    try:
        telegram_file = await context.bot.get_file(item["file_id"])
        file_bytes = bytes(await telegram_file.download_as_bytearray())
        engine: InvoiceOCREngine = context.bot_data["ocr_engine"]
        buyer_rnc = active_company.rnc if active_company else None

        if media_type == "photo":
            result = await asyncio.to_thread(engine.analyze, file_bytes, buyer_rnc=buyer_rnc)
        else:
            from handlers.pdf_handler import _pdf_to_images
            pages = await asyncio.to_thread(_pdf_to_images, file_bytes)
            if not pages:
                raise ValueError("PDF sin páginas legibles")
            result = await asyncio.to_thread(engine.analyze_pdf_pages, pages, buyer_rnc=buyer_rnc)

    except Exception:
        logger.warning("Error procesando elemento en cola (file_id=%s) para chat_id=%s", item["file_id"], chat_id, exc_info=True)
        if not file_bytes:
            state_manager.clear_session(chat_id)
            if target_msg:
                await target_msg.reply_text(f"⚠️ No pude descargar el archivo *{file_name}*. Continuando con la cola...", parse_mode="Markdown")
            return await process_next_queued_media(update, context)
        result = OCRResult(raw_text="", image_bytes=file_bytes, missing_fields=list(REQUIRED_FIELDS))

    await dispatch_ocr_result(update, context, result)
    return True
