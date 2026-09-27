"""Capture and apply a USD to DOP exchange rate to the pending invoice."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re

from telegram import Update
from telegram.ext import ContextTypes

from fiscal_validators import decimal_text, normalize_amount
from handlers.confirmation_handler import send_category_selection, proceed_to_next_step
from state_manager import state_manager

TASA_MIN = Decimal("50")
TASA_MAX = Decimal("200")


def _parse_rate(text: str) -> Decimal | None:
    cleaned = re.sub(r"[^\d.,]", "", text)
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    else:
        cleaned = cleaned.replace(",", ".")
    try:
        rate = Decimal(cleaned).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return None
    return rate if TASA_MIN <= rate <= TASA_MAX else None


async def handle_exchange_rate_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rate = _parse_rate(update.message.text)
    if rate is None:
        await update.message.reply_text("La tasa debe estar entre 50.00 y 200.00 RD$/USD. Ejemplo: 61.50.")
        return
    chat_id = update.effective_chat.id
    session = state_manager.get_session(chat_id)
    try:
        total_usd = normalize_amount(session.extracted_data["monto_total"])
        itbis_usd = normalize_amount(session.extracted_data["itbis_facturado"])
    except (KeyError, ValueError):
        await update.message.reply_text("Los montos de la factura no son válidos. Envía la factura nuevamente.")
        return
    state_manager.update_field(chat_id, "tasa_cambio", decimal_text(rate))
    state_manager.update_field(chat_id, "monto_total_usd", decimal_text(total_usd))
    state_manager.update_field(chat_id, "itbis_facturado_usd", decimal_text(itbis_usd))
    state_manager.update_field(chat_id, "monto_total", decimal_text(total_usd * rate))
    state_manager.update_field(chat_id, "itbis_facturado", decimal_text(itbis_usd * rate))
    await update.message.reply_text(f"Tasa registrada: RD$ {decimal_text(rate)} por USD.")
    await proceed_to_next_step(update, context)
