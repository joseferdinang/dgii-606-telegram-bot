"""Capture and apply a USD to DOP exchange rate to the pending invoice."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re

from telegram import Update
from telegram.ext import ContextTypes

from fiscal_validators import decimal_text, normalize_amount
from handlers.confirmation_handler import proceed_to_next_step
from state_manager import state_manager

TASA_MIN = Decimal("50")
TASA_MAX = Decimal("200")


def _parse_rate(text: str) -> Decimal | None:
    cleaned = re.sub(r"[^\d.,]", "", str(text or ""))
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    else:
        cleaned = cleaned.replace(",", ".")
    try:
        rate = Decimal(cleaned).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return None
    return rate if TASA_MIN <= rate <= TASA_MAX else None


def apply_usd_exchange_rate(fields: dict, rate: Decimal) -> dict:
    """Convierte todos los montos en USD de la factura a DOP evitando descuadres de redondeo."""
    if fields.get("_usd_converted"):
        return fields

    fields["tasa_cambio"] = decimal_text(rate)
    monetary_keys = (
        "monto_total",
        "itbis_facturado",
        "monto_propina_legal",
        "impuesto_selectivo_consumo",
        "otros_impuestos_tasas",
        "itbis_retenido",
        "isr_retenido",
        "monto_servicios",
        "monto_bienes",
    )
    for key in monetary_keys:
        raw_val = fields.get(key)
        if raw_val not in (None, ""):
            amt_usd = normalize_amount(raw_val, allow_empty=True)
            if amt_usd is not None:
                fields[f"{key}_usd"] = decimal_text(amt_usd)
                fields[key] = decimal_text(amt_usd * rate)

    # Si tanto monto_servicios como monto_bienes estaban definidos, ajustar diferencia de 1 centavo por redondeo
    if fields.get("monto_servicios") is not None and fields.get("monto_bienes") is not None:
        total_dop = normalize_amount(fields.get("monto_total"), allow_empty=True) or Decimal("0.00")
        itbis_dop = normalize_amount(fields.get("itbis_facturado"), allow_empty=True) or Decimal("0.00")
        isc_dop = normalize_amount(fields.get("impuesto_selectivo_consumo"), allow_empty=True) or Decimal("0.00")
        otros_dop = normalize_amount(fields.get("otros_impuestos_tasas"), allow_empty=True) or Decimal("0.00")
        tip_dop = normalize_amount(fields.get("monto_propina_legal"), allow_empty=True) or Decimal("0.00")
        serv_dop = normalize_amount(fields.get("monto_servicios"), allow_empty=True) or Decimal("0.00")
        bienes_dop = normalize_amount(fields.get("monto_bienes"), allow_empty=True) or Decimal("0.00")
        expected_base = total_dop - itbis_dop - isc_dop - otros_dop - tip_dop
        diff = expected_base - (serv_dop + bienes_dop)
        if expected_base >= Decimal("0.00") and abs(diff) <= Decimal("0.05") and diff != Decimal("0.00"):
            if bienes_dop >= serv_dop and bienes_dop + diff >= Decimal("0.00"):
                fields["monto_bienes"] = decimal_text(bienes_dop + diff)
            elif serv_dop + diff >= Decimal("0.00"):
                fields["monto_servicios"] = decimal_text(serv_dop + diff)

    fields["_usd_converted"] = True
    return fields


async def handle_exchange_rate_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rate = _parse_rate(update.message.text)
    if rate is None:
        await update.message.reply_text("La tasa debe estar entre 50.00 y 200.00 RD$/USD. Ejemplo: 61.50.")
        return
    chat_id = update.effective_chat.id
    session = state_manager.get_session(chat_id)
    try:
        normalize_amount(session.extracted_data["monto_total"])
        normalize_amount(session.extracted_data["itbis_facturado"])
    except (KeyError, ValueError):
        await update.message.reply_text("Los montos de la factura no son válidos. Envía la factura nuevamente.")
        return
    apply_usd_exchange_rate(session.extracted_data, rate)
    state_manager.update_session(chat_id, session)
    await update.message.reply_text(f"Tasa registrada: RD$ {decimal_text(rate)} por USD.")
    await proceed_to_next_step(update, context)
