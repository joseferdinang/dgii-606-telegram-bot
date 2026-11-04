"""Telegram confirmation flow for a persistent 606 invoice batch."""

from __future__ import annotations

import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from fiscal_validators import (
    ValidationIssue,
    decimal_text,
    normalize_amount,
    normalize_date,
    normalize_identifier,
    normalize_ncf,
    validate_invoice,
)
from state_manager import DuplicateInvoiceError, UserState, state_manager

logger = logging.getLogger(__name__)

TIPO_BIENES_SERVICIOS = {
    "01": "GASTOS DE PERSONAL",
    "02": "TRABAJOS, SUMINISTROS Y SERVICIOS",
    "03": "ARRENDAMIENTOS",
    "04": "GASTOS DE ACTIVOS FIJOS",
    "05": "GASTOS DE REPRESENTACIÓN",
    "06": "OTRAS DEDUCCIONES ADMITIDAS",
    "07": "GASTOS FINANCIEROS",
    "08": "GASTOS EXTRAORDINARIOS",
    "09": "COMPRAS Y GASTOS DEL COSTO DE VENTA",
    "10": "ADQUISICIONES DE ACTIVOS",
    "11": "GASTOS DE SEGUROS",
}
TIPO_FORMA_PAGO = {
    "01": "EFECTIVO",
    "02": "CHEQUES/TRANSFERENCIAS/DEPÓSITO",
    "03": "TARJETA CRÉDITO/DÉBITO",
    "04": "COMPRA A CRÉDITO",
    "05": "PERMUTA",
    "06": "NOTA DE CRÉDITO",
    "07": "MIXTO",
}
FIELD_LABELS = {
    "rnc_cedula": "RNC/Cédula del proveedor",
    "ncf": "NCF o e-CF",
    "fecha_comprobante": "Fecha del comprobante",
    "monto_total": "Total del comprobante, con impuestos",
    "itbis_facturado": "ITBIS facturado",
    "monto_servicios": "Monto facturado en servicios, sin ITBIS",
    "monto_bienes": "Monto facturado en bienes, sin ITBIS",
    "fecha_pago": "Fecha de pago",
    "ncf_modificado": "NCF o e-CF modificado",
}


def _target_message(update: Update):
    return update.message or update.callback_query.message


def _format_money(value) -> str:
    try:
        return f"RD$ {decimal_text(normalize_amount(value))}"
    except ValueError:
        return str(value)


def _format_preview_text(fields: dict) -> str:
    return "\n".join(
        (
            "Revisa la factura antes de añadirla al lote 606:",
            f"RNC/Cédula: {fields.get('rnc_cedula', '')}",
            f"NCF: {fields.get('ncf', '')}",
            f"Fecha: {fields.get('fecha_comprobante', '')}",
            f"Total comprobante: {_format_money(fields.get('monto_total', ''))}",
            f"Servicios sin ITBIS: {_format_money(fields.get('monto_servicios', '0'))}",
            f"Bienes sin ITBIS: {_format_money(fields.get('monto_bienes', '0'))}",
            f"ITBIS: {_format_money(fields.get('itbis_facturado', ''))}",
            f"Categoría: {fields.get('tipo_bienes_servicios', '')} - "
            f"{TIPO_BIENES_SERVICIOS.get(fields.get('tipo_bienes_servicios', ''), '')}",
            f"Forma de pago: {fields.get('forma_pago', '')} - "
            f"{TIPO_FORMA_PAGO.get(fields.get('forma_pago', ''), '')}",
        )
    )


async def send_preview_message(update: Update, context: ContextTypes.DEFAULT_TYPE, fields: dict) -> None:
    keyboard = InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("Confirmar y añadir al lote", callback_data="confirm_invoice"),
            InlineKeyboardButton("Corregir", callback_data="correct_invoice"),
        ]]
    )
    await _target_message(update).reply_text(_format_preview_text(fields), reply_markup=keyboard)


async def _request_manual_fields(update: Update, chat_id: int, fields: list[str], intro: str) -> None:
    session = state_manager.get_session(chat_id)
    session.state = UserState.WAITING_MANUAL_FIELD
    session.missing_fields = fields
    session.field_being_corrected = fields[0]
    state_manager.update_session(chat_id, session)
    await _target_message(update).reply_text(
        f"{intro}\n\nEnvía {FIELD_LABELS.get(fields[0], fields[0])}:"
    )


async def send_missing_field_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE, missing_fields: list[str]) -> None:
    await _request_manual_fields(
        update,
        update.effective_chat.id,
        missing_fields,
        "No pude confirmar este dato desde la factura.",
    )


def _category_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(f"{code} - {label[:21]}", callback_data=f"cat_{code}")
        for code, label in TIPO_BIENES_SERVICIOS.items()
    ]
    return InlineKeyboardMarkup([buttons[index:index + 2] for index in range(0, len(buttons), 2)])


async def send_category_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state_manager.set_state(update.effective_chat.id, UserState.WAITING_CATEGORY)
    await _target_message(update).reply_text(
        "Selecciona el tipo de bienes y servicios comprados para el Formato 606.",
        reply_markup=_category_keyboard(),
    )


async def handle_category_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = update.effective_chat.id
    session = state_manager.get_session(chat_id)
    if session.state != UserState.WAITING_CATEGORY:
        await query.answer("Esta selección ya no está activa.", show_alert=True)
        return
    code = query.data.removeprefix("cat_")
    state_manager.update_field(chat_id, "tipo_bienes_servicios", code)
    await query.edit_message_text(f"Categoría seleccionada: {code} - {TIPO_BIENES_SERVICIOS[code]}")
    if state_manager.get_session(chat_id).extracted_data.get("forma_pago"):
        await send_amount_allocation(update, context)
    else:
        await send_forma_pago_selection(update, context)


def _payment_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"{code} - {label}", callback_data=f"fp_{code}")]
        for code, label in TIPO_FORMA_PAGO.items()
    ])


async def send_forma_pago_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state_manager.set_state(update.effective_chat.id, UserState.WAITING_FORMA_PAGO)
    await _target_message(update).reply_text(
        "Selecciona la forma de pago reportable para esta compra.",
        reply_markup=_payment_keyboard(),
    )


async def handle_forma_pago_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = update.effective_chat.id
    if state_manager.get_session(chat_id).state != UserState.WAITING_FORMA_PAGO:
        await query.answer("Esta selección ya no está activa.", show_alert=True)
        return
    code = query.data.removeprefix("fp_")
    state_manager.update_field(chat_id, "forma_pago", code)
    await query.edit_message_text(f"Forma de pago seleccionada: {code} - {TIPO_FORMA_PAGO[code]}")
    await send_amount_allocation(update, context)


async def send_amount_allocation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    state_manager.set_state(chat_id, UserState.WAITING_ALLOCATION)
    await _target_message(update).reply_text(
        "Indica dónde reportar la base sin ITBIS. Selecciona Mixto solo si debes dividirla.",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("Todo bienes", callback_data="alloc_goods")],
            [InlineKeyboardButton("Todo servicios", callback_data="alloc_services")],
            [InlineKeyboardButton("Mixto", callback_data="alloc_mixed")],
        ]),
    )


async def handle_allocation_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = update.effective_chat.id
    session = state_manager.get_session(chat_id)
    if session.state != UserState.WAITING_ALLOCATION:
        await query.answer("Esta selección ya no está activa.", show_alert=True)
        return

    if query.data == "alloc_mixed":
        await query.edit_message_text("Se solicitarán ambos importes sin ITBIS.")
        await _request_manual_fields(
            update, chat_id, ["monto_servicios", "monto_bienes"],
            "Para una factura mixta, indica primero el importe de servicios sin ITBIS.",
        )
        return

    data = session.extracted_data
    try:
        base = (
            normalize_amount(data.get("monto_total"))
            - normalize_amount(data.get("itbis_facturado"))
            - (normalize_amount(data.get("impuesto_selectivo_consumo"), allow_empty=True) or 0)
            - (normalize_amount(data.get("otros_impuestos_tasas"), allow_empty=True) or 0)
            - (normalize_amount(data.get("monto_propina_legal"), allow_empty=True) or 0)
        )
    except ValueError:
        await _request_manual_fields(update, chat_id, ["monto_total", "itbis_facturado"], "Los montos requieren corrección.")
        return
    if base < 0:
        await _request_manual_fields(update, chat_id, ["monto_total", "itbis_facturado"], "El ITBIS no puede ser mayor al total del comprobante.")
        return

    from decimal import Decimal
    state_manager.update_field(chat_id, "monto_servicios", decimal_text(base if query.data == "alloc_services" else Decimal("0")))
    state_manager.update_field(chat_id, "monto_bienes", decimal_text(base if query.data == "alloc_goods" else Decimal("0")))
    await query.edit_message_text("Distribución de base registrada.")
    await show_validation_or_preview(update, context)


def _normalise_manual_value(field_name: str, raw_value: str) -> str:
    if field_name == "rnc_cedula":
        return normalize_identifier(raw_value)
    if field_name in {"ncf", "ncf_modificado"}:
        return normalize_ncf(raw_value)
    if field_name in {"fecha_comprobante", "fecha_pago"}:
        return normalize_date(raw_value)
    if field_name in {"monto_total", "itbis_facturado", "monto_servicios", "monto_bienes"}:
        return decimal_text(normalize_amount(raw_value))
    return raw_value.strip()


async def handle_manual_field_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    session = state_manager.get_session(chat_id)
    if session.state != UserState.WAITING_MANUAL_FIELD or not session.field_being_corrected:
        return
    field_name = session.field_being_corrected
    try:
        value = _normalise_manual_value(field_name, update.message.text)
    except ValueError as exc:
        await update.message.reply_text(f"Dato inválido: {exc}\nEnvía nuevamente {FIELD_LABELS.get(field_name, field_name)}:")
        return

    state_manager.update_field(chat_id, field_name, value)
    session = state_manager.get_session(chat_id)
    if session.missing_fields:
        session.field_being_corrected = session.missing_fields[0]
        state_manager.update_session(chat_id, session)
        await update.message.reply_text(f"Ahora envía {FIELD_LABELS.get(session.field_being_corrected, session.field_being_corrected)}:")
        return

    session.field_being_corrected = None
    state_manager.update_session(chat_id, session)
    if field_name == "monto_bienes" and session.extracted_data.get("monto_servicios") is not None:
        await show_validation_or_preview(update, context)
        return
    for allocation_field in ("monto_servicios", "monto_bienes"):
        session.extracted_data.pop(allocation_field, None)
    state_manager.update_session(chat_id, session)
    await proceed_to_next_step(update, context)

async def proceed_to_next_step(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    session = state_manager.get_session(chat_id)
    active_company = state_manager.get_active_company(chat_id)
    comp_rnc = active_company.rnc if active_company else ""
    
    rnc = session.extracted_data.get("rnc_cedula")
    if rnc and not session.extracted_data.get("tipo_bienes_servicios"):
        memory = state_manager.get_supplier_memory(rnc, company_rnc=comp_rnc)
        if memory:
            session.extracted_data["tipo_bienes_servicios"] = memory.get("tipo_bienes_servicios")
            session.extracted_data["forma_pago"] = memory.get("forma_pago")
            state_manager.update_session(chat_id, session)
            await _target_message(update).reply_text("Datos de categoría y forma de pago recuperados de factura anterior.")

    if session.extracted_data.get("tipo_bienes_servicios"):
        if session.extracted_data.get("forma_pago"):
            has_bienes = session.extracted_data.get("monto_bienes") is not None
            has_servicios = session.extracted_data.get("monto_servicios") is not None
            if has_bienes and has_servicios:
                await show_validation_or_preview(update, context)
            else:
                await send_amount_allocation(update, context)
        else:
            await send_forma_pago_selection(update, context)
    else:
        await send_category_selection(update, context)


async def show_validation_or_preview(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    session = state_manager.get_session(chat_id)
    issues = validate_invoice(session.extracted_data)
    if not issues and state_manager.has_invoice(chat_id, session.extracted_data):
        issues = [ValidationIssue("ncf", "Ese NCF ya se encuentra en el lote actual.")]
    if issues:
        fields = _correction_fields(issues)
        await _request_manual_fields(update, chat_id, fields, "La validación fiscal requiere una corrección.")
        return
    state_manager.set_state(chat_id, UserState.WAITING_CONFIRMATION)
    await send_preview_message(update, context, session.extracted_data)


def _correction_fields(issues: list[ValidationIssue]) -> list[str]:
    supported = [issue.field for issue in issues if issue.field in FIELD_LABELS]
    return list(dict.fromkeys(supported)) or ["monto_total"]


async def handle_confirmation_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = update.effective_chat.id
    session = state_manager.get_session(chat_id)

    if query.data == "correct_invoice":
        if session.state != UserState.WAITING_CONFIRMATION:
            await query.answer("Esta confirmación ya no está activa.", show_alert=True)
            return
        for field in ("monto_servicios", "monto_bienes"):
            session.extracted_data.pop(field, None)
        session.extracted_data["_was_corrected"] = True
        state_manager.update_session(chat_id, session)
        await query.edit_message_text("Indica los datos corregidos uno por uno.")
        await _request_manual_fields(
            update, chat_id,
            ["rnc_cedula", "ncf", "fecha_comprobante", "monto_total", "itbis_facturado"],
            "Corrección de factura.",
        )
        return

    if session.state != UserState.WAITING_CONFIRMATION:
        await query.answer("La confirmación ya no está activa.", show_alert=True)
        return
    issues = validate_invoice(session.extracted_data)
    if issues or state_manager.has_invoice(chat_id, session.extracted_data):
        await query.edit_message_text("La factura ya no cumple las validaciones. Corrígela antes de confirmar.")
        if not issues:
            issues = [ValidationIssue("ncf", "Ese NCF ya se encuentra en el lote actual.")]
        await _request_manual_fields(update, chat_id, _correction_fields(issues), "Corrección requerida.")
        return
    try:
        raw_text = session.extracted_data.pop("_raw_text", None)
        was_corrected = session.extracted_data.pop("_was_corrected", False)

        active_company = state_manager.get_active_company(chat_id)
        comp_rnc = active_company.rnc if active_company else None

        state_manager.add_confirmed_invoice(chat_id, session.extracted_data, rnc=comp_rnc)
        
        # Guardar memoria del proveedor (RNC -> Categoria y Forma de Pago) aislada por empresa
        rnc = session.extracted_data.get("rnc_cedula")
        cat = session.extracted_data.get("tipo_bienes_servicios")
        fp = session.extracted_data.get("forma_pago")
        if rnc and cat and fp:
            state_manager.save_supplier_memory(rnc, cat, fp, company_rnc=comp_rnc or "")

        # Data Flywheel: Registrar muestra confirmada para aprendizaje continuo y re-entrenamiento
        try:
            from data_flywheel import record_training_sample, retrain_model_from_corpus
            from pathlib import Path
            import asyncio

            record_training_sample(
                session.extracted_data,
                chat_id=chat_id,
                was_corrected=was_corrected,
                raw_text=raw_text,
            )

            # Re-entrenamiento automático continuo en segundo plano en cada factura subida
            ml_model_path = Path(__file__).parent.parent / "ml" / "models" / "field_classifier.joblib"
            async def _bg_retrain():
                try:
                    retrain_res = await asyncio.to_thread(retrain_model_from_corpus, ml_model_path, min_samples=1)
                    if retrain_res.get("status") == "success":
                        ml_ext = context.bot_data.get("ml_extractor") if context and hasattr(context, "bot_data") else None
                        if ml_ext:
                            ml_ext.reload(ml_model_path)
                        logger.info("Modelo ML re-entrenado y recargado con éxito en memoria.")
                except Exception:
                    logger.debug("Error en re-entrenamiento ML de fondo", exc_info=True)

            asyncio.create_task(_bg_retrain())
        except Exception:
            logger.debug("Failed recording sample or triggering retraining", exc_info=True)
            
    except DuplicateInvoiceError:
        await query.edit_message_text("Ese NCF ya se encuentra en el lote actual.")
        await _request_manual_fields(update, chat_id, ["ncf"], "Indica el NCF correcto.")
        return
    except Exception:
        logger.exception("No fue posible persistir la factura confirmada", extra={"chat_id": chat_id})
        await query.edit_message_text("No pude guardar la factura. El lote no fue modificado; intenta confirmar de nuevo.")
        return

    count = state_manager.batch_count(chat_id, rnc=comp_rnc)
    state_manager.clear_session(chat_id)

    # Procesar automáticamente la siguiente factura en cola si existe
    from handlers.queue_handler import process_next_queued_media
    has_next = await process_next_queued_media(update, context)

    if not has_next:
        await query.edit_message_text(
            f"✅ Factura añadida al lote (Lote: {count} facturas).\n"
            f"🧠 El modelo ML local ha aprendido de esta factura para futuras extracciones.\n\n"
            "Envía otra factura o escribe Terminar para exportar."
        )
    else:
        await query.edit_message_text(
            f"✅ Factura añadida al lote (Lote: {count} facturas).\n"
            f"🧠 Modelo ML actualizado. Pasando a la siguiente factura en cola..."
        )
