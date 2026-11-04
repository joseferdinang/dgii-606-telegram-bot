"""Handlers for multi-company registration, authentication (PIN login), and company switching.

Ensures strict confidentiality: users can only see, list, or access the companies
for which they have entered the correct secret PIN. No user can ever see the RNCs
or invoices of other organizations.
"""

from __future__ import annotations

import logging
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from state_manager import UserState, state_manager

logger = logging.getLogger(__name__)


async def cmd_registro(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Registra una nueva empresa y la activa para el usuario actual.

    Uso: /registro <RNC_o_Cédula> <PIN> <Nombre de la Empresa...>
    """
    chat_id = update.effective_chat.id
    args = context.args or []

    if len(args) < 3:
        await update.message.reply_text(
            "🏢 *Registro de Empresa*\n\n"
            "Para registrar una empresa en el sistema, escribe:\n"
            "`/registro <RNC_o_Cédula> <PIN> <Nombre de la Empresa>`\n\n"
            "*Ejemplo:*\n"
            "`/registro 131000001 1234 Constructora Dominicana SRL`\n\n"
            "🔒 *Nota de Seguridad:* El PIN (mínimo 4 caracteres) protege tu empresa. "
            "Nadie podrá ver tu RNC ni tus facturas sin este PIN.",
            parse_mode="Markdown",
        )
        return

    rnc = args[0]
    pin = args[1]
    razon_social = " ".join(args[2:]).strip()

    try:
        await update.message.delete()
    except Exception:
        pass

    try:
        company = state_manager.register_company(
            chat_id=chat_id,
            rnc=rnc,
            razon_social=razon_social,
            pin=pin,
        )
        logger.info("Empresa registrada con éxito: rnc=%s por chat_id=%s", company.rnc, chat_id)
        await update.message.reply_text(
            f"✅ *¡Empresa registrada y activada con éxito!*\n\n"
            f"🏢 *Empresa:* {company.razon_social}\n"
            f"🆔 *RNC:* `{company.rnc}`\n"
            f"🔒 *PIN de acceso configurado correctamente.* *(Guárdalo de forma segura para autorizar a tu equipo)*\n\n"
            "Todas las facturas que envíes ahora se registrarán para esta empresa.\n"
            "Escribe Terminar cuando desees exportar el archivo TXT para la DGII.",
            parse_mode="Markdown",
        )
    except ValueError as exc:
        await update.message.reply_text(f"⚠️ {exc}")


async def cmd_login(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Inicia sesión en una empresa existente mediante su RNC y PIN de acceso.

    Uso: /login <RNC_o_Cédula> <PIN>
    """
    chat_id = update.effective_chat.id
    args = context.args or []

    if len(args) < 2:
        await update.message.reply_text(
            "🔐 *Iniciar Sesión en una Empresa*\n\n"
            "Para identificarte con tu empresa, escribe:\n"
            "`/login <RNC_o_Cédula> <PIN>`\n\n"
            "*Ejemplo:*\n"
            "`/login 131000001 1234`\n\n"
            "Si la empresa es nueva y aún no ha sido registrada, usa `/registro`.",
            parse_mode="Markdown",
        )
        return

    rnc = args[0]
    pin = args[1]

    try:
        await update.message.delete()
    except Exception:
        pass

    try:
        company = state_manager.login_company(chat_id=chat_id, rnc=rnc, pin=pin)
        count = state_manager.batch_count(chat_id, rnc=company.rnc)
        period = state_manager.batch_period(chat_id, rnc=company.rnc) or "sin período"
        logger.info("Inicio de sesión exitoso: rnc=%s para chat_id=%s", company.rnc, chat_id)
        await update.message.reply_text(
            f"✅ *¡Inicio de sesión exitoso!*\n\n"
            f"🏢 *Empresa activa:* {company.razon_social}\n"
            f"🆔 *RNC:* `{company.rnc}`\n"
            f"📋 *Facturas en lote:* {count} (Período: {period})\n\n"
            "Ya puedes enviar fotos o PDFs de tus facturas.",
            parse_mode="Markdown",
        )
    except ValueError as exc:
        await update.message.reply_text(f"❌ {exc}")


async def cmd_empresa(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Muestra la empresa activa o el listado de empresas autorizadas para este usuario."""
    chat_id = update.effective_chat.id
    user_companies = state_manager.get_user_companies(chat_id)

    if not user_companies:
        await update.message.reply_text(
            "⚠️ *No has iniciado sesión en ninguna empresa.*\n\n"
            "Para comenzar a declarar facturas debes identificarte:\n"
            "👉 `/login <RNC> <PIN>`\n\n"
            "O si tu empresa es nueva:\n"
            "👉 `/registro <RNC> <PIN> <Nombre>`",
            parse_mode="Markdown",
        )
        return

    if len(user_companies) == 1:
        comp = user_companies[0]
        count = state_manager.batch_count(chat_id, rnc=comp.rnc)
        period = state_manager.batch_period(chat_id, rnc=comp.rnc) or "sin facturas"
        await update.message.reply_text(
            f"🏢 *Empresa Activa*\n\n"
            f"**Nombre:** {comp.razon_social}\n"
            f"**RNC:** `{comp.rnc}`\n"
            f"**Período actual:** {period}\n"
            f"**Facturas en lote:** {count}\n\n"
            "Para agregar otra empresa a tu cuenta: `/login <RNC> <PIN>`\n"
            "Para cerrar sesión: `/logout`",
            parse_mode="Markdown",
        )
        return

    # Si el usuario maneja múltiples empresas (ej: contador)
    keyboard = []
    for comp in user_companies:
        tag = " ✅ [Activa]" if comp.is_active else ""
        button_text = f"🏢 {comp.razon_social} ({comp.rnc}){tag}"
        keyboard.append([InlineKeyboardButton(button_text, callback_data=f"switch_company:{comp.rnc}")])

    keyboard.append([InlineKeyboardButton("🚪 Cerrar Sesión Actual", callback_data="logout_active")])

    await update.message.reply_text(
        "🏢 *Tus Empresas Autorizadas*\n\n"
        "Toca una empresa para seleccionarla como la empresa activa para tus facturas:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown",
    )


async def handle_company_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gestiona los botones inline para cambiar de empresa o cerrar sesión."""
    query = update.callback_query
    await query.answer()
    chat_id = update.effective_chat.id
    data = query.data or ""

    if data.startswith("switch_company:"):
        session = state_manager.get_session(chat_id)
        if session.state != UserState.IDLE:
            await query.answer("⚠️ Tienes una factura en revisión. Confírmala o descártala con /cancelar antes de cambiar de empresa.", show_alert=True)
            return

        target_rnc = data.split(":", 1)[1]
        try:
            company = state_manager.switch_active_company(chat_id, target_rnc)
            count = state_manager.batch_count(chat_id, rnc=company.rnc)
            period = state_manager.batch_period(chat_id, rnc=company.rnc) or "sin período"
            await query.edit_message_text(
                f"✅ *Empresa activa cambiada a:*\n\n"
                f"🏢 *{company.razon_social}*\n"
                f"🆔 *RNC:* `{company.rnc}`\n"
                f"📋 *Facturas en este lote:* {count} (Período: {period})\n\n"
                "Todas las facturas que envíes ahora pertenecerán a este RNC.",
                parse_mode="Markdown",
            )
        except Exception as exc:
            await query.edit_message_text(f"⚠️ Error al cambiar de empresa: {exc}")

    elif data == "logout_active":
        active = state_manager.get_active_company(chat_id)
        if active:
            state_manager.logout_company(chat_id, active.rnc)
            await query.edit_message_text(
                f"👋 Sesión cerrada para la empresa *{active.razon_social}* (`{active.rnc}`).\n\n"
                "Para volver a ingresar usa `/login <RNC> <PIN>`.",
                parse_mode="Markdown",
            )
        else:
            await query.edit_message_text("No hay ninguna sesión activa actualmente.")


async def cmd_logout(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Cierra la sesión de la empresa activa para este chat."""
    chat_id = update.effective_chat.id
    active = state_manager.get_active_company(chat_id)
    if not active:
        await update.message.reply_text("No tienes ninguna sesión de empresa activa actualmente.")
        return

    state_manager.logout_company(chat_id, active.rnc)
    # Verificar si le queda alguna otra empresa
    remaining = state_manager.get_active_company(chat_id)
    if remaining:
        await update.message.reply_text(
            f"👋 Has cerrado sesión en *{active.razon_social}*.\n"
            f"Tu empresa activa ahora es: *{remaining.razon_social}* (`{remaining.rnc}`).",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(
            f"👋 Has cerrado sesión en *{active.razon_social}* (`{active.rnc}`).\n\n"
            "Para volver a utilizar el bot, inicia sesión con `/login <RNC> <PIN>`.",
            parse_mode="Markdown",
        )
