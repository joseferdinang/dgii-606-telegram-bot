"""Entrypoint for the persistent Telegram to DGII 606 batch bot."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from telegram import InputFile, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import Settings, load_settings
from dgii_txt import generate_606_txt
from excel_exporter import Excel606Exporter
from fiscal_validators import validate_invoice
from handlers.company_handler import (
    cmd_empresa,
    cmd_login,
    cmd_logout,
    cmd_registro,
    handle_company_callback,
)
from handlers.confirmation_handler import (
    handle_allocation_callback,
    handle_category_callback,
    handle_confirmation_callback,
    handle_forma_pago_callback,
    handle_manual_field_input,
)
from handlers.exchange_rate_handler import handle_exchange_rate_input
from handlers.pdf_handler import handle_invoice_pdf
from handlers.photo_handler import handle_invoice_photo
from llm_extractor import LLMFieldExtractor
from ml_extractor import MLFieldExtractor
from ocr_engine import InvoiceOCREngine
from state_manager import UserState, state_manager

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    state_manager.clear_session(chat_id)
    active = state_manager.get_active_company(chat_id)
    if active:
        count = state_manager.batch_count(chat_id, rnc=active.rnc)
        period = state_manager.batch_period(chat_id, rnc=active.rnc) or "sin facturas"
        await update.message.reply_text(
            f"👋 ¡Hola! Bienvenido de nuevo.\n\n"
            f"🏢 *Empresa activa:* {active.razon_social}\n"
            f"🆔 *RNC:* `{active.rnc}`\n"
            f"📋 *Facturas en lote actual:* {count} (Período: {period})\n\n"
            "Envía fotos o PDFs de tus facturas para agregarlas al 606.\n"
            "Escribe *Terminar* cuando quieras recibir el archivo TXT para la DGII.\n\n"
            "💡 *Comandos disponibles:*\n"
            "/empresa - Ver o cambiar tu empresa activa\n"
            "/logout - Cerrar sesión en esta empresa\n"
            "/estado - Ver estado del lote actual\n"
            "/cancelar - Descartar factura en proceso",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(
            "👋 ¡Bienvenido al Bot DGII 606!\n\n"
            "🔒 *Por motivos de privacidad y confidencialidad, cada empresa tiene su propio PIN secreto.*\n"
            "Nadie más puede ver tu RNC ni tus facturas.\n\n"
            "Para comenzar a declarar facturas debes identificarte:\n\n"
            "👉 `/login <RNC> <PIN>` (Si ya está registrada)\n"
            "*Ejemplo:* `/login 131000001 1234`\n\n"
            "👉 `/registro <RNC> <PIN> <Nombre>` (Si es una empresa nueva)\n"
            "*Ejemplo:* `/registro 131000001 1234 Constructora Dominicana SRL`\n\n"
            "Una vez iniciada sesión, podrás subir tus fotos o PDFs.",
            parse_mode="Markdown",
        )


async def cmd_cancelar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state_manager.clear_session(update.effective_chat.id)
    await update.message.reply_text("La factura en proceso fue descartada. Las facturas ya confirmadas siguen en el lote.")


async def cmd_estado(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    active = state_manager.get_active_company(chat_id)
    rnc_scope = active.rnc if active else None
    count = state_manager.batch_count(chat_id, rnc=rnc_scope)
    period = state_manager.batch_period(chat_id, rnc=rnc_scope) or "sin período"
    pending = state_manager.get_session(chat_id).state != UserState.IDLE
    company_str = f"🏢 Empresa activa: {active.razon_social} (`{active.rnc}`)\n" if active else ""
    await update.message.reply_text(
        f"{company_str}📋 Lote actual: {count} factura(s). Período: {period}. Factura pendiente: {'sí' if pending else 'no'}."
    )


async def cmd_descartar_lote(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    active = state_manager.get_active_company(chat_id)
    rnc_scope = active.rnc if active else None
    state_manager.clear_batch_and_session(chat_id, rnc=rnc_scope)
    await update.message.reply_text("El lote actual y la factura pendiente fueron descartados.")


async def cmd_terminar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    pending = state_manager.get_session(chat_id)
    if pending.state != UserState.IDLE:
        await update.message.reply_text(
            "Hay una factura sin confirmar. Confírmala o usa /cancelar antes de exportar el lote."
        )
        return

    active = state_manager.get_active_company(chat_id)
    rnc_scope = active.rnc if active else None
    invoices = state_manager.get_batch(chat_id, rnc=rnc_scope)
    if not invoices:
        await update.message.reply_text("No hay facturas confirmadas en el lote actual.")
        return
    invalid = [
        (index, issue.message)
        for index, invoice in enumerate(invoices, start=1)
        for issue in validate_invoice(invoice)
    ]
    if invalid:
        await update.message.reply_text(
            f"El lote no puede exportarse: la factura {invalid[0][0]} requiere corrección ({invalid[0][1]})."
        )
        return

    settings: Settings = context.bot_data["settings"]
    # Priorizar el RNC de la empresa activa del usuario autenticado
    reporting_rnc = active.rnc if active else settings.reporting_rnc
    company_name = active.razon_social if active else "Empresa"
    if not reporting_rnc:
        await update.message.reply_text(
            "No se ha identificado el RNC declarante.\n"
            "Por favor inicia sesión con `/login <RNC> <PIN>` para generar el archivo TXT oficial de la DGII."
        )
        return

    output_path: Path | None = None
    delivered = False
    try:
        output_path = await asyncio.to_thread(
            generate_606_txt,
            invoices,
            reporting_rnc=reporting_rnc,
            output_dir=settings.export_directory,
        )
        with output_path.open("rb") as txt_file:
            await update.message.reply_document(
                document=InputFile(txt_file, filename=output_path.name),
                caption=(
                    f"📄 Archivo TXT del Formato 606 generado con {len(invoices)} factura(s).\n"
                    f"🏢 {company_name} | RNC: {reporting_rnc}\n"
                    f"📅 Período: {output_path.stem.split('_')[-1]}\n\n"
                    "Listo para importar directamente en la Oficina Virtual (OFV) de la DGII."
                ),
            )
        delivered = True
    except Exception:
        logger.exception("No fue posible exportar el lote en TXT", extra={"chat_id": chat_id})
        await update.message.reply_text("No pude generar o enviar el archivo TXT 606. El lote se conserva; intenta Terminar nuevamente.")
    finally:
        if output_path:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                logger.exception("No fue posible eliminar el archivo TXT temporal", extra={"path": str(output_path)})
    if delivered:
        state_manager.clear_batch_and_session(chat_id, rnc=rnc_scope)


async def route_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text.strip().casefold()
    if text == "terminar":
        await cmd_terminar(update, context)
        return
    chat_id = update.effective_chat.id
    state = state_manager.get_session(chat_id).state
    if state == UserState.WAITING_EXCHANGE_RATE:
        await handle_exchange_rate_input(update, context)
    elif state == UserState.WAITING_MANUAL_FIELD:
        await handle_manual_field_input(update, context)
    else:
        active = state_manager.get_active_company(chat_id)
        if not active:
            await update.message.reply_text(
                "⚠️ *No has iniciado sesión en ninguna empresa.*\n\n"
                "Usa `/login <RNC> <PIN>` para identificarte o `/registro` para dar de alta una nueva empresa.",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(
                f"Envía una foto o PDF para la empresa *{active.razon_social}*, o escribe Terminar para exportar el archivo TXT.",
                parse_mode="Markdown",
            )


def build_application() -> Application:
    settings = load_settings()
    application = ApplicationBuilder().token(settings.telegram_bot_token).build()
    application.bot_data["settings"] = settings
    application.bot_data["ocr_engine"] = InvoiceOCREngine(
        credentials_path=settings.google_vision_credentials_path,
    )
    application.bot_data["excel_exporter"] = Excel606Exporter(settings.export_directory)
    ml_model_path = Path(__file__).parent / "ml" / "models" / "field_classifier.joblib"
    application.bot_data["ml_extractor"] = MLFieldExtractor(ml_model_path)
    application.bot_data["llm_extractor"] = LLMFieldExtractor(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
    )

    # Comandos generales
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("terminar", cmd_terminar))
    application.add_handler(CommandHandler("cancelar", cmd_cancelar))
    application.add_handler(CommandHandler("estado", cmd_estado))
    application.add_handler(CommandHandler("descartar_lote", cmd_descartar_lote))

    # Comandos de autenticación y multi-empresa
    application.add_handler(CommandHandler("registro", cmd_registro))
    application.add_handler(CommandHandler("login", cmd_login))
    application.add_handler(CommandHandler("empresa", cmd_empresa))
    application.add_handler(CommandHandler("empresas", cmd_empresa))
    application.add_handler(CommandHandler("rnc", cmd_empresa))
    application.add_handler(CommandHandler("logout", cmd_logout))

    # Handlers multimedia
    application.add_handler(MessageHandler(filters.PHOTO, handle_invoice_photo))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_invoice_pdf))

    # Callbacks interactivos
    application.add_handler(CallbackQueryHandler(handle_company_callback, pattern="^(switch_company:.+|logout_active)$"))
    application.add_handler(CallbackQueryHandler(handle_confirmation_callback, pattern="^(confirm_invoice|correct_invoice)$"))
    application.add_handler(CallbackQueryHandler(handle_category_callback, pattern="^cat_\\d{2}$"))
    application.add_handler(CallbackQueryHandler(handle_forma_pago_callback, pattern="^fp_\\d{2}$"))
    application.add_handler(CallbackQueryHandler(handle_allocation_callback, pattern="^alloc_(goods|services|mixed)$"))

    # Mensajes de texto libres
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, route_text_message))
    return application


async def _run_bot() -> None:
    application = build_application()
    logger.info("Bot 606 started")
    async with application:
        await application.start()
        await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        stop_event = asyncio.Event()
        try:
            await stop_event.wait()
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            await application.updater.stop()
            await application.stop()


def main() -> None:
    asyncio.run(_run_bot())


if __name__ == "__main__":
    main()
