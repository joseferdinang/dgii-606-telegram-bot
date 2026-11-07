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
            "✨ *¡Puedes enviar varias fotos o archivos al mismo tiempo!* Se añadirán a la cola y se procesarán en orden.\n\n"
            "Escribe *Terminar* cuando quieras recibir el archivo TXT para la DGII.\n\n"
            "💡 *Comandos disponibles:*\n"
            "/resumen - Ver resumen y facturas del lote actual\n"
            "/eliminar - Eliminar una factura por NCF\n"
            "/empresa - Ver o cambiar tu empresa activa\n"
            "/logout - Cerrar sesión en esta empresa\n"
            "/estado - Ver estado del lote y cola actual\n"
            "/cola - Ver facturas pendientes en cola\n"
            "/cancelar_cola - Limpiar la cola de espera\n"
            "/cancelar - Descartar factura en proceso\n"
            "/reentrenar - Re-entrenar modelo ML con facturas reales",
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
    chat_id = update.effective_chat.id
    state_manager.clear_session(chat_id)
    from handlers.queue_handler import process_next_queued_media
    has_next = await process_next_queued_media(update, context)
    if not has_next:
        await update.message.reply_text("La factura en proceso fue descartada. Las facturas ya confirmadas siguen en el lote.")


async def cmd_estado(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    active = state_manager.get_active_company(chat_id)
    rnc_scope = active.rnc if active else None
    count = state_manager.batch_count(chat_id, rnc=rnc_scope)
    period = state_manager.batch_period(chat_id, rnc=rnc_scope) or "sin período"
    pending = state_manager.get_session(chat_id).state != UserState.IDLE
    queue_count = state_manager.get_queue_count(chat_id, rnc=rnc_scope)
    company_str = f"🏢 Empresa activa: {active.razon_social} (`{active.rnc}`)\n" if active else ""
    queue_str = f" Facturas en cola: {queue_count}." if queue_count > 0 else ""
    await update.message.reply_text(
        f"{company_str}📋 Lote actual: {count} factura(s). Período: {period}. Factura en revisión: {'sí' if pending else 'no'}.{queue_str}"
    )


async def cmd_cola(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    active = state_manager.get_active_company(chat_id)
    rnc_scope = active.rnc if active else None
    count = state_manager.get_queue_count(chat_id, rnc=rnc_scope)
    if count == 0:
        await update.message.reply_text("📭 No hay facturas en cola de espera.")
    else:
        await update.message.reply_text(f"📥 Tienes {count} factura(s) en cola esperando a ser revisadas.")


async def cmd_resumen(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Muestra un resumen financiero completo y el listado de facturas en el lote actual."""
    chat_id = update.effective_chat.id
    active = state_manager.get_active_company(chat_id)
    rnc_scope = active.rnc if active else None

    summary = state_manager.get_batch_summary(chat_id, rnc=rnc_scope)
    queue_count = state_manager.get_queue_count(chat_id, rnc=rnc_scope)
    pending_sess = state_manager.get_session(chat_id)
    has_pending = pending_sess.state != UserState.IDLE

    if summary["count"] == 0:
        company_info = f" para *{active.razon_social}* (`{active.rnc}`)" if active else ""
        queue_info = f"\n📥 Tienes {queue_count} factura(s) en cola de espera." if queue_count > 0 else ""
        pending_info = "\n⏳ Tienes una factura pendiente de revisión." if has_pending else ""
        await update.message.reply_text(
            f"📋 *Lote 606 vacío*{company_info}.\n\n"
            f"No hay facturas confirmadas en el lote actual.{queue_info}{pending_info}\n"
            "Envía fotos o archivos PDF para comenzar a agregar comprobantes.",
            parse_mode="Markdown",
        )
        return

    from fiscal_validators import decimal_text

    company_title = f"🏢 *Empresa:* {active.razon_social} (`{active.rnc}`)\n" if active else ""
    period_str = summary["period"] or "Sin período"

    header = (
        f"📊 *Resumen de Facturas del Lote 606*\n"
        f"{company_title}"
        f"📅 *Período:* `{period_str}`\n"
        f"📋 *Total Comprobantes:* {summary['count']}\n\n"
        f"💰 *Totales Acumulados:*\n"
        f"• *Total Facturado:* RD$ {decimal_text(summary['total_monto'])}\n"
        f"• *ITBIS Facturado:* RD$ {decimal_text(summary['total_itbis'])}\n"
        f"• *Bienes (sin ITBIS):* RD$ {decimal_text(summary['total_bienes'])}\n"
        f"• *Servicios (sin ITBIS):* RD$ {decimal_text(summary['total_servicios'])}\n\n"
        f"📑 *Detalle de Comprobantes:*\n"
    )

    items_text = []
    display_items = summary["items"][:25]
    for it in display_items:
        it_fecha = it["fecha"]
        fecha_fmt = f"{it_fecha[6:8]}/{it_fecha[4:6]}/{it_fecha[:4]}" if len(it_fecha) == 8 else it_fecha
        items_text.append(
            f"`{it['index']}.` *{it['ncf']}* | RNC: `{it['rnc_cedula']}` | {fecha_fmt}\n"
            f"   Total: RD$ {decimal_text(it['monto_total'])} | ITBIS: RD$ {decimal_text(it['itbis'])} | Cat: `{it['categoria']}`"
        )

    if len(summary["items"]) > 25:
        items_text.append(f"\n_... y {len(summary['items']) - 25} comprobante(s) adicional(es)_")

    footer = "\n\n💡 *Comandos útiles:*\n• Escribe *Terminar* o /terminar para exportar el TXT de la DGII.\n• Usa `/eliminar <NCF>` para retirar una factura del lote."
    if queue_count > 0:
        footer = f"\n\n📥 *En cola de espera:* {queue_count} factura(s).\n" + footer

    await update.message.reply_text(header + "\n".join(items_text) + footer, parse_mode="Markdown")


async def cmd_eliminar_factura(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Elimina una factura específica del lote actual mediante su NCF."""
    chat_id = update.effective_chat.id
    args = context.args or []
    if not args:
        await update.message.reply_text(
            "🗑️ *Eliminar factura del lote actual*\n\n"
            "Uso: `/eliminar <NCF>`\n"
            "*Ejemplo:* `/eliminar B0100000001`",
            parse_mode="Markdown",
        )
        return

    ncf_to_delete = args[0].strip().upper()
    active = state_manager.get_active_company(chat_id)
    rnc_scope = active.rnc if active else None

    deleted = state_manager.delete_invoice_from_batch(chat_id, ncf_to_delete, rnc=rnc_scope)
    if deleted:
        remaining = state_manager.batch_count(chat_id, rnc=rnc_scope)
        await update.message.reply_text(
            f"✅ Factura con NCF `{ncf_to_delete}` eliminada del lote.\n"
            f"📋 Facturas restantes en el lote: {remaining}",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(
            f"⚠️ No se encontró ninguna factura con el NCF `{ncf_to_delete}` en el lote actual.",
            parse_mode="Markdown",
        )


async def cmd_cancelar_cola(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    active = state_manager.get_active_company(chat_id)
    rnc_scope = active.rnc if active else None
    removed = state_manager.clear_media_queue(chat_id, rnc=rnc_scope)
    await update.message.reply_text(f"🗑️ Se han descartado {removed} factura(s) de la cola de espera.")


async def cmd_descartar_lote(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    active = state_manager.get_active_company(chat_id)
    rnc_scope = active.rnc if active else None
    state_manager.clear_batch_and_session(chat_id, rnc=rnc_scope)
    state_manager.clear_media_queue(chat_id, rnc=rnc_scope)
    await update.message.reply_text("El lote actual, la factura pendiente y la cola fueron descartados.")


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
    queue_count = state_manager.get_queue_count(chat_id, rnc=rnc_scope)
    if queue_count > 0:
        await update.message.reply_text(
            f"⚠️ Tienes {queue_count} factura(s) pendiente(s) en la cola de revisión.\n"
            "Confírmalas o usa /cancelar_cola para descartarlas antes de exportar el lote."
        )
        return

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

        # Archivar lote en historial de auditoría
        period_str = output_path.stem.split('_')[-1]
        state_manager.archive_batch(chat_id, reporting_rnc, period_str)

        # Generar y enviar respaldo en Excel si el exportador está disponible
        excel_path: Path | None = None
        try:
            excel_exporter: Excel606Exporter = context.bot_data.get("excel_exporter")
            if excel_exporter:
                excel_path = await asyncio.to_thread(
                    excel_exporter.export,
                    invoices,
                    chat_id=chat_id,
                    reporting_rnc=reporting_rnc,
                )
                with excel_path.open("rb") as excel_file:
                    await update.message.reply_document(
                        document=InputFile(excel_file, filename=excel_path.name),
                        caption="📊 Respaldo en Excel Formato 606 con el desglose de los comprobantes.",
                    )
        except Exception:
            logger.warning("No fue posible generar exportación Excel de respaldo", exc_info=True)
        finally:
            if excel_path:
                try:
                    excel_path.unlink(missing_ok=True)
                except OSError:
                    pass
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



async def cmd_reentrenar_ml(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from data_flywheel import get_sample_count, retrain_model_from_corpus
    count = get_sample_count()
    if count == 0:
        await update.message.reply_text(
            "🧠 *Aprendizaje Continuo ML*\n\n"
            "Aún no hay facturas confirmadas en el corpus de aprendizaje.\n"
            "A medida que subas y confirmes facturas (o corrijas datos), el modelo irá acumulando "
            "conocimiento específico de tus facturas dominicanas.",
            parse_mode="Markdown",
        )
        return

    await update.message.reply_text(
        f"🧠 Re-entrenando el modelo local con las {count} facturas confirmadas en el corpus...\n"
        "Esto puede tardar unos segundos.",
        parse_mode="Markdown",
    )
    ml_model_path = Path(__file__).parent / "ml" / "models" / "field_classifier.joblib"
    res = await asyncio.to_thread(retrain_model_from_corpus, ml_model_path)
    if res.get("status") == "success":
        ml_ext = context.bot_data.get("ml_extractor")
        if ml_ext:
            ml_ext.reload(ml_model_path)
        await update.message.reply_text(
            f"✅ *¡Modelo re-entrenado y recargado con éxito!*\n\n"
            f"📊 Muestras reales incorporadas: {res.get('real_samples')}\n"
            f"🎯 Tokens procesados: {res.get('tokens')}\n\n"
            "El extractor local ahora reconoce mejor los patrones de tus facturas.",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(f"⚠️ No fue posible re-entrenar: {res.get('message', 'muestras insuficientes')}")

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
    application.add_handler(CommandHandler("exportar", cmd_terminar))
    application.add_handler(CommandHandler("cancelar", cmd_cancelar))
    application.add_handler(CommandHandler("estado", cmd_estado))
    application.add_handler(CommandHandler("resumen", cmd_resumen))
    application.add_handler(CommandHandler("facturas", cmd_resumen))
    application.add_handler(CommandHandler("eliminar", cmd_eliminar_factura))
    application.add_handler(CommandHandler("borrar", cmd_eliminar_factura))
    application.add_handler(CommandHandler("descartar_lote", cmd_descartar_lote))
    application.add_handler(CommandHandler("reentrenar", cmd_reentrenar_ml))
    application.add_handler(CommandHandler("reentrenar_ml", cmd_reentrenar_ml))
    application.add_handler(CommandHandler("cola", cmd_cola))
    application.add_handler(CommandHandler("cancelar_cola", cmd_cancelar_cola))

    # Comandos de autenticación y multi-empresa
    application.add_handler(CommandHandler("registro", cmd_registro))
    application.add_handler(CommandHandler("login", cmd_login))
    application.add_handler(CommandHandler("empresa", cmd_empresa))
    application.add_handler(CommandHandler("empresas", cmd_empresa))
    application.add_handler(CommandHandler("rnc", cmd_empresa))
    application.add_handler(CommandHandler("logout", cmd_logout))

    # Handlers multimedia (incluyendo imágenes enviadas sin compresión como documento)
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_invoice_photo))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_invoice_pdf))

    # Callbacks interactivos
    application.add_handler(CallbackQueryHandler(handle_company_callback, pattern="^(switch_company:.+|logout_active)$"))
    application.add_handler(
        CallbackQueryHandler(
            handle_confirmation_callback,
            pattern="^(confirm_invoice|correct_invoice|realloc_invoice|back_to_preview|correct_field:.+)$",
        )
    )
    application.add_handler(CallbackQueryHandler(handle_category_callback, pattern="^cat_\\d{2}$"))
    application.add_handler(CallbackQueryHandler(handle_forma_pago_callback, pattern="^fp_\\d{2}$"))
    application.add_handler(CallbackQueryHandler(handle_allocation_callback, pattern="^alloc_(goods|services|mixed)$"))

    # Mensajes de texto libres
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, route_text_message))

    # Manejador global de excepciones
    application.add_error_handler(global_error_handler)
    return application


async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Excepción no controlada procesando update: %s", context.error, exc_info=context.error)
    if isinstance(update, Update) and update.effective_chat:
        chat_id = update.effective_chat.id
        sess = state_manager.get_session(chat_id)
        if sess.state == UserState.PROCESSING_OCR:
            state_manager.clear_session(chat_id)
        target = update.effective_message or (update.callback_query.message if update.callback_query else None)
        if target:
            try:
                await target.reply_text(
                    "⚠️ Ocurrió un error inesperado al procesar la solicitud. Tu sesión ha sido restablecida. "
                    "Por favor intenta de nuevo."
                )
            except Exception:
                pass


async def _run_bot() -> None:
    stale_count = state_manager.reset_stale_processing_sessions()
    if stale_count > 0:
        logger.info("Restablecidas %d sesiones colgadas al inicio", stale_count)

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
