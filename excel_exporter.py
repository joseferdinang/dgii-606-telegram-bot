"""Creation of a clean, paste-ready DGII Form 606 workbook."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

try:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
except ImportError:
    Workbook = Alignment = Border = Font = PatternFill = Side = None
try:
    from openpyxl.worksheet.table import Table, TableStyleInfo
except ImportError:
    Table = TableStyleInfo = None

from fiscal_validators import decimal_text, infer_tipo_id, normalize_amount, normalize_date


SHEET_NAME = "606"
HEADERS = (
    "Líneas",
    "RNC o Cédula",
    "Tipo Id",
    "Tipo de Bienes y Servicios Comprados",
    "NCF",
    "NCF o Documento Modificado",
    "Fecha Comprobante (AAAAMM)",
    "Fecha Comprobante (DD)",
    "Fecha Pago (AAAAMM)",
    "Fecha Pago (DD)",
    "Monto Facturado en Servicios",
    "Monto Facturado en Bienes",
    "Total Monto Facturado",
    "ITBIS Facturado",
    "ITBIS Retenido",
    "ITBIS Sujeto a Proporcionalidad (Art. 349)",
    "ITBIS Llevado al Costo",
    "ITBIS por Adelantar",
    "ITBIS Percibido en Compras",
    "Tipo de Retención en ISR",
    "Monto Retención Renta",
    "ISR Percibido en Compras",
    "Impuesto Selectivo al Consumo",
    "Otros Impuestos/Tasas",
    "Monto Propina Legal",
    "Forma de Pago",
)

MONEY_FIELDS = (
    "monto_servicios", "monto_bienes", "itbis_facturado", "itbis_retenido",
    "itbis_proporcionalidad", "itbis_llevado_costo", "itbis_percibido_compras",
    "isr_retenido", "isr_percibido_compras", "impuesto_selectivo_consumo",
    "otros_impuestos_tasas", "monto_propina_legal",
)


class Excel606Exporter:
    """Exports only data entered by the user; no original DGII macro is altered."""

    def __init__(self, export_directory: Path) -> None:
        self.export_directory = Path(export_directory)
        self.export_directory.mkdir(parents=True, exist_ok=True)

    def export(self, invoices: list[dict], *, chat_id: int, reporting_rnc: str | None = None) -> Path:
        if not invoices:
            raise ValueError("No hay facturas confirmadas para exportar.")

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = SHEET_NAME
        sheet.freeze_panes = "A2"
        sheet.sheet_view.showGridLines = False

        self._write_headers(sheet)
        for line_number, invoice in enumerate(invoices, start=1):
            sheet.append(self._row(line_number, invoice))

        end_row = len(invoices) + 1
        table = Table(displayName="Formato606", ref=f"A1:Z{end_row}")
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2", showFirstColumn=False, showLastColumn=False,
            showRowStripes=True, showColumnStripes=False,
        )
        sheet.add_table(table)
        self._format_data(sheet, end_row)
        self._add_readme(workbook, invoices, reporting_rnc)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"DGII_Formato_606_{chat_id}_{timestamp}_{uuid4().hex[:8]}.xlsx"
        output_path = self.export_directory / filename
        workbook.save(output_path)
        return output_path

    @staticmethod
    def _money(invoice: dict, field: str) -> Decimal:
        return normalize_amount(invoice.get(field), allow_empty=True) or Decimal("0.00")

    def _row(self, line_number: int, invoice: dict) -> list:
        invoice_date = normalize_date(invoice["fecha_comprobante"])
        payment_date = normalize_date(invoice["fecha_pago"]) if invoice.get("fecha_pago") else ""
        services = self._money(invoice, "monto_servicios")
        goods = self._money(invoice, "monto_bienes")
        itbis = self._money(invoice, "itbis_facturado")
        cost_itbis = self._money(invoice, "itbis_llevado_costo")
        return [
            line_number,
            str(invoice["rnc_cedula"]),
            str(invoice.get("tipo_id") or infer_tipo_id(invoice["rnc_cedula"])),
            str(invoice["tipo_bienes_servicios"]),
            str(invoice["ncf"]),
            str(invoice.get("ncf_modificado", "")),
            invoice_date[:6], invoice_date[6:], payment_date[:6], payment_date[6:],
            services, goods, services + goods, itbis,
            self._money(invoice, "itbis_retenido"),
            self._money(invoice, "itbis_proporcionalidad"),
            cost_itbis, itbis - cost_itbis,
            self._money(invoice, "itbis_percibido_compras"),
            str(invoice.get("tipo_retencion_isr", "")),
            self._money(invoice, "isr_retenido"),
            self._money(invoice, "isr_percibido_compras"),
            self._money(invoice, "impuesto_selectivo_consumo"),
            self._money(invoice, "otros_impuestos_tasas"),
            self._money(invoice, "monto_propina_legal"),
            str(invoice["forma_pago"]),
        ]

    @staticmethod
    def _write_headers(sheet) -> None:
        fill = PatternFill("solid", fgColor="1F4E78")
        font = Font(color="FFFFFF", bold=True)
        border = Border(bottom=Side(style="thin", color="D9E2F3"))
        for column, header in enumerate(HEADERS, start=1):
            cell = sheet.cell(row=1, column=column, value=header)
            cell.fill = fill
            cell.font = font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = border
        sheet.row_dimensions[1].height = 46

    @staticmethod
    def _format_data(sheet, end_row: int) -> None:
        widths = {
            "A": 9, "B": 16, "C": 9, "D": 18, "E": 18, "F": 22,
            "G": 22, "H": 20, "I": 18, "J": 16, "K": 24, "L": 22,
            "M": 21, "N": 16, "O": 16, "P": 31, "Q": 21, "R": 19,
            "S": 23, "T": 22, "U": 22, "V": 22, "W": 27, "X": 22,
            "Y": 19, "Z": 16,
        }
        for column, width in widths.items():
            sheet.column_dimensions[column].width = width
        for row in range(2, end_row + 1):
            for column in range(11, 26):
                sheet.cell(row=row, column=column).number_format = "0.00"
            for column in (1, 3, 4, 7, 8, 9, 10, 20, 26):
                sheet.cell(row=row, column=column).number_format = "@"
                sheet.cell(row=row, column=column).alignment = Alignment(horizontal="center")

    @staticmethod
    def _add_readme(workbook: Workbook, invoices: list[dict], reporting_rnc: str | None) -> None:
        sheet = workbook.create_sheet("Instrucciones")
        period = normalize_date(invoices[0]["fecha_comprobante"])[:6]
        rows = (
            ("Formato", "606 - Compras de Bienes y Servicios"),
            ("Período del lote", period),
            ("RNC declarante", reporting_rnc or "Completar en la plantilla oficial"),
            ("Registros", str(len(invoices))),
            ("Uso", "Copie las filas de la hoja 606 en la herramienta oficial de la DGII."),
            ("Nota", "Este archivo no contiene macros. La plantilla oficial genera el TXT de remisión."),
        )
        for row in rows:
            sheet.append(row)
        sheet.column_dimensions["A"].width = 24
        sheet.column_dimensions["B"].width = 105
        for cell in sheet["A"]:
            cell.font = Font(bold=True)
