"""Generation of DGII Formato 606 text files (.TXT) for Oficina Virtual (OFV).

Complies with Dominican DGII Norma General 07-18 and official Formato 606 instructions.
Delimited by vertical bars ('|'), encoded in Windows-1252 (ANSI) with CRLF line endings.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional

from fiscal_validators import (
    normalize_identifier,
    normalize_ncf,
    normalize_date,
    normalize_amount,
    infer_tipo_id,
)

MONEY_QUANTUM = Decimal("0.01")
MAX_REGISTROS = 10000


class TipoId(str, Enum):
    RNC = "1"
    CEDULA = "2"


class TipoBienesServicios(str, Enum):
    GASTOS_PERSONAL = "01"
    GASTOS_TRABAJO_SUMINISTRO_SERVICIOS = "02"
    ARRENDAMIENTOS = "03"
    GASTOS_ACTIVOS_FIJOS = "04"
    GASTOS_REPRESENTACION = "05"
    OTRAS_DEDUCCIONES_ADMITIDAS = "06"
    GASTOS_FINANCIEROS = "07"
    GASTOS_EXTRAORDINARIOS = "08"
    COMPRAS_COSTO_VENTA = "09"
    ADQUISICION_ACTIVOS = "10"
    GASTOS_SEGUROS = "11"


class TipoRetencionISR(str, Enum):
    ALQUILERES = "01"
    HONORARIOS_SERVICIOS = "02"
    OTRAS_RENTAS = "03"
    RENTAS_PRESUNTAS = "04"
    INTERESES_PAGADOS_PERSONAS_JURIDICAS = "05"
    INTERESES_PAGADOS_PERSONAS_FISICAS = "06"
    RETENCION_POR_PROVEEDORES_DEL_ESTADO = "07"
    JUEGOS_TELEFONICOS = "08"
    GANANCIAS_CAPITAL = "09"


class FormaPago(str, Enum):
    EFECTIVO = "01"
    CHEQUE_TRANSFERENCIA_DEPOSITO = "02"
    TARJETA_CREDITO_DEBITO = "03"
    COMPRA_A_CREDITO = "04"
    PERMUTA = "05"
    NOTA_DE_CREDITO = "06"
    MIXTO = "07"


def _fmt_money(value: Decimal) -> str:
    return format(value.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP), ".2f")


@dataclass(frozen=True)
class Record606:
    rnc_cedula: str
    tipo_id: TipoId
    tipo_bienes_servicios: TipoBienesServicios
    ncf: str
    ncf_modificado: str
    fecha_comprobante: str
    fecha_pago: str
    monto_servicios: Decimal
    monto_bienes: Decimal
    total_monto_facturado: Decimal
    itbis_facturado: Decimal
    itbis_retenido: Decimal
    itbis_proporcionalidad: Decimal
    itbis_llevado_costo: Decimal
    itbis_por_adelantar: Decimal
    itbis_percibido_compras: Decimal
    tipo_retencion_isr: Optional[TipoRetencionISR]
    isr_retenido: Decimal
    isr_percibido_compras: Decimal
    impuesto_selectivo_consumo: Decimal
    otros_impuestos_tasas: Decimal
    monto_propina_legal: Decimal
    forma_pago: FormaPago

    @classmethod
    def from_dict(cls, data: dict[str, Any], reporting_rnc: Optional[str] = None) -> Record606:
        raw_ncf = normalize_ncf(data["ncf"])
        # B17/E47 (Pagos al Exterior) y B13/E43 (Gastos Menores) son auto-emitidos con el RNC del declarante
        is_self_issued = raw_ncf.startswith(("B17", "E47", "B13", "E43"))

        if is_self_issued:
            if not reporting_rnc:
                raise ValueError(f"Se requiere reporting_rnc para procesar comprobantes tipo {raw_ncf[:3]}.")
            rnc_cedula = normalize_identifier(reporting_rnc)
            tipo_id = TipoId(infer_tipo_id(rnc_cedula))
        else:
            rnc_cedula = normalize_identifier(data["rnc_cedula"])
            tipo_id = TipoId(str(data.get("tipo_id") or infer_tipo_id(rnc_cedula)))

        # Categoría
        cat_str = str(data["tipo_bienes_servicios"]).zfill(2)
        tipo_bs = TipoBienesServicios(cat_str)

        # Forma de pago
        fp_str = str(data["forma_pago"]).zfill(2)
        forma_pago = FormaPago(fp_str)

        # NCF modificado
        ncf_mod = ""
        if data.get("ncf_modificado"):
            ncf_mod = normalize_ncf(data["ncf_modificado"])

        # Fechas
        fecha_comp = normalize_date(data["fecha_comprobante"])
        fecha_pago = normalize_date(data["fecha_pago"]) if data.get("fecha_pago") else ""

        def dec(field: str) -> Decimal:
            val = data.get(field)
            if val in (None, ""):
                return Decimal("0.00")
            return normalize_amount(val, allow_empty=True) or Decimal("0.00")

        servicios = dec("monto_servicios")
        bienes = dec("monto_bienes")
        total_facturado = servicios + bienes

        itbis = dec("itbis_facturado") if not is_self_issued else Decimal("0.00")
        itbis_costo = dec("itbis_llevado_costo") if not is_self_issued else Decimal("0.00")
        itbis_adelantar = itbis - itbis_costo

        itbis_ret = dec("itbis_retenido") if not is_self_issued else Decimal("0.00")
        itbis_prop = dec("itbis_proporcionalidad") if not is_self_issued else Decimal("0.00")
        itbis_perc = dec("itbis_percibido_compras") if not is_self_issued else Decimal("0.00")

        isr_ret = dec("isr_retenido") if not is_self_issued else Decimal("0.00")
        isr_perc = dec("isr_percibido_compras") if not is_self_issued else Decimal("0.00")

        tipo_ret_isr = None
        if not is_self_issued and data.get("tipo_retencion_isr"):
            tipo_ret_isr = TipoRetencionISR(str(data["tipo_retencion_isr"]).zfill(2))

        isc = dec("impuesto_selectivo_consumo") if not is_self_issued else Decimal("0.00")
        otros = dec("otros_impuestos_tasas") if not is_self_issued else Decimal("0.00")
        propina = dec("monto_propina_legal") if not is_self_issued else Decimal("0.00")

        return cls(
            rnc_cedula=rnc_cedula,
            tipo_id=tipo_id,
            tipo_bienes_servicios=tipo_bs,
            ncf=raw_ncf,
            ncf_modificado=ncf_mod,
            fecha_comprobante=fecha_comp,
            fecha_pago=fecha_pago,
            monto_servicios=servicios,
            monto_bienes=bienes,
            total_monto_facturado=total_facturado,
            itbis_facturado=itbis,
            itbis_retenido=itbis_ret,
            itbis_proporcionalidad=itbis_prop,
            itbis_llevado_costo=itbis_costo,
            itbis_por_adelantar=itbis_adelantar,
            itbis_percibido_compras=itbis_perc,
            tipo_retencion_isr=tipo_ret_isr,
            isr_retenido=isr_ret,
            isr_percibido_compras=isr_perc,
            impuesto_selectivo_consumo=isc,
            otros_impuestos_tasas=otros,
            monto_propina_legal=propina,
            forma_pago=forma_pago,
        )

    def to_line(self) -> str:
        """Serializes the record into the 23 pipe-delimited fields."""
        fields = [
            self.rnc_cedula,
            self.tipo_id.value,
            self.tipo_bienes_servicios.value,
            self.ncf,
            self.ncf_modificado,
            self.fecha_comprobante,
            self.fecha_pago,
            _fmt_money(self.monto_servicios),
            _fmt_money(self.monto_bienes),
            _fmt_money(self.total_monto_facturado),
            _fmt_money(self.itbis_facturado),
            _fmt_money(self.itbis_retenido),
            _fmt_money(self.itbis_proporcionalidad),
            _fmt_money(self.itbis_llevado_costo),
            _fmt_money(self.itbis_por_adelantar),
            _fmt_money(self.itbis_percibido_compras),
            self.tipo_retencion_isr.value if self.tipo_retencion_isr else "",
            _fmt_money(self.isr_retenido),
            _fmt_money(self.isr_percibido_compras),
            _fmt_money(self.impuesto_selectivo_consumo),
            _fmt_money(self.otros_impuestos_tasas),
            _fmt_money(self.monto_propina_legal),
            self.forma_pago.value,
        ]
        return "|".join(fields)


def generate_606_txt(
    invoices: List[dict[str, Any]],
    *,
    reporting_rnc: str,
    output_dir: str | Path,
    period: Optional[str] = None,
) -> Path:
    """Generates the DGII_F_606_<RNC>_<AAAAMM>.TXT file ready for DGII Oficina Virtual."""
    if not invoices:
        raise ValueError("No hay facturas para generar el Formato 606.")
    if len(invoices) > MAX_REGISTROS:
        raise ValueError(
            f"El lote contiene {len(invoices)} facturas, excediendo el límite máximo de {MAX_REGISTROS} registros."
        )

    norm_reporting_rnc = normalize_identifier(reporting_rnc)
    first_date = normalize_date(invoices[0]["fecha_comprobante"])
    detected_period = period or first_date[:6]

    records = [Record606.from_dict(inv, reporting_rnc=norm_reporting_rnc) for inv in invoices]

    header = f"606|{norm_reporting_rnc}|{detected_period}|{len(records)}"
    lines = [header] + [rec.to_line() for rec in records]

    # Delimited lines with CRLF as required by DGII
    content = "\r\n".join(lines) + "\r\n"

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"DGII_F_606_{norm_reporting_rnc}_{detected_period}.TXT"

    with open(file_path, "wb") as f:
        f.write(content.encode("windows-1252"))

    return file_path
