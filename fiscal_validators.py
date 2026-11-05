"""Fiscal validation and normalisation rules for Dominican DGII Form 606.

The module deliberately validates the structure, tax rules (Normas Generales
06-18, 07-18, 05-19, Ley 253-12) and arithmetic that can be verified locally.
Includes check-digit verification algorithms (Modulo 11 for RNC and Luhn Modulo 10
for Cédula based on ``python-stdnum`` ``stdnum.do``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import os
import re
from typing import Any


MONEY_QUANTUM = Decimal("0.01")
MONEY_TOLERANCE = Decimal("0.01")
ITBIS_ROUNDING_TOLERANCE = Decimal("0.05")
MAX_AMOUNT = Decimal("999999999999.99")

RNC_PATTERN = re.compile(r"^\d{9}$")
CEDULA_PATTERN = re.compile(r"^\d{11}$")

# Tipos autorizados exclusivamente para compras y gastos en el Formato 606
NCF_606_PATTERN = re.compile(r"^B(01|03|04|11|13|14|17)\d{8}$")
ECF_606_PATTERN = re.compile(r"^E(31|33|34|41|43|44|47)\d{10}$")

# Todos los tipos de NCF existentes en la DGII (para detectar y explicar rechazos como B02/E32)
ALL_NCF_PATTERN = re.compile(r"^B(01|02|03|04|11|12|13|14|15|16|17)\d{8}$")
ALL_ECF_PATTERN = re.compile(r"^E(31|32|33|34|41|43|44|45|46|47)\d{10}$")

DATE_PATTERN = re.compile(r"^\d{8}$")

VALID_CATEGORIES = {
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"
}
VALID_PAYMENT_METHODS = {"01", "02", "03", "04", "05", "06", "07"}
VALID_ISR_RETENTION_TYPES = {"01", "02", "03", "04", "05", "06", "07", "08", "09"}

NOTE_TYPES = {"03", "04", "33", "34"}
CONSUMO_TYPES = {"02", "32"}
NON_606_SALES_TYPES = {"12", "15", "16", "45", "46"}
INFORMAL_PURCHASE_TYPES = {"11", "41"}
MINOR_EXPENSE_TYPES = {"13", "43"}
SPECIAL_REGIME_TYPES = {"14", "44"}
FOREIGN_PAYMENT_TYPES = {"17", "47"}

# Retenciones ISR permitidas a Personas Jurídicas (RNC 9 dígitos) en compras locales
LEGAL_ENTITY_ALLOWED_ISR_TYPES = {"05", "07"}


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    message: str


def _clean_digits(value: Any) -> str:
    return re.sub(r"\D", "", str(value or ""))


def calc_rnc_check_digit(number_8: str) -> str:
    """Calculate the DGII Modulo 11 check digit for the first 8 digits of an RNC."""
    weights = (7, 9, 8, 6, 5, 4, 3, 2)
    check = sum(w * int(n) for w, n in zip(weights, number_8[:8])) % 11
    return str((10 - check) % 9 + 1)


# Excepciones históricas de RNC registradas en la DGII (documentadas en python-stdnum)
DGII_RNC_WHITELIST = frozenset({
    "101581601", "101582245", "101595422", "101595785", "10233317",
    "130067291", "131188691", "131188693", "401007374", "401506254",
})


def _default_strict_checksum() -> bool:
    return os.getenv("STRICT_RNC_CHECKSUM", "false").strip().lower() in ("1", "true", "yes")


def is_valid_rnc_checksum(rnc: str) -> bool:
    """Verify a 9-digit Dominican RNC using the official DGII Modulo 11 algorithm."""
    digits = _clean_digits(rnc)
    if not RNC_PATTERN.fullmatch(digits) or len(set(digits)) == 1:
        return False
    if digits in DGII_RNC_WHITELIST:
        return True
    return calc_rnc_check_digit(digits[:8]) == digits[8]


def is_valid_cedula_checksum(cedula: str) -> bool:
    """Verify an 11-digit Dominican Cédula using the Modulo 10 (Luhn) algorithm."""
    digits = _clean_digits(cedula)
    if not CEDULA_PATTERN.fullmatch(digits) or digits == "00000000000":
        return False
    weights = (1, 2, 1, 2, 1, 2, 1, 2, 1, 2)
    total = 0
    for w, d in zip(weights, digits[:10]):
        prod = w * int(d)
        if prod >= 10:
            prod = (prod // 10) + (prod % 10)
        total += prod
    check_digit = (10 - (total % 10)) % 10
    return check_digit == int(digits[10])


def is_valid_identifier_checksum(identifier: Any) -> bool:
    """Return True if the RNC (9 digits) or Cédula (11 digits) passes mathematical checksum."""
    digits = _clean_digits(identifier)
    if len(digits) == 9:
        return is_valid_rnc_checksum(digits)
    if len(digits) == 11:
        return is_valid_cedula_checksum(digits)
    return False


def normalize_identifier(value: Any, *, strict_checksum: bool | None = None) -> str:
    """Return a nine-digit RNC or eleven-digit cedula, or raise ValueError."""
    digits = _clean_digits(value)
    if not (RNC_PATTERN.fullmatch(digits) or CEDULA_PATTERN.fullmatch(digits)):
        raise ValueError("El RNC debe tener 9 dígitos o la cédula 11 dígitos.")
    use_strict = _default_strict_checksum() if strict_checksum is None else strict_checksum
    if use_strict and not is_valid_identifier_checksum(digits):
        tipo = "RNC" if len(digits) == 9 else "cédula"
        raise ValueError(f"El dígito verificador del {tipo} ({digits}) no es válido según la DGII.")
    return digits


def infer_tipo_id(rnc_cedula: Any) -> str:
    return "1" if len(normalize_identifier(rnc_cedula)) == 9 else "2"


def normalize_ncf(value: Any, *, allow_non_606: bool = False) -> str:
    """Validate a paper NCF (B + 10 digits) or e-CF (E + 12 digits) for Form 606."""
    ncf = re.sub(r"[\s._-]", "", str(value or "")).upper()
    if NCF_606_PATTERN.fullmatch(ncf) or ECF_606_PATTERN.fullmatch(ncf):
        return ncf

    if ALL_NCF_PATTERN.fullmatch(ncf) or ALL_ECF_PATTERN.fullmatch(ncf):
        if allow_non_606:
            return ncf
        code = ncf[1:3]
        if code in CONSUMO_TYPES:
            raise ValueError(
                f"NCF inválido para el 606 ({ncf}): las Facturas de Consumo (B02/E32) "
                "no son deducibles en el Formato 606. Solicite Crédito Fiscal (B01/E31)."
            )
        if code in NON_606_SALES_TYPES:
            raise ValueError(
                f"NCF inválido para el 606 ({ncf}): el tipo {ncf[:3]} corresponde a "
                "comprobantes de ventas/ingresos (607), no a compras (606)."
            )

    raise ValueError(
        "NCF inválido. Use B + tipo autorizado 606 (01,03,04,11,13,14,17) + 8 dígitos, "
        "o E + tipo electrónico (31,33,34,41,43,44,47) + 10 dígitos."
    )


def ncf_type(value: Any) -> str:
    return normalize_ncf(value, allow_non_606=True)[1:3]


def normalize_date(value: Any) -> str:
    """Normalise common manual date formats to the DGII YYYYMMDD representation."""
    raw = str(value or "").strip()
    if DATE_PATTERN.fullmatch(raw):
        candidate = raw
    else:
        match = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", raw)
        if not match:
            raise ValueError("La fecha debe tener formato DD/MM/AAAA o AAAAMMDD.")
        day, month, year = (int(part) for part in match.groups())
        candidate = f"{year:04d}{month:02d}{day:02d}"

    try:
        parsed = datetime.strptime(candidate, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError("La fecha indicada no existe en el calendario.") from exc

    today = date.today()
    if parsed.year < 2000 or parsed > today:
        raise ValueError("El año o la fecha está fuera del rango permitido (no puede ser una fecha futura).")
    return candidate


def as_date(value: Any) -> date:
    return datetime.strptime(normalize_date(value), "%Y%m%d").date()


def normalize_amount(value: Any, *, allow_empty: bool = False) -> Decimal | None:
    """Parse an amount safely, accepting Dominican and international separators."""
    raw = str(value or "").strip().replace("RD$", "").replace("US$", "")
    raw = raw.replace("DOP", "").replace("USD", "").replace("$", "").strip()
    raw = raw.replace(" ", "")
    if not raw:
        if allow_empty:
            return None
        raise ValueError("El monto es obligatorio.")

    if re.fullmatch(r"\d{1,3}(?:\.\d{3})*,\d{1,2}", raw):
        raw = raw.replace(".", "").replace(",", ".")
    elif "," in raw and "." not in raw:
        raw = raw.replace(",", ".")
    else:
        raw = raw.replace(",", "")

    try:
        amount = Decimal(raw).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("El monto debe ser un número válido con hasta dos decimales.") from exc

    if amount < 0 or amount > MAX_AMOUNT:
        raise ValueError("El monto debe estar entre 0.00 y 999,999,999,999.99.")
    return amount


def decimal_text(value: Decimal) -> str:
    return format(value.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP), ".2f")


def _amount(data: dict[str, Any], field: str) -> Decimal:
    return normalize_amount(data.get(field), allow_empty=True) or Decimal("0.00")


def normalize_extracted_fields(fields: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Normalise OCR values and return field names that need human correction."""
    normalized = dict(fields)
    invalid: list[str] = []
    normalizers = {
        "rnc_cedula": normalize_identifier,
        "ncf": normalize_ncf,
        "fecha_comprobante": normalize_date,
        "monto_total": normalize_amount,
        "itbis_facturado": normalize_amount,
    }
    for field, normalizer in normalizers.items():
        if normalized.get(field) in (None, ""):
            continue
        try:
            value = normalizer(normalized[field])
            normalized[field] = decimal_text(value) if isinstance(value, Decimal) else value
        except ValueError:
            normalized.pop(field, None)
            invalid.append(field)

    optional_amounts = (
        "monto_propina_legal",
        "impuesto_selectivo_consumo",
        "otros_impuestos_tasas",
        "itbis_retenido",
        "isr_retenido",
    )
    for field in optional_amounts:
        if normalized.get(field) in (None, ""):
            continue
        try:
            value = normalize_amount(normalized[field], allow_empty=True)
            if value is not None:
                normalized[field] = decimal_text(value)
        except ValueError:
            normalized.pop(field, None)
    return normalized, invalid


def validate_invoice(data: dict[str, Any], *, strict_checksum: bool | None = None) -> list[ValidationIssue]:
    """Return every locally verifiable blocking issue for an exportable 606 row."""
    issues: list[ValidationIssue] = []
    use_strict = _default_strict_checksum() if strict_checksum is None else strict_checksum

    def require_normalized(field: str, normalizer, label: str) -> Any | None:
        try:
            return normalizer(data.get(field))
        except ValueError as exc:
            issues.append(ValidationIssue(field, f"{label}: {exc}"))
            return None

    identifier = require_normalized(
        "rnc_cedula",
        lambda v: normalize_identifier(v, strict_checksum=strict_checksum),
        "RNC/Cédula inválido",
    )
    ncf = require_normalized("ncf", normalize_ncf, "NCF inválido")
    invoice_date = require_normalized("fecha_comprobante", as_date, "Fecha de comprobante inválida")
    payment_date: date | None = None
    if data.get("fecha_pago"):
        payment_date = require_normalized("fecha_pago", as_date, "Fecha de pago inválida")
        if invoice_date and payment_date and payment_date < invoice_date:
            issues.append(ValidationIssue("fecha_pago", "La fecha de pago no puede ser anterior a la fecha del comprobante."))

    expected_tipo_id: str | None = None
    if identifier:
        expected_tipo_id = "1" if len(identifier) == 9 else "2"
        if data.get("tipo_id") and str(data["tipo_id"]) != expected_tipo_id:
            issues.append(ValidationIssue("tipo_id", "El Tipo Id no coincide con la longitud del RNC/Cédula."))

    if str(data.get("tipo_bienes_servicios", "")) not in VALID_CATEGORIES:
        issues.append(ValidationIssue("tipo_bienes_servicios", "Seleccione un tipo de bienes y servicios válido (01 a 11)."))
    if str(data.get("forma_pago", "")) not in VALID_PAYMENT_METHODS:
        issues.append(ValidationIssue("forma_pago", "Seleccione una forma de pago válida (01 a 07)."))

    monetary_fields = (
        "monto_total", "monto_servicios", "monto_bienes", "itbis_facturado",
        "itbis_retenido", "itbis_proporcionalidad", "itbis_llevado_costo",
        "isr_retenido", "impuesto_selectivo_consumo", "otros_impuestos_tasas",
        "monto_propina_legal",
    )
    parsed_amounts: dict[str, Decimal] = {}
    for field in monetary_fields:
        try:
            parsed_amounts[field] = _amount(data, field)
        except ValueError as exc:
            issues.append(ValidationIssue(field, str(exc)))

    ncf_code = ncf[1:3] if ncf else ""
    is_exterior = ncf_code in FOREIGN_PAYMENT_TYPES
    is_gasto_menor = ncf_code in MINOR_EXPENSE_TYPES
    is_regimen_especial = ncf_code in SPECIAL_REGIME_TYPES
    is_compra_informal = ncf_code in INFORMAL_PURCHASE_TYPES

    if "monto_total" in parsed_amounts:
        base = parsed_amounts.get("monto_servicios", Decimal("0")) + parsed_amounts.get("monto_bienes", Decimal("0"))
        itbis = parsed_amounts.get("itbis_facturado", Decimal("0"))
        itbis_ret = parsed_amounts.get("itbis_retenido", Decimal("0"))
        itbis_prop = parsed_amounts.get("itbis_proporcionalidad", Decimal("0"))
        itbis_costo = parsed_amounts.get("itbis_llevado_costo", Decimal("0"))
        isr_ret = parsed_amounts.get("isr_retenido", Decimal("0"))
        isc = parsed_amounts.get("impuesto_selectivo_consumo", Decimal("0"))
        others = parsed_amounts.get("otros_impuestos_tasas", Decimal("0"))
        tip = parsed_amounts.get("monto_propina_legal", Decimal("0"))

        expected_total = base + itbis + isc + others + tip
        if abs(parsed_amounts["monto_total"] - expected_total) > MONEY_TOLERANCE:
            issues.append(ValidationIssue(
                "monto_total",
                "El total del comprobante debe ser igual a bienes + servicios + ITBIS + ISC + otros impuestos + propina.",
            ))

        maximum_itbis = ((base + isc) * Decimal("0.18")).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)
        if itbis > maximum_itbis + ITBIS_ROUNDING_TOLERANCE:
            issues.append(ValidationIssue("itbis_facturado", "El ITBIS no puede exceder el 18% de la base imponible más ISC."))

        if itbis_ret > itbis:
            issues.append(ValidationIssue("itbis_retenido", "El ITBIS retenido no puede superar el ITBIS facturado."))
        if itbis_prop > itbis:
            issues.append(ValidationIssue("itbis_proporcionalidad", "El ITBIS sujeto a proporcionalidad no puede superar el ITBIS facturado."))
        if itbis_costo > itbis:
            issues.append(ValidationIssue("itbis_llevado_costo", "El ITBIS llevado al costo no puede superar el ITBIS facturado."))
        if itbis_costo + itbis_prop > itbis + MONEY_TOLERANCE:
            issues.append(ValidationIssue(
                "itbis_llevado_costo",
                "La suma de ITBIS llevado al costo e ITBIS sujeto a proporcionalidad no puede superar el ITBIS facturado.",
            ))

        # Validación de Propina Legal (tope 10% sobre base según Ley 18-88)
        if tip > Decimal("0") and base > Decimal("0"):
            max_tip = (base * Decimal("0.10")).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)
            if tip > max_tip + ITBIS_ROUNDING_TOLERANCE:
                issues.append(ValidationIssue(
                    "monto_propina_legal",
                    "La propina legal no puede exceder el 10% del monto consumido (servicios + bienes).",
                ))

        # Reglas específicas por tipo de NCF
        if is_gasto_menor and (itbis > Decimal("0") or itbis_ret > Decimal("0") or isr_ret > Decimal("0")):
            issues.append(ValidationIssue(
                "itbis_facturado",
                "Los comprobantes de Gastos Menores (B13/E43) no deben incluir ITBIS ni retenciones.",
            ))
        if is_regimen_especial and itbis > Decimal("0"):
            issues.append(ValidationIssue(
                "itbis_facturado",
                "Los comprobantes de Regímenes Especiales (B14/E44) están exentos y no deben incluir ITBIS facturado.",
            ))
        if is_exterior and itbis > Decimal("0"):
            issues.append(ValidationIssue(
                "itbis_facturado",
                "Los comprobantes de Pagos al Exterior (B17/E47) no deben incluir ITBIS facturado.",
            ))
        if is_compra_informal and itbis > Decimal("0") and abs(itbis_ret - itbis) > MONEY_TOLERANCE:
            issues.append(ValidationIssue(
                "itbis_retenido",
                "En Comprobantes de Compras (B11/E41) con ITBIS, la Norma 05-19 exige retener el 100% del ITBIS facturado.",
            ))

    has_retention = any(
        parsed_amounts.get(field, Decimal("0")) > 0
        for field in ("itbis_retenido", "isr_retenido")
    ) or bool(data.get("tipo_retencion_isr"))
    if has_retention and not payment_date:
        issues.append(ValidationIssue("fecha_pago", "La fecha de pago es obligatoria cuando existe una retención."))

    tipo_ret_isr = str(data.get("tipo_retencion_isr") or "").strip()
    if tipo_ret_isr and tipo_ret_isr not in VALID_ISR_RETENTION_TYPES:
        issues.append(ValidationIssue("tipo_retencion_isr", "El tipo de retención ISR no es válido."))
    if parsed_amounts.get("isr_retenido", Decimal("0")) > 0 and not tipo_ret_isr:
        issues.append(ValidationIssue("tipo_retencion_isr", "Indique el tipo de retención ISR cuando exista monto retenido."))

    # Prohibición de retención de ISR por honorarios/alquileres a Personas Jurídicas locales
    effective_tipo_id = str(data.get("tipo_id") or expected_tipo_id or "")
    if (
        effective_tipo_id == "1"
        and not is_exterior
        and (parsed_amounts.get("isr_retenido", Decimal("0")) > 0 or bool(tipo_ret_isr))
        and tipo_ret_isr not in LEGAL_ENTITY_ALLOWED_ISR_TYPES
    ):
        issues.append(ValidationIssue(
            "tipo_retencion_isr",
            "No se debe aplicar retención de ISR por alquileres/honorarios a Personas Jurídicas (RNC) en el 606.",
        ))

    if ncf and ncf_code in NOTE_TYPES:
        try:
            normalize_ncf(data.get("ncf_modificado"))
        except ValueError:
            issues.append(ValidationIssue("ncf_modificado", "Las notas de débito o crédito requieren el NCF modificado."))

    return issues
