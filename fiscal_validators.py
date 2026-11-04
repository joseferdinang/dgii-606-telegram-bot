"""Fiscal validation and normalisation rules for Dominican DGII Form 606.

The module deliberately validates the structure and arithmetic that can be
verified locally.  It does not claim to verify whether a supplier is active or
whether an NCF was authorised: those checks require the DGII services and are
still performed when the return is submitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re
from typing import Any


MONEY_QUANTUM = Decimal("0.01")
MONEY_TOLERANCE = Decimal("0.01")
MAX_AMOUNT = Decimal("999999999999.99")

RNC_PATTERN = re.compile(r"^\d{9}$")
CEDULA_PATTERN = re.compile(r"^\d{11}$")
NCF_PATTERN = re.compile(r"^B(01|02|03|04|11|12|13|14|15|16|17)\d{8}$")
ECF_PATTERN = re.compile(r"^E(31|32|33|34|41|43|44|45|46|47)\d{10}$")
DATE_PATTERN = re.compile(r"^\d{8}$")

VALID_CATEGORIES = {
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"
}
VALID_PAYMENT_METHODS = {"01", "02", "03", "04", "05", "06", "07"}
VALID_ISR_RETENTION_TYPES = {"01", "02", "03", "04", "05", "06", "07", "08", "09"}
NOTE_TYPES = {"03", "04", "33", "34"}


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    message: str


def _clean_digits(value: Any) -> str:
    return re.sub(r"\D", "", str(value or ""))


def validate_rnc_check_digit(rnc: Any) -> bool:
    """Valida el dígito verificador de un RNC (9 dígitos) según el algoritmo oficial Módulo 11 de la DGII."""
    digits = _clean_digits(rnc)
    if not RNC_PATTERN.fullmatch(digits):
        return False
    weights = [7, 9, 8, 6, 5, 4, 3, 2]
    total = sum(int(d) * w for d, w in zip(digits[:8], weights))
    remainder = total % 11
    expected = 2 if remainder == 0 else (1 if remainder == 1 else 11 - remainder)
    return expected == int(digits[8])


def validate_cedula_check_digit(cedula: Any) -> bool:
    """Valida el dígito verificador de una Cédula (11 dígitos) según el algoritmo oficial Módulo 10 (Luhn)."""
    digits = _clean_digits(cedula)
    if not CEDULA_PATTERN.fullmatch(digits):
        return False
    multipliers = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    total = 0
    for d, m in zip(digits[:10], multipliers):
        prod = int(d) * m
        total += (prod // 10) + (prod % 10) if prod > 9 else prod
    expected = (10 - (total % 10)) % 10
    return expected == int(digits[10])


def validate_identifier_check_digit(value: Any) -> bool:
    """Valida el dígito verificador para un RNC (9 dígitos) o Cédula (11 dígitos)."""
    digits = _clean_digits(value)
    if len(digits) == 9:
        return validate_rnc_check_digit(digits)
    if len(digits) == 11:
        return validate_cedula_check_digit(digits)
    return False


def normalize_identifier(value: Any, *, validate_check_digit: bool = False) -> str:
    """Return a nine-digit RNC or eleven-digit cedula, or raise ValueError."""
    digits = _clean_digits(value)
    if not (RNC_PATTERN.fullmatch(digits) or CEDULA_PATTERN.fullmatch(digits)):
        raise ValueError("El RNC debe tener 9 dígitos o la cédula 11 dígitos.")
    if validate_check_digit and not validate_identifier_check_digit(digits):
        doc_type = "El RNC" if len(digits) == 9 else "La cédula"
        raise ValueError(f"{doc_type} no tiene un dígito verificador válido.")
    return digits


def infer_tipo_id(rnc_cedula: Any) -> str:
    return "1" if len(normalize_identifier(rnc_cedula)) == 9 else "2"


def normalize_ncf(value: Any) -> str:
    """Validate a paper NCF (B + 10 digits) or e-CF (E + 12 digits)."""
    ncf = re.sub(r"[\s._-]", "", str(value or "")).upper()
    if NCF_PATTERN.fullmatch(ncf) or ECF_PATTERN.fullmatch(ncf):
        return ncf
    raise ValueError(
        "NCF inválido. Use B + tipo autorizado + 8 dígitos, "
        "o E + tipo electrónico + 10 dígitos."
    )


def ncf_type(value: Any) -> str:
    return normalize_ncf(value)[1:3]


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
        raise ValueError("La fecha indicada no existe.") from exc

    if not 2000 <= parsed.year <= date.today().year + 1:
        raise ValueError("El año de la fecha está fuera del rango permitido.")
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
    return normalized, invalid


def validate_invoice(data: dict[str, Any], *, strict_check_digit: bool = False) -> list[ValidationIssue]:
    """Return every locally verifiable blocking issue for an exportable 606 row."""
    issues: list[ValidationIssue] = []

    def require_normalized(field: str, normalizer, label: str) -> Any | None:
        try:
            return normalizer(data.get(field))
        except ValueError as exc:
            issues.append(ValidationIssue(field, f"{label}: {exc}"))
            return None

    identifier = require_normalized("rnc_cedula", normalize_identifier, "RNC/Cédula inválido")
    if identifier and strict_check_digit and not validate_identifier_check_digit(identifier):
        doc_type = "RNC" if len(identifier) == 9 else "Cédula"
        issues.append(ValidationIssue("rnc_cedula", f"El {doc_type} no supera la validación de dígito verificador oficial DGII."))
    ncf = require_normalized("ncf", normalize_ncf, "NCF inválido")
    invoice_date = require_normalized("fecha_comprobante", as_date, "Fecha de comprobante inválida")
    payment_date: date | None = None
    if data.get("fecha_pago"):
        payment_date = require_normalized("fecha_pago", as_date, "Fecha de pago inválida")
        if invoice_date and payment_date and payment_date < invoice_date:
            issues.append(ValidationIssue("fecha_pago", "La fecha de pago no puede ser anterior a la fecha del comprobante."))

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

    if "monto_total" not in parsed_amounts:
        pass
    else:
        base = parsed_amounts.get("monto_servicios", Decimal("0")) + parsed_amounts.get("monto_bienes", Decimal("0"))
        itbis = parsed_amounts.get("itbis_facturado", Decimal("0"))
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
        if itbis > maximum_itbis + MONEY_TOLERANCE:
            issues.append(ValidationIssue("itbis_facturado", "El ITBIS no puede exceder el 18% de la base imponible más ISC."))
        if parsed_amounts.get("itbis_retenido", Decimal("0")) > itbis:
            issues.append(ValidationIssue("itbis_retenido", "El ITBIS retenido no puede superar el ITBIS facturado."))
        if parsed_amounts.get("itbis_proporcionalidad", Decimal("0")) > itbis:
            issues.append(ValidationIssue("itbis_proporcionalidad", "El ITBIS sujeto a proporcionalidad no puede superar el ITBIS facturado."))
        if parsed_amounts.get("itbis_llevado_costo", Decimal("0")) > itbis:
            issues.append(ValidationIssue("itbis_llevado_costo", "El ITBIS llevado al costo no puede superar el ITBIS facturado."))

    has_retention = any(
        parsed_amounts.get(field, Decimal("0")) > 0
        for field in ("itbis_retenido", "isr_retenido")
    ) or bool(data.get("tipo_retencion_isr"))
    if has_retention and not payment_date:
        issues.append(ValidationIssue("fecha_pago", "La fecha de pago es obligatoria cuando existe una retención."))
    if data.get("tipo_retencion_isr") and str(data["tipo_retencion_isr"]) not in VALID_ISR_RETENTION_TYPES:
        issues.append(ValidationIssue("tipo_retencion_isr", "El tipo de retención ISR no es válido."))
    if parsed_amounts.get("isr_retenido", Decimal("0")) > 0 and not data.get("tipo_retencion_isr"):
        issues.append(ValidationIssue("tipo_retencion_isr", "Indique el tipo de retención ISR cuando exista monto retenido."))

    if ncf and ncf[1:3] in NOTE_TYPES:
        try:
            normalize_ncf(data.get("ncf_modificado"))
        except ValueError:
            issues.append(ValidationIssue("ncf_modificado", "Las notas de débito o crédito requieren el NCF modificado."))

    return issues
