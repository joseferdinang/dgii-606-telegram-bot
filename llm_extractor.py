"""LLM-based invoice field extractor powered by OpenAI (ChatGPT).

Extracts and normalizes Dominican DGII Form 606 fields from OCR text or images
using structured JSON output with high precision.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres un experto tributario y contable de la República Dominicana especializado en el Formato 606 de la Dirección General de Impuestos Internos (DGII).
Tu tarea es extraer con máxima precisión los datos fiscales de la factura o comprobante provisto.

Reglas fiscales obligatorias de la DGII:
1. RNC/Cédula del emisor (rnc_cedula):
   - Extrae el RNC (9 dígitos) o Cédula (11 dígitos) del PROVEEDOR / EMISOR (quien vende).
   - NUNCA confundas con el RNC o Cédula del CLIENTE / COMPRADOR. Si dice "RNC Cliente", "RNC Comprador", ese no es.
   - Remueve guiones y espacios.

2. Número de Comprobante Fiscal (ncf):
   - Estructura válida: 'B' seguido de 10 dígitos (ej: B0100000001, B0200000001, B0400000001, B1400000001, B1500000001) o 'E' seguido de 12 dígitos (factura electrónica e-CF, ej: E310000000001, E32..., E34...).
   - Si es Nota de Crédito (B04/E34) o Débito (B03/E33), debes extraer también el 'ncf_modificado' (el comprobante original afectado).

3. Fecha del comprobante (fecha_comprobante):
   - Formato estricto AAAAMMDD (ejemplo: 15 de octubre de 2026 -> 20261015).
   - No confundir con fecha de vencimiento ni fecha de impresión del ticket.

4. Desglose de montos monetarios (números flotantes con hasta 2 decimales):
   - monto_total: Valor total final de la factura incluyendo todos los impuestos.
   - itbis_facturado: Total del ITBIS facturado (18% o 16%). Si la factura está exenta de ITBIS, es 0.00.
   - monto_servicios: Valor neto facturado exclusivamente en servicios (sin ITBIS).
   - monto_bienes: Valor neto facturado exclusivamente en compra de bienes físicos (sin ITBIS).
   - monto_propina_legal: 10% de propina de ley en restaurantes o bares (no debe sumarse a bienes/servicios).
   - impuesto_selectivo_consumo: ISC si aplica.
   - otros_impuestos_tasas: Tasas adicionales (ej: CDT, tasas de aeropuerto, etc.).

5. Clasificación DGII:
   - tipo_bienes_servicios: Código de 2 dígitos según corresponda:
     '01': Gastos de personal
     '02': Gastos por trabajos, suministros y servicios
     '03': Arrendamientos
     '04': Gastos de activos fijos
     '05': Gastos de representación (restaurantes, comidas con clientes)
     '06': Otras deducciones admitidas
     '07': Gastos financieros
     '08': Gastos extraordinarios
     '09': Compras y gastos del costo de venta (mercancía para revender, materia prima, supermercado)
     '10': Adquisición de activos
     '11': Gastos de seguros
   - forma_pago: Código de 2 dígitos:
     '01': Efectivo
     '02': Cheques / Transferencias / Depósito
     '03': Tarjeta de crédito o débito (POS, Azul, CardNet, Visa, Mastercard)
     '04': Compra a crédito
     '05': Permuta
     '06': Nota de crédito
     '07': Mixto

6. Moneda:
   - 'DOP' o 'USD'. Si es USD, extrae la 'tasa_cambio' si está impresa.
"""

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "rnc_cedula": {
            "type": ["string", "null"],
            "description": "RNC (9 dígitos) o Cédula (11 dígitos) del emisor/proveedor sin guiones.",
        },
        "ncf": {
            "type": ["string", "null"],
            "description": "NCF oficial emitido por el vendedor (ej. B0100000001 o E310000000001).",
        },
        "ncf_modificado": {
            "type": ["string", "null"],
            "description": "NCF afectado si es nota de crédito o débito.",
        },
        "fecha_comprobante": {
            "type": ["string", "null"],
            "description": "Fecha de emisión en formato AAAAMMDD.",
        },
        "fecha_pago": {
            "type": ["string", "null"],
            "description": "Fecha de pago en formato AAAAMMDD si está disponible.",
        },
        "monto_total": {
            "type": ["number", "null"],
            "description": "Monto total pagado o facturado con impuestos incluidos.",
        },
        "monto_servicios": {
            "type": "number",
            "description": "Monto neto en servicios sin ITBIS.",
        },
        "monto_bienes": {
            "type": "number",
            "description": "Monto neto en bienes físicos sin ITBIS.",
        },
        "itbis_facturado": {
            "type": "number",
            "description": "Total ITBIS facturado.",
        },
        "itbis_retenido": {
            "type": "number",
            "description": "Monto de ITBIS retenido si aplica.",
        },
        "tipo_bienes_servicios": {
            "type": ["string", "null"],
            "description": "Código de 2 dígitos (01 a 11) de clasificación de gasto DGII.",
        },
        "forma_pago": {
            "type": ["string", "null"],
            "description": "Código de 2 dígitos (01 a 07) del método de pago.",
        },
        "monto_propina_legal": {
            "type": "number",
            "description": "Monto de propina legal (10% restaurantes).",
        },
        "impuesto_selectivo_consumo": {
            "type": "number",
            "description": "ISC si aplica.",
        },
        "otros_impuestos_tasas": {
            "type": "number",
            "description": "Otros impuestos o tasas.",
        },
        "moneda": {
            "type": "string",
            "enum": ["DOP", "USD"],
            "description": "Moneda de la factura.",
        },
        "tasa_cambio": {
            "type": ["number", "null"],
            "description": "Tasa de cambio DOP/USD si está impresa.",
        },
    },
    "required": [
        "monto_servicios",
        "monto_bienes",
        "itbis_facturado",
        "itbis_retenido",
        "monto_propina_legal",
        "impuesto_selectivo_consumo",
        "otros_impuestos_tasas",
        "moneda",
    ],
    "additionalProperties": False,
}


class LLMFieldExtractor:
    """Uses OpenAI models (ChatGPT / GPT-4o-mini) to extract Dominican invoice fields."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key.strip() if api_key else None
        self.model = model or "gpt-4o-mini"
        self._client = None

        if self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI LLMFieldExtractor initialized successfully with model %s", self.model)
            except Exception:
                logger.exception("Failed to initialize OpenAI client")
        else:
            logger.info("OpenAI LLMFieldExtractor initialized in mock/fallback mode (no API key provided).")

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def extract_from_text(self, raw_text: str) -> Dict[str, Any]:
        """Extracts structured invoice fields from OCR raw text."""
        if not self.is_available or not raw_text.strip():
            return {}

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Extrae los campos fiscales de esta factura según el Formato 606:\n\n{raw_text}",
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "invoice_fields",
                        "strict": True,
                        "schema": EXTRACTION_SCHEMA,
                    },
                },
                temperature=0.0,
            )

            result_json = response.choices[0].message.content
            extracted = json.loads(result_json)
            return self._normalize_extracted_dict(extracted)

        except Exception:
            logger.exception("Error extracting invoice fields via OpenAI API")
            return {}

    def extract_from_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> Dict[str, Any]:
        """Extracts fields directly from invoice image using multimodal vision."""
        if not self.is_available or not image_bytes:
            return {}

        try:
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            image_url = f"data:{mime_type};base64,{b64_image}"

            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extrae los campos fiscales de esta imagen de factura para la DGII 606:"},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "invoice_fields",
                        "strict": True,
                        "schema": EXTRACTION_SCHEMA,
                    },
                },
                temperature=0.0,
            )

            result_json = response.choices[0].message.content
            extracted = json.loads(result_json)
            return self._normalize_extracted_dict(extracted)

        except Exception:
            logger.exception("Error extracting invoice image via OpenAI API")
            return {}

    @staticmethod
    def _normalize_extracted_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitizes and converts float amounts to formatted strings compatible with the bot state."""
        normalized: Dict[str, Any] = {}

        if data.get("rnc_cedula"):
            normalized["rnc_cedula"] = str(data["rnc_cedula"]).replace("-", "").strip()

        if data.get("ncf"):
            normalized["ncf"] = str(data["ncf"]).upper().strip()

        if data.get("ncf_modificado"):
            normalized["ncf_modificado"] = str(data["ncf_modificado"]).upper().strip()

        if data.get("fecha_comprobante"):
            normalized["fecha_comprobante"] = str(data["fecha_comprobante"]).strip()

        if data.get("fecha_pago"):
            normalized["fecha_pago"] = str(data["fecha_pago"]).strip()

        if data.get("tipo_bienes_servicios"):
            normalized["tipo_bienes_servicios"] = str(data["tipo_bienes_servicios"]).zfill(2)

        if data.get("forma_pago"):
            normalized["forma_pago"] = str(data["forma_pago"]).zfill(2)

        if data.get("moneda"):
            normalized["moneda"] = data["moneda"]

        if data.get("tasa_cambio") is not None:
            normalized["tasa_cambio"] = f"{float(data['tasa_cambio']):.2f}"

        # Decimal / monetary fields
        monetary_fields = [
            "monto_total",
            "monto_servicios",
            "monto_bienes",
            "itbis_facturado",
            "itbis_retenido",
            "monto_propina_legal",
            "impuesto_selectivo_consumo",
            "otros_impuestos_tasas",
        ]

        for mf in monetary_fields:
            val = data.get(mf)
            if val is not None:
                try:
                    normalized[mf] = f"{float(val):.2f}"
                except (ValueError, TypeError):
                    pass

        return normalized
