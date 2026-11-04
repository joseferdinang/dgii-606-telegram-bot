"""
ocr_engine.py
Módulo 2: Motor OCR

Responsabilidades:
    1. Preprocesar la imagen antes de enviarla (escala de grises, contraste, escala mínima).
    2. Enviar la imagen a Google Cloud Vision API (document_text_detection).
    3. Parsear el texto crudo con regex tolerantes a variaciones de OCR.
    4. Reportar qué campos obligatorios no pudieron detectarse.

Nota: analyze_pdf_pages() es el punto de entrada para PDFs.
      Recibe una lista de bytes (una por página) y consolida el resultado.
"""

import io
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

try:
    from google.cloud import vision
except ImportError:
    vision = None

try:
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError:
    Image = ImageEnhance = ImageFilter = None

logger = logging.getLogger(__name__)

# Campos que el OCR puede extraer razonablemente de una factura.
# "tipo_bienes_servicios" siempre se pide al usuario (no aparece en la factura).
REQUIRED_FIELDS = ["rnc_cedula", "ncf", "fecha_comprobante", "monto_total", "itbis_facturado"]

# Resolución mínima en píxeles del lado más corto para enviar a Vision API.
# Imágenes más pequeñas se escalan para mejorar la detección.
MIN_SHORT_SIDE_PX = 1024

# Meses en español para parseo de fechas textuales
_MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}


@dataclass
class FieldConfidence:
    value: str
    confidence: float  # 0.0 to 1.0
    source: str  # "labeled", "pattern", "heuristic"

@dataclass
class OCRResult:
    raw_text: str
    fields: dict = field(default_factory=dict)
    missing_fields: list = field(default_factory=list)
    confidence: dict = field(default_factory=dict)  # field_name -> float
    image_bytes: Optional[bytes] = None


class InvoiceOCREngine:
    """
    Encapsula la llamada a Vision API, el preprocesamiento de imagen y el parseo
    específico de facturas dominicanas.

    Se separa en pasos (preprocess → detect → parse → fallback) para poder
    testear cada capa de forma aislada sin llamar a las APIs reales.
    """

    def __init__(self, credentials_path: Optional[str] = None) -> None:
        self.available = False
        self._client = None
        if credentials_path:
            import os
            os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", credentials_path)
            try:
                self._client = vision.ImageAnnotatorClient()
                self.available = True
            except Exception:
                logger.warning("No fue posible inicializar Google Vision API", exc_info=True)

    # ------------------------------------------------------------------
    # Puntos de entrada públicos
    # ------------------------------------------------------------------

    def analyze(self, image_bytes: bytes, buyer_rnc: Optional[str] = None) -> OCRResult:
        """Punto de entrada para imágenes (JPEG/PNG). bytes → OCRResult estructurado."""
        # 1. Intentar decodificar código QR de factura electrónica DGII (e-CF)
        qr_fields = None
        try:
            from qr_reader import extract_qr_from_image_bytes
            _, qr_fields = extract_qr_from_image_bytes(image_bytes)
        except Exception:
            pass

        raw_text = ""
        if self.available:
            try:
                processed = self._preprocess_image(image_bytes)
                raw_text = self._detect_text(processed)
            except Exception:
                logger.warning("Error llamando a Google Vision API", exc_info=True)

        result = self._build_result(raw_text, image_bytes, buyer_rnc=buyer_rnc)
        result.image_bytes = image_bytes

        # Si se detectó QR de la DGII, los campos del QR tienen prioridad máxima (confianza 1.0)
        if qr_fields:
            for k, v in qr_fields.items():
                if v:
                    result.fields[k] = v
                    result.confidence[k] = 1.0
            result.missing_fields = [f for f in REQUIRED_FIELDS if not result.fields.get(f)]
            logger.info("Campos QR DGII aplicados con éxito: %s", list(qr_fields.keys()))

        return result

    def analyze_pdf_pages(self, pages_bytes: list[bytes], buyer_rnc: Optional[str] = None) -> OCRResult:
        """
        Punto de entrada para PDFs multi-página.

        Estrategia de consolidación:
          - Escaneo de código QR de e-CF en todas las páginas (prioridad 1.0).
          - RNC/NCF/Fecha: se toman de la primera página que los detecte (encabezado).
          - Monto/ITBIS: se toman de la última página que los detecte (totales).
          - Fallback visual: si faltan totales se envía la última página; si faltan datos de emisor, la primera.
        Retorna un único OCRResult consolidado.
        """
        if not pages_bytes:
            return OCRResult(raw_text="", missing_fields=list(REQUIRED_FIELDS))

        # 1. Escanear cada página del PDF en busca de código QR oficial de la DGII
        qr_fields = None
        try:
            from qr_reader import extract_qr_from_image_bytes
            for p_bytes in pages_bytes:
                _, p_qr = extract_qr_from_image_bytes(p_bytes)
                if p_qr:
                    qr_fields = p_qr
                    break
        except Exception:
            pass

        all_texts: list[str] = []
        all_fields: list[dict] = []
        all_confidences: list[dict] = []

        for idx, page_bytes in enumerate(pages_bytes):
            try:
                processed = self._preprocess_image(page_bytes)
                page_text = self._detect_text(processed)
                page_fields, page_confs = self._parse_fields(page_text, buyer_rnc=buyer_rnc)
                all_texts.append(page_text)
                all_fields.append(page_fields)
                all_confidences.append(page_confs)
                logger.info("Página %d/%d procesada. Campos: %s", idx + 1, len(pages_bytes), list(page_fields))
            except Exception:
                logger.exception("Error procesando página %d del PDF", idx + 1)
                all_texts.append("")
                all_fields.append({})
                all_confidences.append({})

        # Consolidar: header fields de la primera página que los tenga,
        # footer fields de la última que los tenga.
        header_fields = ["rnc_cedula", "ncf", "fecha_comprobante"]
        footer_fields = ["monto_total", "itbis_facturado"]
        secondary_fields = ["moneda", "tasa_cambio", "forma_pago", 
                            "voucher_procesador", "voucher_tipo_tarjeta", "voucher_ultimos4"]

        merged: dict = {}
        merged_confs: dict = {}
        for f in header_fields + secondary_fields:
            for idx, page_f in enumerate(all_fields):
                if page_f.get(f):
                    merged[f] = page_f[f]
                    if all_confidences[idx].get(f):
                        merged_confs[f] = all_confidences[idx][f]
                    break

        for f in footer_fields:
            for idx, page_f in enumerate(reversed(all_fields)):
                real_idx = len(all_fields) - 1 - idx
                if page_f.get(f):
                    merged[f] = page_f[f]
                    if all_confidences[real_idx].get(f):
                        merged_confs[f] = all_confidences[real_idx][f]
                    break

        # Si se detectó QR en alguna página del PDF, sus campos tienen prioridad máxima (1.0)
        if qr_fields:
            for k, v in qr_fields.items():
                if v:
                    merged[k] = v
                    merged_confs[k] = 1.0
            logger.info("Campos QR DGII aplicados con éxito desde PDF: %s", list(qr_fields.keys()))

        combined_text = "\n\n--- PÁGINA SIGUIENTE ---\n\n".join(all_texts)
        missing = [f for f in REQUIRED_FIELDS if not merged.get(f)]

        # Seleccionar la mejor página para fallback visual: si faltan montos, la última página; si no, la primera
        target_idx = -1 if ("monto_total" in missing or "itbis_facturado" in missing) else 0
        selected_image = pages_bytes[target_idx] if pages_bytes else None

        result = OCRResult(
            raw_text=combined_text,
            fields=merged,
            missing_fields=missing,
            confidence=merged_confs,
            image_bytes=selected_image,
        )

        return result

        # ------------------------------------------------------------------
    # Preprocesamiento de imagen
    # ------------------------------------------------------------------

    @staticmethod
    def _correct_perspective_cv2(image_bytes: bytes) -> bytes:
        """Attempts to detect a 4-point receipt contour and unwarps perspective using OpenCV."""
        try:
            import cv2
            import numpy as np

            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                return image_bytes

            h, w = img.shape[:2]
            if h < 250 or w < 250:
                return image_bytes

            ratio = 800.0 / h
            small = cv2.resize(img, (int(w * ratio), 800))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 50, 150)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(edged, kernel, iterations=1)

            contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

            doc_contour = None
            img_area = 800 * (w * ratio)
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4 and cv2.contourArea(c) > (img_area * 0.25):
                    doc_contour = approx
                    break

            if doc_contour is not None:
                pts = doc_contour.reshape(4, 2) / ratio
                rect = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]

                (tl, tr, br, bl) = rect
                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))

                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))

                if maxWidth > 200 and maxHeight > 200:
                    dst = np.array([
                        [0, 0],
                        [maxWidth - 1, 0],
                        [maxWidth - 1, maxHeight - 1],
                        [0, maxHeight - 1]], dtype="float32")
                    M = cv2.getPerspectiveTransform(rect, dst)
                    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
                    success, encoded = cv2.imencode(".jpg", warped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    if success:
                        logger.info("Perspectiva de factura corregida exitosamente (%dx%d)", maxWidth, maxHeight)
                        return encoded.tobytes()
        except Exception:
            logger.debug("Perspective correction skipped or failed", exc_info=True)

        return image_bytes

    def _preprocess_image(self, image_bytes: bytes) -> bytes:
        """
        Mejora la imagen antes de enviarla a Vision API.
        """
        # 0. Aplanado de perspectiva para recibos en ángulo
        image_bytes = self._correct_perspective_cv2(image_bytes)

        if Image is None:
            return image_bytes

        try:
            img = Image.open(io.BytesIO(image_bytes))
            w, h = img.size
            
            # 1. Downscale ultra-large images (max 3000px on longest side)
            MAX_LONG_SIDE = 3000
            long_side = max(w, h)
            if long_side > MAX_LONG_SIDE:
                scale = MAX_LONG_SIDE / long_side
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            
            # 2. Upscale small images (min 1024px on shortest side)
            w, h = img.size
            short_side = min(w, h)
            if short_side < MIN_SHORT_SIDE_PX:
                scale = MIN_SHORT_SIDE_PX / short_side
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            
            # 3. Convert to grayscale
            img = img.convert("L")
            
            # 4. Adaptive contrast based on image histogram
            histogram = img.histogram()
            pixels = sum(histogram)
            # Calculate the range where 95% of pixels fall
            cumulative = 0
            low_val = 0
            for i, count in enumerate(histogram):
                cumulative += count
                if cumulative >= pixels * 0.025:
                    low_val = i
                    break
            cumulative = 0
            high_val = 255
            for i in range(255, -1, -1):
                cumulative += histogram[i]
                if cumulative >= pixels * 0.025:
                    high_val = i
                    break
            dynamic_range = high_val - low_val
            
            if dynamic_range < 80:        # Very low contrast (faded thermal receipt)
                contrast_factor = 2.5
            elif dynamic_range < 150:     # Medium contrast
                contrast_factor = 1.8
            else:                          # Already good contrast
                contrast_factor = 1.2
            
            img = ImageEnhance.Contrast(img).enhance(contrast_factor)
            
            # 5. Sharpen
            img = img.filter(ImageFilter.SHARPEN)
            
            # 6. Export as high-quality JPEG
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            return buffer.getvalue()
        except Exception:
            logger.warning("Image preprocessing failed, using original", exc_info=True)
            return image_bytes

    # ------------------------------------------------------------------
    # Vision API
    # ------------------------------------------------------------------

    def _detect_text(self, image_bytes: bytes) -> str:
        """Llama a document_text_detection, óptimo para documentos densos como facturas."""
        if not self.available or not self._client:
            return ""
        image = vision.Image(content=image_bytes)
        response = self._client.document_text_detection(
            image=image,
            image_context={"language_hints": ["es"]},
        )
        if response.error.message:
            raise RuntimeError(f"Error de Vision API: {response.error.message}")
        return response.full_text_annotation.text

    # ------------------------------------------------------------------
    # Construcción del resultado
    # ------------------------------------------------------------------

    def _build_result(self, raw_text: str, original_bytes: bytes, buyer_rnc: Optional[str] = None) -> OCRResult:
        """Parsea el texto crudo y extrae campos fiscales normalizados."""
        fields, confidences = self._parse_fields(raw_text, buyer_rnc=buyer_rnc)
        missing = [f for f in REQUIRED_FIELDS if not fields.get(f)]
        result = OCRResult(raw_text=raw_text, fields=fields, missing_fields=missing, confidence=confidences)

        return result

    # ------------------------------------------------------------------
    # Parseo de campos con regex
    # ------------------------------------------------------------------

    def _parse_fields(self, text: str, buyer_rnc: Optional[str] = None) -> tuple[dict, dict]:
        """
        Extrae campos mediante regex ajustados a formatos fiscales dominicanos.
        Los patrones son tolerantes a variaciones de OCR (espacios extra,
        mayúsculas/minúsculas, separadores, puntos/comas en montos).
        Incluye detección de moneda (USD/DOP), tasa de cambio y datos de
        voucher POS (procesador, tipo de tarjeta, últimos 4 dígitos).
        """
        result: dict[str, Optional[FieldConfidence]] = {}

        result["rnc_cedula"] = self._extract_rnc_cedula(text, buyer_rnc=buyer_rnc)
        result["ncf"] = self._extract_ncf(text)
        result["fecha_comprobante"] = self._extract_fecha(text)
        result["monto_total"] = self._extract_monto(text)
        result["itbis_facturado"] = self._extract_itbis(text)
        result["moneda"] = self._extract_currency(text)               # 'USD' o 'DOP'
        result["tasa_cambio"] = self._extract_tasa_cambio(text)       # None si no en la factura
        result["forma_pago"] = self._extract_forma_pago(text)         # código 01-07
        result["voucher_procesador"] = self._extract_payment_processor(text)  # AZUL/CARDNET/PORTAL/…
        result["voucher_tipo_tarjeta"] = self._extract_card_type(text)        # CRÉDITO/DÉBITO
        result["voucher_ultimos4"] = self._extract_card_last4(text)           # ej. '1234'

        fields_dict = {k: v.value for k, v in result.items() if v is not None}
        conf_dict = {k: v.confidence for k, v in result.items() if v is not None}
        return fields_dict, conf_dict

    @classmethod
    def _extract_rnc_cedula(cls, text: str, buyer_rnc: Optional[str] = None) -> Optional[FieldConfidence]:
        """
        Extrae el RNC (9 dígitos) o Cédula (11 dígitos) del emisor/proveedor.
        Excluye números asignados a etiquetas de cliente/comprador y descarta
        el RNC de la empresa declarante (buyer_rnc).
        Verifica el dígito verificador oficial DGII para priorizar RNCs legítimos.
        """
        from fiscal_validators import validate_identifier_check_digit

        clean_buyer = re.sub(r"\D", "", buyer_rnc or "") if buyer_rnc else None

        candidates = []

        # 1. Candidatos con etiqueta explícita
        for m in re.finditer(
            r"(?:RNC|C[EÉ]DULA|TAX\s*ID|NIF)\s*[:.#\-]?\s*([\d]{3}[\-\s]?[\d]{5,7}[\-\s]?[\d]{1})",
            text,
            re.IGNORECASE,
        ):
            digits = re.sub(r"\D", "", m.group(1))
            if len(digits) in (9, 11):
                prefix = text[max(0, m.start() - 40):m.start()].upper()
                is_client = bool(re.search(r"(?:CLIENTE|COMPRADOR|RECEPTOR|FACTURAD[OA]\s*A|CONSIGNEE|CUSTOMER)", prefix))
                is_vendor = bool(re.search(r"(?:EMISOR|PROVEEDOR|VENDEDOR|SUPLIDOR)", prefix))
                candidates.append((digits, True, is_client, is_vendor, m.start()))

        # 2. Candidatos por formato numérico directo (3-5-1 o 3-7-1 o 9/11 dígitos continuos)
        for m in re.finditer(r"\b(\d{3}-\d{5}-\d|\d{3}-\d{7}-\d|\d{9}|\d{11})\b", text):
            digits = re.sub(r"\D", "", m.group(1))
            if len(digits) in (9, 11):
                if not any(c[0] == digits and abs(c[4] - m.start()) < 20 for c in candidates):
                    prefix = text[max(0, m.start() - 30):m.start()].upper()
                    if re.search(r"(?:TEL|TEL[EÉ]FONO|CEL|WHATSAPP|PEDIDO|ORDEN|NCF|FACTURA)\s*[:.#\-]?\s*$", prefix):
                        continue
                    is_client = bool(re.search(r"(?:CLIENTE|COMPRADOR|RECEPTOR)", prefix))
                    candidates.append((digits, False, is_client, False, m.start()))

        if not candidates:
            return None

        scored = []
        for digits, labeled, is_client, is_vendor, pos in candidates:
            score = 0.5
            if labeled:
                score += 0.3
            if is_vendor:
                score += 0.2
            if is_client:
                score -= 0.6
            if clean_buyer and digits == clean_buyer:
                score -= 0.9

            has_valid_chk = validate_identifier_check_digit(digits)
            if has_valid_chk:
                score += 0.3
            else:
                score -= 0.1

            if pos < len(text) * 0.35:
                score += 0.1

            scored.append((score, digits, labeled, has_valid_chk))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_digits, best_labeled, best_chk = scored[0]

        if best_score > 0.2 or (clean_buyer and best_digits != clean_buyer):
            confidence = min(0.95 if (best_labeled and best_chk) else (0.85 if best_chk else (0.7 if best_labeled else 0.5)), 1.0)
            return FieldConfidence(value=best_digits, confidence=confidence, source="labeled" if best_labeled else "pattern")

        return None

    @staticmethod
    def _extract_ncf(text: str) -> Optional[FieldConfidence]:
        """
        NCF moderno (e-CF): letra B o E + 2 dígitos tipo + 8-10 dígitos de secuencia.
        Ejemplos válidos: B0100000123, E310000012345.

        Busca primero con etiqueta "NCF:" o "e-CF:" y luego por patrón puro.
        """
        # Con etiqueta
        match = re.search(
            r"(?:NCF|e[\-\s]?CF|COMP\.?\s*FISCAL)\s*[:.#]?\s*([BEbe]\d{10,12})",
            text,
            re.IGNORECASE,
        )
        if match:
            return FieldConfidence(value=match.group(1).upper(), confidence=0.9, source="labeled")

        # Sin etiqueta: patrón puro (posición en línea propia o precedido por espacio)
        match = re.search(r"(?<!\d)([BEbe]\d{2}\d{8,10})(?!\d)", text)
        if match:
            return FieldConfidence(value=match.group(1).upper(), confidence=0.6, source="pattern")

        # Con espacios internos (artefacto OCR común): "B01 00000001" → "B0100000001"
        match = re.search(
            r"(?<!\w)([BEbe]\d{2})\s+(\d{8,10})(?!\d)", text
        )
        if match:
            ncf = (match.group(1) + match.group(2)).upper()
            if len(ncf) in (11, 13):
                return FieldConfidence(value=ncf, confidence=0.5, source="pattern")

        return None

    @staticmethod
    def _extract_fecha(text: str) -> Optional[FieldConfidence]:
        """
        Detecta la fecha del comprobante y la normaliza a YYYYMMDD (formato DGII).
        """
        import datetime
        _HOY = datetime.date.today()
        _YEAR_MIN = 2000          # La DGII digitalizó comprobantes desde ~2001
        _YEAR_MAX = _HOY.year + 1  # Toleramos facturas con fecha máximo 1 año adelante

        def _year_ok(y: int) -> bool:
            return _YEAR_MIN <= y <= _YEAR_MAX

        def _date_ok(day: int, month: int, year: int) -> bool:
            return 1 <= day <= 31 and 1 <= month <= 12 and _year_ok(year)

        def _make(year: int, month: int, day: int) -> Optional[str]:
            if _date_ok(day, month, year):
                return f"{year}{month:02d}{day:02d}"
            return None

        def _resolve_2digit_year(yy: str) -> Optional[int]:
            n = int(yy)
            year = 2000 + n if n < 50 else 1900 + n
            return year if _year_ok(year) else None

        # Excluir fechas de vencimiento y otras fechas irrelevantes
        _EXCLUDE_CTX = re.compile(
            r"(?:VENCE|VENCIMIENTO|VENCE\s*EL|EXP\.?\s*DATE|EXPIRY|EXPIRATION|VALID\s*THRU|THRU|HASTA|"
            r"ENTREGA|DELIVERY|ORDEN|ORDER|PEDIDO|DESPACHO|RECIBO|COBRO)"
            r"\s*[:.#]?\s*\d{1,2}[/\-]\d{2,4}"
            r"|\d{1,2}[/\-]\d{2,4}\s*(?:VENCE|EXP\.?|EXPIRY)",
            re.IGNORECASE,
        )
        texto_limpio = _EXCLUDE_CTX.sub(" ", text)

        # ── Prioridad 0: Etiquetas explícitas de emisión (máxima confianza) ──────
        m = re.search(
            r"(?:FECHA\s+(?:DE\s+)?EMISI[OÓ]N|FECHA\s+(?:DEL?\s+)?COMPROBANTE)\s*[:.#]?\s*(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})",
            texto_limpio, re.IGNORECASE,
        )
        if m:
            d_raw, mo_raw, y_raw = m.group(1), m.group(2), m.group(3)
            year = _resolve_2digit_year(y_raw) if len(y_raw) == 2 else (int(y_raw) if _year_ok(int(y_raw)) else None)
            if year:
                result = _make(year, int(mo_raw), int(d_raw))
                if result:
                    return FieldConfidence(value=result, confidence=0.9, source="labeled")

        # ── Prioridad 1: fecha textual en español ────────────────────────
        m = re.search(
            r"(\d{1,2})\s+de\s+([a-záéíóú]+)\s+de\s+(\d{4})",
            texto_limpio, re.IGNORECASE,
        )
        if m:
            day = int(m.group(1))
            month = _MESES_ES.get(m.group(2).lower())
            year = int(m.group(3))
            if month:
                result = _make(year, month, day)
                if result:
                    return FieldConfidence(value=result, confidence=0.9, source="pattern")

        # ── Prioridad 2: ISO YYYY-MM-DD ──────────────────────────────────
        m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", texto_limpio)
        if m:
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            result = _make(year, month, day)
            if result:
                return FieldConfidence(value=result, confidence=0.9, source="pattern")

        # ── Prioridad 3: con etiqueta FECHA / DATE / F. COMP. ────────────
        m = re.search(
            r"(?:FECHA|DATE|F\.?\s*COMP\.?)\s*[:.#]?\s*(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})",
            texto_limpio, re.IGNORECASE,
        )
        if m:
            d_raw, mo_raw, y_raw = m.group(1), m.group(2), m.group(3)
            if len(y_raw) == 2:
                year = _resolve_2digit_year(y_raw)
            else:
                year = int(y_raw) if _year_ok(int(y_raw)) else None
            if year:
                result = _make(year, int(mo_raw), int(d_raw))
                if result:
                    return FieldConfidence(value=result, confidence=0.9, source="labeled")

        # ── Prioridad 4: sin etiqueta, primer DD/MM/YYYY del texto ───────
        for m in re.finditer(r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b", texto_limpio):
            d_raw, mo_raw, y_raw = m.group(1), m.group(2), m.group(3)

            if len(y_raw) == 2:
                year = _resolve_2digit_year(y_raw)
            else:
                year = int(y_raw) if _year_ok(int(y_raw)) else None

            if year is None:
                continue

            result = _make(year, int(mo_raw), int(d_raw))
            if result:
                return FieldConfidence(value=result, confidence=0.6, source="pattern")

        return None

    @staticmethod
    def _normalize_amount(raw: str) -> Optional[str]:
        """
        Normaliza una cadena monetaria cruda a un float string con dos decimales.
        Maneja:
          - Separador de miles con coma:  1,234.56  → 1234.56
          - Separador de miles con punto: 1.234,56  → 1234.56
          - Sin separador de miles:       1234.56   → 1234.56
        """
        raw = raw.strip().replace(" ", "")
        # Formato europeo: punto como miles, coma como decimal
        if re.match(r"^\d{1,3}(\.\d{3})*(,\d{2})$", raw):
            raw = raw.replace(".", "").replace(",", ".")
        elif re.match(r"^\d+,\d{2}$", raw):
            raw = raw.replace(",", ".")
        else:
            # Formato americano: coma como miles, punto como decimal
            raw = raw.replace(",", "")
        try:
            return f"{float(raw):.2f}"
        except ValueError:
            return None

    @classmethod
    def _extract_monto(cls, text: str) -> Optional[FieldConfidence]:
        """
        Extrae el monto total de la factura con 5 estrategias en cascada.
        """
        _CUR = r"(?:RD\s*\$|US\s*\$|DOP|USD|\$)?\s*"
        _SEP = r"\s*[:.#\-]?\s*"
        _AMT = r"([\d][\d.,]+)"

        # ---- PRIORIDAD 1: Etiquetas de total definitivo ----
        P1_KEYWORDS = [
            r"TOTAL\s+A\s+PAGAR", r"TOTAL\s+FACTURADO", r"TOTAL\s+GENERAL",
            r"GRAN\s+TOTAL", r"IMPORTE\s+TOTAL", r"MONTO\s+TOTAL",
            r"TOTAL\s+COMPRA", r"TOTAL\s+VENTA", r"TOTAL\s+NETO",
            r"TOTAL\s+NETA", r"VALOR\s+TOTAL", r"AMOUNT\s+DUE",
            r"BALANCE\s+DUE", r"TOTAL\s+DUE", r"INVOICE\s+TOTAL",
        ]
        for kw in P1_KEYWORDS:
            m = re.search(rf"(?:{kw}){_SEP}{_CUR}{_AMT}", text, re.IGNORECASE)
            if m:
                val = cls._normalize_amount(m.group(1))
                if val:
                    return FieldConfidence(value=val, confidence=0.9, source="labeled")

        # ---- PRIORIDAD 2: "TOTAL" genérico línea por línea ----
        _EXCLUIR = re.compile(
            r"ITBIS|\bTAX\b|ISR|RETENCI[OÓ]N|DESCUENTO|DISCOUNT|PROPINA|\bTIP\b"
            r"|SUB\s*TOTAL|\bITEMS?\b|ART[IÍ]CULOS?|CANTIDAD",
            re.IGNORECASE,
        )
        total_matches = []
        for m in re.finditer(
            rf"^[^\n]*\bTOTAL\b[^\n]*?{_CUR}{_AMT}[^\n]*$",
            text,
            re.IGNORECASE | re.MULTILINE,
        ):
            if _EXCLUIR.search(m.group(0)):
                continue
            val = cls._normalize_amount(m.group(1))
            if val:
                total_matches.append(val)
        if total_matches:
            return FieldConfidence(value=total_matches[-1], confidence=0.8, source="labeled")

        # ---- PRIORIDAD 3: Base imponible / subtotales ----
        P3_KEYWORDS = [
            r"SUBTOTAL", r"MONTO\s+GRAVADO", r"BASE\s+IMPONIBLE",
            r"NET\s+AMOUNT", r"NET\s+TOTAL", r"MONTO\s+SIN\s+ITBIS",
        ]
        for kw in P3_KEYWORDS:
            m = re.search(rf"(?:{kw}){_SEP}{_CUR}{_AMT}", text, re.IGNORECASE)
            if m:
                val = cls._normalize_amount(m.group(1))
                if val:
                    return FieldConfidence(value=val, confidence=0.7, source="labeled")

        # ---- PRIORIDAD 4: Etiqueta en línea propia, monto en la siguiente ----
        m = re.search(
            rf"(?:TOTAL|IMPORTE|MONTO)[^\n]{{0,30}}\n\s*{_CUR}{_AMT}",
            text,
            re.IGNORECASE,
        )
        if m:
            val = cls._normalize_amount(m.group(1))
            if val:
                return FieldConfidence(value=val, confidence=0.6, source="pattern")

        # ---- PRIORIDAD 5: Heurística — mayor monto en formato monetario ----
        
        # Filtro de exclusión antes del monto
        _EXCLUIR_PREV = re.compile(r"(?:TEL|TEL[EÉ]FONO|CEL|WHATSAPP|PEDIDO|ORDEN|FACTURA\s+NO|SUCURSAL)\s*[:.#\-]?\s*$", re.IGNORECASE)

        candidates = re.finditer(
            r"(?:RD\s*\$|US\s*\$|\$|DOP|USD)\s*([\d]{1,3}(?:[.,]\d{3})*[.,]\d{2})",
            text,
        )
        matches = list(candidates)
        if not matches:
            matches = list(re.finditer(
                r"\b([\d]{1,3}(?:,\d{3})+\.\d{2}|\d+\.\d{2})\b",
                text,
            ))
            
        valid_values = []
        for m in matches:
            val_str = m.group(1)
            
            # Verificar prefijo
            prefix = text[max(0, m.start() - 30):m.start()]
            if _EXCLUIR_PREV.search(prefix):
                continue
                
            # Verificar si parece número de teléfono (10-11 dígitos, sin decimal, que aquí ya sabemos que no por el regex, pero por si acaso)
            # En realidad el regex de arriba exige . o , seguidos de 2 dígitos, así que no será teléfono puro.
            # Pero para ser seguros:
            digits_only = re.sub(r"\D", "", val_str)
            if len(digits_only) in (10, 11) and "." not in val_str and "," not in val_str:
                continue

            val = cls._normalize_amount(val_str)
            if val and float(val) > 10:
                valid_values.append(float(val))

        if valid_values:
            logger.debug("_extract_monto heurística: candidatos=%s", valid_values)
            return FieldConfidence(value=f"{max(valid_values):.2f}", confidence=0.3, source="heuristic")

        return None

    @classmethod
    def _extract_itbis(cls, text: str) -> Optional[FieldConfidence]:
        """
        Extrae el ITBIS FACTURADO (columna N del Formato 606) con varios
        patrones en orden de especificidad. Cubre: ITBIS, TAX, IVA, IGV, VAT.
        """
        _CUR = r"(?:RD\s*\$|US\s*\$|DOP|USD|\$)?\s*"
        _SEP = r"\s*[:.#\-]?\s*"
        _AMT = r"([\d][\d.,]+)"

        _NO_FACTURADO = r"(?!\s*(?:RETENIDO|PERCIBID[OA]|PROPORCIONAL|COSTO|ADELANTAR|SUJETO))"

        # ---- PASO 1: "ITBIS FACTURADO" explícito (máxima confianza) ----
        m = re.search(
            rf"ITBIS\s+FACTURAD[OA]{_SEP}{_CUR}{_AMT}",
            text,
            re.IGNORECASE,
        )
        if m:
            val = cls._normalize_amount(m.group(1))
            if val:
                return FieldConfidence(value=val, confidence=0.9, source="labeled")

        # ---- PASO 2: "ITBIS" sin calificar o con porcentaje, excluyendo
        #              explícitamente las otras columnas de ITBIS ----
        m = re.search(
            rf"ITBIS\s*(?:\(?\s*1[0-9]\s*%?\s*\)?)?\s*{_NO_FACTURADO}{_SEP}{_CUR}{_AMT}",
            text,
            re.IGNORECASE,
        )
        if m:
            val = cls._normalize_amount(m.group(1))
            if val:
                return FieldConfidence(value=val, confidence=0.8, source="labeled")

        # ---- PASO 3: otras etiquetas equivalentes (I.T.B.I.S., IVA, IGV, VAT...) ----
        specific = [
            r"I\.T\.B\.I\.S\.?", r"IVA\s*(?:\(?\s*1[0-9]\s*%?\s*\)?)?",
            r"IGV\s*(?:\(?\s*\d+\s*%?\s*\)?)?", r"\bVAT\b\s*(?:\(?\s*\d+\s*%?\s*\)?)?",
            r"IMPUESTO\s+(?:AL\s+)?VALOR\s+AGREGADO", r"TAX\s+AMOUNT", r"SALES\s+TAX",
        ]
        for pat in specific:
            m = re.search(rf"(?:{pat}){_SEP}{_CUR}{_AMT}", text, re.IGNORECASE)
            if m:
                val = cls._normalize_amount(m.group(1))
                if val:
                    return FieldConfidence(value=val, confidence=0.7, source="labeled")

        # ---- PASO 4: Fallback — "TAX" o "IMPUESTO" genérico línea por línea ----
        for m in re.finditer(
            rf"^[^\n]*(?:TAX|IMPUESTO)[^\n]*?{_CUR}{_AMT}[^\n]*$",
            text,
            re.IGNORECASE | re.MULTILINE,
        ):
            val = cls._normalize_amount(m.group(1))
            if val:
                return FieldConfidence(value=val, confidence=0.6, source="pattern")

        return None

    # ------------------------------------------------------------------
    # Detección de moneda y tasa de cambio
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_currency(text: str) -> FieldConfidence:
        """
        Detecta la moneda de la factura.
        """
        dop_patterns = [
            r'\bDOP\b', r'\bRD\$', r'\bRD\s*\$', r'\bPESOS?\s+DOMINICANOS?\b',
        ]
        for pat in dop_patterns:
            if re.search(pat, text, re.IGNORECASE):
                return FieldConfidence(value='DOP', confidence=0.9, source="labeled")

        usd_patterns = [
            r'\bUSD\b', r'\bUS\s*\$', r'\bU\.S\.D\b',
            r'\bDOLLARS?\b', r'\bDOLARES\b', r'\bDLS\.?\b', r'\bDLLS\.?\b',
        ]
        for pat in usd_patterns:
            if re.search(pat, text, re.IGNORECASE):
                return FieldConfidence(value='USD', confidence=0.9, source="labeled")

        return FieldConfidence(value='DOP', confidence=0.3, source="heuristic")

    @staticmethod
    def _extract_tasa_cambio(text: str) -> Optional[FieldConfidence]:
        """
        Extrae la tasa de cambio USD→DOP que algunas facturas imprimen explícitamente.
        """
        match = re.search(
            r'(?:TASA\s+DE\s+CAMBIO|TIPO\s+DE\s+CAMBIO|EXCHANGE\s+RATE|TASA|T/?C)\s*[:.#]?\s*([\d]+[.,]\d{1,4})',
            text,
            re.IGNORECASE,
        )
        if match:
            raw = match.group(1).replace(',', '.')
            try:
                val = float(raw)
                if 50 <= val <= 200:
                    return FieldConfidence(value=f"{val:.2f}", confidence=0.9, source="labeled")
            except ValueError:
                pass
        return None

    @staticmethod
    def _extract_forma_pago(text: str) -> Optional[FieldConfidence]:
        """
        Detecta la Forma de Pago.
        """
        if re.search(
            r"\bAZUL\b|\bCARDNET\b|\bPORTAL\b|"
            r"TARJETA\s*(?:DE\s*)?(?:CR[EÉ]DITO|D[EÉ]BITO)|"
            r"\bVISA\b|\bMASTERCARD\b|\bMAESTRO\b|"
            r"AMERICAN\s*EXPRESS|\bAMEX\b|"
            r"TERMINAL\s*POS|\bPOS\b|PUNTO\s*DE\s*VENTA|"
            r"CREDIT\s*CARD|DEBIT\s*CARD|"
            r"APROBADO|APROV\b|APPROVAL|APPROV|"
            r"N[ÚU]M\.?\s*(?:APROBACI[OÓ]N|AUTORIZACI[OÓ]N)|AUTORIZACI[OÓ]N\s*N[ÚU]M",
            text,
            re.IGNORECASE,
        ):
            return FieldConfidence(value="03", confidence=0.9, source="labeled")

        if re.search(r"\bEFECTIVO\b|\bCASH\b|PAGO\s+EN\s+EFECTIVO", text, re.IGNORECASE):
            return FieldConfidence(value="01", confidence=0.9, source="labeled")

        if re.search(
            r"\bCHEQUE\b|TRANSFERENCIA|DEP[OÓ]SITO|\bCHECK\b|WIRE\s*TRANSFER",
            text,
            re.IGNORECASE,
        ):
            return FieldConfidence(value="02", confidence=0.9, source="labeled")

        if re.search(r"COMPRA\s+A\s+CR[EÉ]DITO|\bA\s+CR[EÉ]DITO\b", text, re.IGNORECASE):
            return FieldConfidence(value="04", confidence=0.9, source="labeled")

        return None

    # ------------------------------------------------------------------
    # Detección de voucher POS (procesador, tipo de tarjeta, últimos 4)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_payment_processor(text: str) -> Optional[FieldConfidence]:
        """
        Identifica el procesador/red de pago que aparece en el voucher.
        """
        if re.search(r"\bAZUL\b", text, re.IGNORECASE):
            return FieldConfidence(value="AZUL", confidence=0.9, source="labeled")
        if re.search(r"\bCARDNET\b", text, re.IGNORECASE):
            return FieldConfidence(value="CARDNET", confidence=0.9, source="labeled")
        if re.search(r"\bPORTAL\b", text, re.IGNORECASE):
            return FieldConfidence(value="PORTAL", confidence=0.9, source="labeled")

        if re.search(r"\bVISA\b", text, re.IGNORECASE):
            return FieldConfidence(value="VISA", confidence=0.8, source="labeled")
        if re.search(r"\bMASTERCARD\b|\bMAESTRO\b", text, re.IGNORECASE):
            return FieldConfidence(value="MASTERCARD", confidence=0.8, source="labeled")
        if re.search(r"AMERICAN\s*EXPRESS|\bAMEX\b", text, re.IGNORECASE):
            return FieldConfidence(value="AMEX", confidence=0.8, source="labeled")
        if re.search(r"\bDISCOVER\b", text, re.IGNORECASE):
            return FieldConfidence(value="DISCOVER", confidence=0.8, source="labeled")

        return None

    @staticmethod
    def _extract_card_type(text: str) -> Optional[FieldConfidence]:
        """
        Determina si la tarjeta es de CRÉDITO o DÉBITO a partir del voucher.
        """
        credit_match = re.search(
            r"TARJETA\s*(?:DE\s*)?CR[EÉ]DITO|\bCR[EÉ]DITO\b|\bCREDIT\b",
            text, re.IGNORECASE,
        )
        debit_match = re.search(
            r"TARJETA\s*(?:DE\s*)?D[EÉ]BITO|\bD[EÉ]BITO\b|\bDEBIT\b",
            text, re.IGNORECASE,
        )

        if credit_match and debit_match:
            val = "CRÉDITO" if credit_match.start() < debit_match.start() else "DÉBITO"
            return FieldConfidence(value=val, confidence=0.9, source="labeled")
        if credit_match:
            return FieldConfidence(value="CRÉDITO", confidence=0.9, source="labeled")
        if debit_match:
            return FieldConfidence(value="DÉBITO", confidence=0.9, source="labeled")
        return None

    @staticmethod
    def _extract_card_last4(text: str) -> Optional[FieldConfidence]:
        """
        Extrae los últimos 4 dígitos de la tarjeta.
        """
        m = re.search(
            r"(?:\*{4,}|X{4,}|•{4,}|[Xx]{4}\s*[Xx]{4}\s*[Xx]{4})\s*(\d{4})\b",
            text,
        )
        if m:
            return FieldConfidence(value=m.group(1), confidence=0.9, source="pattern")

        m = re.search(
            r"(?:TARJETA|CARD|PAN|NO\.?\s*TARJETA|NUM\.?\s*TARJETA)\s*[:.#]?\s*"
            r"(?:\*+\s*)?(\d{4})\b",
            text, re.IGNORECASE,
        )
        if m:
            return FieldConfidence(value=m.group(1), confidence=0.9, source="labeled")

        m = re.search(r"\b(\d{4})[\s\-]?(\d{4})[\s\-]?(\d{4})[\s\-]?(\d{4})\b", text)
        if m:
            return FieldConfidence(value=m.group(4), confidence=0.6, source="pattern")

        return None

