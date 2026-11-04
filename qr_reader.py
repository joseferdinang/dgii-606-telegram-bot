"""
QR Code Reader and DGII e-CF Parser for Dominican Electronic Invoices.

Detects QR codes on invoice images/PDFs and parses DGII e-CF timbre fiscal URLs:
https://ecf.dgii.gov.do/consultatimbre?RncEmisor=...&eNCF=...&MontoTotal=...&MontoITBIS=...&FechaEmision=...

If a valid DGII QR code is found, extraction is 100% exact and instant,
bypassing OCR and ML error margins completely.
"""

from __future__ import annotations

import io
import logging
import re
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)


def _parse_dgii_params(params: Dict[str, list[str]]) -> Optional[Dict[str, str]]:
    """Normalizes case-insensitive query parameters into DGII 606 fields."""
    normalized_keys = {k.strip().lower(): v[0].strip() for k, v in params.items() if v}
    
    # RNC Emisor
    rnc = normalized_keys.get("rncemisor") or normalized_keys.get("rnc")
    if not rnc:
        return None
    rnc_clean = re.sub(r"\D", "", rnc)
    if len(rnc_clean) not in (9, 11):
        return None

    # e-NCF
    encf = normalized_keys.get("encf") or normalized_keys.get("ncf") or normalized_keys.get("e-ncf")
    if not encf:
        return None
    encf_clean = re.sub(r"[\s._-]", "", encf).upper()
    if not re.match(r"^[BEbe]\d{10,12}$", encf_clean):
        return None

    # Fecha Emisión
    fecha = normalized_keys.get("fechaemision") or normalized_keys.get("fechafirma") or normalized_keys.get("fecha")
    fecha_clean = ""
    if fecha:
        digits = re.sub(r"\D", "", fecha)
        if len(digits) >= 8:
            fecha_clean = digits[:8]  # YYYYMMDD
        elif re.match(r"\d{1,2}[/-]\d{1,2}[/-]\d{4}", fecha):
            parts = re.split(r"[/-]", fecha)
            fecha_clean = f"{int(parts[2]):04d}{int(parts[1]):02d}{int(parts[0]):02d}"

    # Monto Total
    monto_raw = normalized_keys.get("montototal") or normalized_keys.get("total") or normalized_keys.get("monto")
    monto_clean = ""
    if monto_raw:
        try:
            monto_clean = f"{float(monto_raw.replace(',', '')):.2f}"
        except ValueError:
            monto_clean = monto_raw

    # ITBIS
    itbis_raw = normalized_keys.get("montoitbis") or normalized_keys.get("itbis")
    itbis_clean = "0.00"
    if itbis_raw:
        try:
            itbis_clean = f"{float(itbis_raw.replace(',', '')):.2f}"
        except ValueError:
            itbis_clean = itbis_raw

    fields = {
        "rnc_cedula": rnc_clean,
        "ncf": encf_clean,
        "fecha_comprobante": fecha_clean,
        "monto_total": monto_clean,
        "itbis_facturado": itbis_clean,
    }
    
    if "codigoseguridad" in normalized_keys:
        fields["codigo_seguridad"] = normalized_keys["codigoseguridad"]
    if "rnccomprador" in normalized_keys:
        fields["rnc_comprador"] = re.sub(r"\D", "", normalized_keys["rnccomprador"])
        
    return fields


def parse_dgii_qr_text(raw_text: str) -> Optional[Dict[str, str]]:
    """
    Parses decoded QR text into DGII fields.
    Supports both URL format (https://ecf.dgii.gov.do/...) and pipe-separated format.
    """
    raw_text = raw_text.strip()
    if not raw_text:
        return None

    # Case 1: URL format (standard DGII e-CF timbre fiscal)
    if "consultatimbre" in raw_text.lower() or "ecf.dgii.gov.do" in raw_text.lower() or "dgii.gov.do" in raw_text.lower():
        try:
            parsed_url = urlparse(raw_text)
            query_params = parse_qs(parsed_url.query)
            extracted = _parse_dgii_params(query_params)
            if extracted:
                logger.info("Successfully parsed DGII e-CF QR URL: eNCF=%s RNC=%s", extracted.get("ncf"), extracted.get("rnc_cedula"))
                return extracted
        except Exception:
            logger.debug("Failed parsing QR as DGII URL: %s", raw_text, exc_info=True)

    # Case 2: Pipe-delimited offline timbre: RncEmisor|RncComprador|eNCF|FechaEmision|MontoTotal|MontoITBIS|CodigoSeguridad
    if "|" in raw_text:
        parts = [p.strip() for p in raw_text.split("|")]
        if len(parts) >= 5:
            rnc_cand = re.sub(r"\D", "", parts[0])
            ncf_cand = parts[2].upper() if len(parts) > 2 else parts[1].upper()
            if len(rnc_cand) in (9, 11) and re.match(r"^[BEbe]\d{10,12}$", ncf_cand):
                params_dict = {
                    "rncemisor": [parts[0]],
                    "rnccomprador": [parts[1]] if len(parts) > 1 else [],
                    "encf": [ncf_cand],
                    "fechaemision": [parts[3]] if len(parts) > 3 else [],
                    "montototal": [parts[4]] if len(parts) > 4 else [],
                    "montoitbis": [parts[5]] if len(parts) > 5 else ["0.00"],
                }
                extracted = _parse_dgii_params(params_dict)
                if extracted:
                    logger.info("Successfully parsed pipe-delimited DGII QR: eNCF=%s", extracted.get("ncf"))
                    return extracted

    return None


def extract_qr_from_image_bytes(image_bytes: bytes) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    """
    Decodes QR code from raw image bytes and parses any DGII e-CF payload.
    Returns (raw_qr_text, dgii_fields_dict).
    """
    if cv2 is None or np is None:
        logger.debug("OpenCV or NumPy not available for image QR detection")
        return None, None
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return None, None

        # Check with pyzbar if available
        try:
            from pyzbar.pyzbar import decode as pyzbar_decode
            pil_img = Image.open(io.BytesIO(image_bytes))
            decoded_objects = pyzbar_decode(pil_img)
            for obj in decoded_objects:
                if obj.type == "QRCODE":
                    raw_txt = obj.data.decode("utf-8", errors="ignore")
                    dgii = parse_dgii_qr_text(raw_txt)
                    if dgii:
                        return raw_txt, dgii
                    return raw_txt, None
        except Exception:
            pass  # Fall back to OpenCV detector

        # OpenCV QRCodeDetector
        detector = cv2.QRCodeDetector()
        
        # 1. Direct pass
        val, pts, _ = detector.detectAndDecode(img)
        if val:
            dgii = parse_dgii_qr_text(val)
            return val, dgii

        # 2. Grayscale & contrast enhancement
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        val, pts, _ = detector.detectAndDecode(gray)
        if val:
            dgii = parse_dgii_qr_text(val)
            return val, dgii

        # 3. Downscale if very large (> 2000px on any side)
        h, w = gray.shape[:2]
        if max(h, w) > 2000:
            scale = 1600 / max(h, w)
            resized = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            val, pts, _ = detector.detectAndDecode(resized)
            if val:
                dgii = parse_dgii_qr_text(val)
                return val, dgii

        # 4. Otsu thresholding for low-contrast/thermal receipts
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        val, pts, _ = detector.detectAndDecode(thresh)
        if val:
            dgii = parse_dgii_qr_text(val)
            return val, dgii

    except Exception:
        logger.warning("Error running QR detector on image", exc_info=True)

    return None, None
