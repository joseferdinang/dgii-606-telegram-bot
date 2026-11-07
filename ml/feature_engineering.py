"""Feature engineering for ML-based invoice field extraction in Dominican Republic."""
import re
try:
    import pandas as pd
except ImportError:
    pd = None
from typing import List, Dict, Any

FEATURE_COLUMNS = [
    'token_length', 'digit_count', 'alpha_count', 'digit_ratio',
    'has_decimal_separator', 'starts_with_B_or_E', 'is_pure_digits',
    'matches_rnc_length', 'matches_cedula_length', 'matches_date_pattern',
    'has_currency_symbol', 'numeric_value', 'is_zero_amount',
    'relative_position', 'relative_horizontal',
    'ctx_has_rnc_label', 'ctx_has_ncf_label', 'ctx_has_date_label',
    'ctx_has_total_label', 'ctx_has_itbis_label', 'ctx_has_exclusion_label',
    'ctx_has_subtotal_label', 'ctx_has_retention_label',
    'ctx_has_combustible_label', 'ctx_has_tech_label',
    'ctx_has_hardware_label', 'ctx_has_exento_label'
]

def tokenize(text: str) -> List[Dict[str, Any]]:
    """Tokenize text while preserving line and horizontal position."""
    tokens = []
    lines = text.split('\n')
    num_lines = max(len(lines), 1)
    
    for line_idx, line in enumerate(lines):
        norm_line = re.sub(r'([A-Za-z]):(?!//)(\S)', r'\1: \2', line)
        norm_line = re.sub(r'(\d)(e-?NCF|NCF|RNC)', r'\1 \2', norm_line, flags=re.I)
        norm_line = re.sub(r'(\b\d{1,2}[/-][01iIl]\d[/-]\d{4})(\d{1,2}:\d{2})', r'\1 \2', norm_line)
        words = norm_line.split()
        num_words = max(len(words), 1)
        for word_idx, word in enumerate(words):
            tokens.append({
                'text': word,
                'line_idx': line_idx,
                'word_idx': word_idx,
                'relative_position': line_idx / num_lines,
                'relative_horizontal': (word_idx + 1) / num_words,
            })
    return tokens

def parse_numeric(text: str) -> float:
    """Parse monetary and numeric representations into float safely."""
    cleaned = re.sub(r'[^\d.,]', '', text)
    if not cleaned:
        return 0.0
    if ',' in cleaned and '.' in cleaned:
        if cleaned.rfind(',') > cleaned.rfind('.'):
            cleaned = cleaned.replace('.', '')
            cleaned = cleaned.replace(',', '.')
        else:
            cleaned = cleaned.replace(',', '')
    elif ',' in cleaned:
        parts = cleaned.split(',')
        if len(parts[-1]) == 2:
            cleaned = cleaned.replace(',', '.')
        else:
            cleaned = cleaned.replace(',', '')
    try:
        return float(cleaned)
    except ValueError:
        return 0.0

def build_features(tokens: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract contextual and structural feature matrix from tokens."""
    features = []
    for i, token_data in enumerate(tokens):
        text = token_data['text']
        t_len = len(text)
        digits = sum(c.isdigit() for c in text)
        alphas = sum(c.isalpha() for c in text)
        clean_text = re.sub(r'[^\w.-]', '', text)
        clean_digits = re.sub(r'\D', '', text)
        
        # Surrounding context (±3 tokens) and same-line preceding context
        window = tokens[max(0, i-3):min(len(tokens), i+4)]
        ctx_text = " ".join([t['text'].upper() for t in window])
        cur_line = token_data.get('line_idx', -1)
        prev_line_tokens = [t['text'].upper() for t in tokens[max(0, i-4):i+1] if t.get('line_idx') == cur_line]
        prev_window = tokens[max(0, i-3):i+1]
        prev_ctx = " ".join([t['text'].upper() for t in prev_window])
        line_ctx = " ".join(prev_line_tokens) if len(prev_line_tokens) > 1 else prev_ctx
        ncf_window = tokens[max(0, i-3):min(len(tokens), i+2)]
        ncf_ctx = " ".join([t['text'].upper() for t in ncf_window])
        num_val = parse_numeric(text)
        
        feat = {
            'token_length': t_len,
            'digit_count': digits,
            'alpha_count': alphas,
            'digit_ratio': digits / t_len if t_len > 0 else 0.0,
            'has_decimal_separator': bool(re.search(r'[.,]\d{2}[A-Za-z]?$', text)),
            'starts_with_B_or_E': bool(re.match(r'^[BE]\d', text, re.I)),
            'is_pure_digits': text.isdigit(),
            'matches_rnc_length': len(clean_digits) == 9 and t_len <= 12,
            'matches_cedula_length': len(clean_digits) == 11 and t_len <= 14,
            'matches_date_pattern': bool(re.search(r'\b\d{1,2}[/-][01iIl]\d[/-]\d{4}\b|\b\d{4}[/-]\d{2}[/-]\d{2}\b|\b\d{8}\b', text)),
            'has_currency_symbol': bool(re.search(r'\$|RD\$|US\$|DOP|USD', text, re.I)),
            'numeric_value': num_val,
            'is_zero_amount': num_val == 0.0 and digits > 0,
            'relative_position': token_data.get('relative_position', 0.0),
            'relative_horizontal': token_data.get('relative_horizontal', 0.0),
            
            'ctx_has_rnc_label': bool(re.search(r'\b(RNC|CEDULA|C[EÉEO][DO0](?:ULA)?|TAX\s*ID|NIF)\b', prev_ctx)),
            'ctx_has_ncf_label': bool(re.search(r'\b(NCF|E-?CF|E-?NCF|COMPROBANTE|COMP\.?\s*FISCAL)\b', ncf_ctx)),
            'ctx_has_date_label': bool(re.search(r'(FECHA|TECHA|DATE|EMISI[OÓ]N|F\.?\s*COMP|DIGITAL)', line_ctx)),
            'ctx_has_total_label': bool(re.search(r'\b(TOTAL|LOTAL|TOTALUSD|MONTO|IMPORTE|PAGAR|NETO)\b', line_ctx)) and not bool(re.search(r'(SUB\s*[-]?\s*TOTA|DESC\.?\s*TOTAL)', line_ctx)),
            'ctx_has_itbis_label': bool(re.search(r'\b(ITBIS|IT8IS|TTBIS|TIRIS|I\.?T\.?B\.?I\.?S|IVA|TAX|IMPUESTOS?)\b', line_ctx)) and not bool(re.search(r'\b(VALOR|PRECIO|PRECTO)\b', line_ctx)),
            'ctx_has_exclusion_label': bool(re.search(r'(VENCE|VENCIMIENTO|VALIDA\s*HASTA|VALIIAHASTD|VA[L1I7][I1L][A-Z0-9._\-]*\s*HAST|TEL[EÉ]FONO|TEL|ORDEN|PEDIDO|FACTURA\s*NO|DOC\.?\s*NO|SUCURSAL|AUTORIZACI[OÓ]N|CLIENTE|CLENTE|CI1ENIF|CITENIF|C[I1L][I1T]EN|COMPRADOR|CONDIC\.?\s*PAGO|DESC\.?\s*TOTAL|MID|TID|TERMINAL|LOTE|APROB)', line_ctx)),
            'ctx_has_subtotal_label': bool(re.search(r'\b(SUBTOTAL|SUBTOTAI|SUB-TOTAL|BASE|GRAVABLE|VALOR|PRECIO|PRECTO)\b', line_ctx)),
            'ctx_has_retention_label': bool(re.search(r'\b(RETENIDO|RETENCI[OÓ]N|PERCIBIDO)\b', line_ctx)),
            'ctx_has_combustible_label': bool(re.search(r'\b(BOMBA|DISPENSADOR|MANGUERA|SURTIDOR|GAL|GALONES?|GLP|GASOLINA|GASOIL|DIESEL|COMBUSTIBLE)\b', ctx_text)),
            'ctx_has_tech_label': bool(re.search(r'\b(SERIAL|S/?N|GARANT[IÍ]A|MODELO|MARCA|CPU|RAM|SSD|LAPTOP|MONITOR|DELL|LENOVO)\b', ctx_text)),
            'ctx_has_hardware_label': bool(re.search(r'\b(CEMENTO|VARILLA|PINTURA|TUBOS?|PVC|CLAVOS?|CONDUCE|DESPACHO|OCHOA|BELLON)\b', ctx_text)),
            'ctx_has_exento_label': bool(re.search(r'\b(EXENTO|EXENTA|0%|LEY\s*112-00)\b', ctx_text)),
        }
        features.append(feat)
    
    if not features:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    df = pd.DataFrame(features)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df[FEATURE_COLUMNS]
