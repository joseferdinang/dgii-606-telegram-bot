# 🤖 Bot de Telegram para Declaración DGII Formato 606

Bot empresarial asíncrono desarrollado con `python-telegram-bot` (v21) para la recepción, extracción fiscal con Inteligencia Artificial / OCR y declaración de comprobantes fiscales dominicanos en el **Formato 606 de la Dirección General de Impuestos Internos (DGII)**.

El bot genera directamente el archivo **TXT oficial** listo para importar en la **Oficina Virtual (OFV)** de la DGII y, de manera complementaria, un libro en **Excel (.xlsx)** con las 26 columnas reglamentarias como respaldo contable.

---

## 🚀 Características Principales

1. **Aislamiento Multi-Empresa con PIN Secreto:**
   - Cada empresa registrada cuenta con su propio PIN de seguridad hasheado con **PBKDF2-HMAC-SHA256 (100,000 iteraciones)**.
   - Protección contra fuerza bruta (bloqueo automático de 15 minutos tras 5 intentos fallidos).
   - Memoria de proveedores (`supplier_memory`) aislada por RNC: una empresa nunca ve ni contamina la configuración de otra.
2. **Subida Múltiple Simultánea (Cola de Archivos):**
   - Soporte para enviar álbumes de fotos o múltiples documentos PDF al mismo tiempo.
   - El sistema encola los archivos en una cola FIFO persistente y los procesa de forma secuencial y fluida.
3. **Extracción Multimodal con IA:**
   - **QR e-CF Instantáneo:** Detección de timbre fiscal DGII (URL o delimitado por pipes) con prioridad 1.0 (100% exacto).
   - **OpenAI GPT-4o-mini Multimodal Vision:** Inspección visual directa y por texto OCR.
   - **Google Cloud Vision OCR:** Lectura y preprocesamiento de comprobantes físicos y electrónicos.
   - **Machine Learning Local (Data Flywheel):** Clasificador Gradient Boosting que aprende continuamente con cada factura confirmada o corregida.
4. **Validaciones Fiscales DGII:**
   - Validación de dígito verificador: **Módulo 11** para RNCs y **Módulo 10 (Luhn)** para Cédulas.
   - Reglas de comprobantes: B01, B02, B03, B04, B11, B14, B15, B16, B17 y e-CFs (E31 a E45).
   - Detección de duplicados en el lote actual e historial de auditoría contra períodos anteriores (`exported_invoices`).
5. **Control y Resumen en Telegram:**
   - Visualización de resumen financiero con acumulados de ITBIS, Bienes y Servicios (`/resumen`).
   - Retiro de comprobantes individuales (`/eliminar <NCF>`).

---

## 🛠️ Comandos de Telegram

| Comando | Descripción |
|---|---|
| `/registro <RNC> <PIN> <Nombre>` | Registra una nueva empresa en el sistema protegida por PIN. |
| `/login <RNC> <PIN>` | Inicia sesión en una empresa existente. |
| `/empresa` *(o `/rnc`)* | Consulta la empresa activa o alterna entre empresas autorizadas. |
| `/logout` | Cierra la sesión activa actual. |
| `/resumen` *(o `/facturas`)* | Muestra el desglose de importes acumulados y la lista de facturas del lote. |
| `/eliminar <NCF>` | Elimina una factura específica del lote actual. |
| `/estado` | Muestra cantidad de facturas cargadas, período y estado de la cola. |
| `/cola` | Informa cuántos archivos están pendientes en espera de revisión. |
| `/cancelar_cola` | Limpia los archivos pendientes en la cola. |
| `/cancelar` | Descarta la factura en revisión y toma la siguiente de la cola. |
| `/descartar_lote` | Borra todo el lote actual y la cola. |
| `/terminar` *(o escribir `Terminar`)* | Valida el lote, genera el **TXT para la DGII** y el **Excel de respaldo**. |
| `/reentrenar` | Fuerza el re-entrenamiento del clasificador ML con las facturas históricas. |

---

## 📁 Estructura del Proyecto

```text
├── config.py                 # Configuración por variables de entorno (.env)
├── data_flywheel.py          # Aprendizaje continuo ML (Data Flywheel)
├── dgii_txt.py               # Generador del archivo oficial DGII_F_606_<RNC>_<PERIODO>.TXT
├── docker-compose.yml        # Configuración de Docker Compose para Linux/Producción
├── Dockerfile                # Imagen base python:3.12-slim con poppler y libzbar
├── excel_exporter.py         # Exportador del reporte en Excel (.xlsx) con 26 columnas
├── fiscal_validators.py      # Algoritmos DGII (Módulo 11, Módulo 10, NCF, montos)
├── llm_extractor.py          # Extractor con OpenAI (visión y texto)
├── main.py                   # Entrypoint del bot de Telegram y ruteo de comandos
├── ocr_engine.py             # Motor OCR tolerante y consolidación de páginas PDF
├── qr_reader.py              # Lector de QR para timbres fiscales e-CF de la DGII
├── state_manager.py          # Repositorio SQLite (sesiones, lotes, empresas, cola FIFO)
├── handlers/                 # Controladores modulares de Telegram
│   ├── company_handler.py    # Login, registro y cambio de empresa
│   ├── confirmation_handler.py # Botoneras interactivas, edición y guardado
│   ├── exchange_rate_handler.py # Conversión de facturas en USD a DOP
│   ├── invoice_flow.py       # Ruteo post-OCR, validaciones y verificación de períodos
│   ├── pdf_handler.py        # Recepción y conversión de PDFs a imágenes
│   ├── photo_handler.py      # Recepción de fotos e imágenes sin compresión
│   └── queue_handler.py      # Procesador de cola multi-archivo FIFO
├── ml/                       # Machine Learning local
│   ├── feature_engineering.py # Tokenización y vectorización de características
│   ├── synthetic_generator.py # Generador de datos sintéticos dominicanos
│   └── train.py              # Script offline de entrenamiento base
└── tests/                    # Suite de 122 pruebas unitarias
```

---

## 🐳 Despliegue con Docker (Linux / Servidor)

### 1. Clonar el repositorio
```bash
git clone https://github.com/joseferdinang/dgii-606-telegram-bot.git
cd dgii-606-telegram-bot
```

### 2. Configurar el archivo `.env`
```bash
cp .env.example .env
nano .env
```
Configura tus credenciales:
```env
TELEGRAM_BOT_TOKEN=tu_token_de_telegram_aqui
OPENAI_API_KEY=tu_api_key_de_openai_aqui
```
*(Opcional: Si vas a usar Google Cloud Vision, coloca el JSON en `credentials/google-vision.json` y descomenta `GOOGLE_VISION_CREDENTIALS=/app/credentials/google-vision.json`).*

### 3. Iniciar el contenedor
```bash
docker compose up -d --build
```

### 4. Consultar registros en tiempo real
```bash
docker compose logs -f
```

---

## 💻 Ejecución Local (Desarrollo)

1. Crear un entorno virtual e instalar las dependencias:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Ejecutar la suite completa de pruebas:
   ```bash
   python -m unittest discover -s tests -v
   ```
3. Iniciar el bot:
   ```bash
   python main.py
   ```
