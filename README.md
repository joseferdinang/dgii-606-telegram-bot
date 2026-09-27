# Bot Telegram a Formato 606 DGII

Bot asíncrono con `python-telegram-bot` v21 que recibe facturas por foto o PDF,
extrae campos por Google Cloud Vision con fallback opcional a Gemini, exige la
confirmación del usuario y acumula un lote persistente por chat. Al escribir
`Terminar` envía un `.xlsx` de 26 columnas para copiar sus filas en la
Herramienta de Envío Formato 606 de la DGII y elimina el archivo temporal del
servidor.

## Arquitectura

```text
Telegram photo/PDF
        |
OCR (Vision, optional Gemini fallback)
        |
normalisation + fiscal validation
        |
manual correction / category / payment / base allocation
        |
SQLite batch per Telegram chat
        |
Terminar -> openpyxl xlsx -> Telegram document -> secure local deletion
```

- `ocr_engine.py`: OCR and tolerant extraction.
- `fiscal_validators.py`: canonical RNC/Cédula, NCF/e-CF, date and monetary rules.
- `state_manager.py`: durable SQLite state, invoice batch and unique `(RNC, NCF)` guard.
- `handlers/`: fully asynchronous Telegram flow; blocking OCR/PDF work runs in a worker thread.
- `excel_exporter.py`: creates the A:Z layout matching the DGII 606 data columns.

## Fiscal controls applied before confirmation

The bot requires a 9-digit RNC or 11-digit Cédula, an approved structural NCF
or e-CF type, a valid calendar date, a valid payment date when there are
retentions, and a valid payment/category code. It checks that:

- `total comprobante = servicios + bienes + ITBIS + ISC + otros impuestos + propina`.
- ITBIS does not exceed 18% of the taxable base plus ISC.
- retained, proportional and cost ITBIS never exceed billed ITBIS.
- notes of credit/debit include the modified NCF.
- an `(RNC/Cédula, NCF)` is not added twice in the same chat batch.

The checks are local controls. Supplier status, authorisation and prior filing
of an NCF still need the official DGII validation at submission time.

`monto_total` is the invoice total including tax. Before confirmation the bot
asks whether the tax-free base is goods, services or mixed, avoiding the prior
error of writing a gross total in the 606 base columns.

## Excel output

The `606` sheet uses all 26 physical columns from `A` through `Z`, including
the split `AAAAMM`/`DD` columns for invoice and payment dates. The additional
`Instrucciones` sheet identifies the detected reporting period and explains
the copy/paste workflow.

The file intentionally has no VBA macro. The official DGII tool contains the
macro that validates and creates the TXT to upload. Copy the data rows from
this export into that official tool, validate and generate the TXT there.

## Local setup

1. Copy `.env.example` to `.env` and set `TELEGRAM_BOT_TOKEN`.
2. Download the Google Vision service-account JSON as
   `credentials/google-vision.json`.
3. In local development, set `GOOGLE_VISION_CREDENTIALS` in `.env` to the
   absolute or project-relative path of that JSON.
4. Install dependencies and run the test suite:

   ```bash
   python -m pip install -r requirements.txt
   python -m unittest discover -s tests -v
   ```

5. Start the bot:

   ```bash
   python main.py
   ```

Useful commands are `/start`, `/estado`, `/cancelar`, `/terminar` and
`/descartar_lote`. Typing `Terminar` behaves the same as `/terminar`.

## Docker deployment on Debian

1. Create `.env` from `.env.example`; keep the Docker path for
   `GOOGLE_VISION_CREDENTIALS` as `/run/secrets/google-vision.json`.
2. Place the credential file at `credentials/google-vision.json`. The
   container runs as UID `10001`; on Debian grant that user read access without
   making the secret writable:

   ```bash
   sudo chown 10001:10001 credentials/google-vision.json
   sudo chmod 0400 credentials/google-vision.json
   ```
3. Build and start:

   ```bash
   docker compose up -d --build
   docker compose logs -f bot606
   ```

The named `bot606_data` volume persists SQLite state across container restarts.
Generated Excel files are deleted after a successful Telegram upload; the
directory remains mounted only to make interrupted exports diagnosable and to
avoid writing to the container layer.

Only run one polling replica. If the service is scaled horizontally, replace
the SQLite repository with a shared database and move to a single Telegram
update consumer or webhook queue.

## Tests still needed before production sign-off

- A corpus of real invoices, including blurred, rotated, low-light and
  multi-page PDFs, labelled with expected values.
- Tests against exact official DGII macro-template versions whenever DGII
  publishes a new one.
- Acceptance cases for exempt goods, ISC, legal tips, mixed tax rates,
  retentions, credit/debit notes and foreign-currency invoices.
- An operational policy for duplicate NCFs across reporting periods and any
  required edits to a confirmed batch.
- Load testing with concurrent Telegram chats, OCR provider rate limits and
  restart/failure injection during OCR, database writes and document upload.
- A decision on shared PostgreSQL/Redis plus webhook delivery if multiple
  containers will be deployed.
