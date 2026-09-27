"""SQLite-backed state and batch repository.

SQLite is sufficient for a single polling container and preserves both the
in-progress invoice and the confirmed batch after a restart.  The Docker volume
must contain the database file.  For multiple bot replicas, move this interface
to PostgreSQL or Redis and run only one Telegram update consumer.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
import sqlite3
import threading
from typing import Iterator, Optional


class UserState(str, Enum):
    IDLE = "idle"
    PROCESSING_OCR = "processing_ocr"
    WAITING_MANUAL_FIELD = "waiting_manual_field"
    WAITING_EXCHANGE_RATE = "waiting_exchange_rate"
    WAITING_CATEGORY = "waiting_category"
    WAITING_FORMA_PAGO = "waiting_forma_pago"
    WAITING_ALLOCATION = "waiting_allocation"
    WAITING_CONFIRMATION = "waiting_confirmation"


@dataclass
class InvoiceSession:
    state: UserState = UserState.IDLE
    extracted_data: dict = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    field_being_corrected: Optional[str] = None


class DuplicateInvoiceError(ValueError):
    pass


class StateManager:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            connection = sqlite3.connect(self.database_path, timeout=10)
            connection.row_factory = sqlite3.Row
            try:
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    chat_id INTEGER PRIMARY KEY,
                    state TEXT NOT NULL,
                    extracted_data TEXT NOT NULL,
                    missing_fields TEXT NOT NULL,
                    field_being_corrected TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS batch_invoices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    invoice_key TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(chat_id, invoice_key)
                );
                CREATE INDEX IF NOT EXISTS idx_batch_invoices_chat_id
                    ON batch_invoices(chat_id, id);
                CREATE TABLE IF NOT EXISTS supplier_memory (
                    rnc_cedula TEXT PRIMARY KEY,
                    tipo_bienes_servicios TEXT,
                    forma_pago TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def get_session(self, chat_id: int) -> InvoiceSession:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT state, extracted_data, missing_fields, field_being_corrected "
                "FROM sessions WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
        if row is None:
            return InvoiceSession()
        try:
            state = UserState(row["state"])
        except ValueError:
            state = UserState.IDLE
        return InvoiceSession(
            state=state,
            extracted_data=json.loads(row["extracted_data"]),
            missing_fields=json.loads(row["missing_fields"]),
            field_being_corrected=row["field_being_corrected"],
        )

    def _save_session(self, chat_id: int, session: InvoiceSession) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO sessions(chat_id, state, extracted_data, missing_fields, field_being_corrected)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    state = excluded.state,
                    extracted_data = excluded.extracted_data,
                    missing_fields = excluded.missing_fields,
                    field_being_corrected = excluded.field_being_corrected,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    chat_id,
                    session.state.value,
                    json.dumps(session.extracted_data, ensure_ascii=False),
                    json.dumps(session.missing_fields, ensure_ascii=False),
                    session.field_being_corrected,
                ),
            )

    def set_state(self, chat_id: int, new_state: UserState) -> None:
        session = self.get_session(chat_id)
        session.state = new_state
        self._save_session(chat_id, session)

    def save_extracted_data(self, chat_id: int, data: dict, missing_fields: list[str]) -> None:
        self._save_session(
            chat_id,
            InvoiceSession(
                state=UserState.WAITING_MANUAL_FIELD if missing_fields else UserState.WAITING_CATEGORY,
                extracted_data=dict(data),
                missing_fields=list(missing_fields),
                field_being_corrected=missing_fields[0] if missing_fields else None,
            ),
        )

    def update_field(self, chat_id: int, field_name: str, value) -> None:
        session = self.get_session(chat_id)
        session.extracted_data[field_name] = value
        if field_name in session.missing_fields:
            session.missing_fields.remove(field_name)
        self._save_session(chat_id, session)

    def update_session(self, chat_id: int, session: InvoiceSession) -> None:
        self._save_session(chat_id, session)

    def clear_session(self, chat_id: int) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM sessions WHERE chat_id = ?", (chat_id,))

    @staticmethod
    def _invoice_key(data: dict) -> str:
        return f"{data['rnc_cedula']}:{data['ncf']}"

    def add_confirmed_invoice(self, chat_id: int, data: dict) -> int:
        invoice_key = self._invoice_key(data)
        with self._connection() as connection:
            try:
                cursor = connection.execute(
                    "INSERT INTO batch_invoices(chat_id, invoice_key, payload) VALUES (?, ?, ?)",
                    (chat_id, invoice_key, json.dumps(data, ensure_ascii=False)),
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateInvoiceError("Ese NCF ya está incluido en el lote actual.") from exc
            return int(cursor.lastrowid)

    def has_invoice(self, chat_id: int, extracted_data: dict) -> bool:
        ncf = extracted_data.get("ncf") or extracted_data.get("ncf_modificado")
        if not ncf:
            return False
        
        data_for_key = dict(extracted_data)
        data_for_key["ncf"] = ncf
        invoice_key = self._invoice_key(data_for_key)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM batch_invoices WHERE chat_id = ? AND invoice_key = ?",
                (chat_id, invoice_key),
            ).fetchone()
        return row is not None

    def get_batch(self, chat_id: int) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT payload FROM batch_invoices WHERE chat_id = ? ORDER BY id",
                (chat_id,),
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def batch_count(self, chat_id: int) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM batch_invoices WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
        return int(row["total"])

    def batch_period(self, chat_id: int) -> str | None:
        invoices = self.get_batch(chat_id)
        if not invoices:
            return None
        return str(invoices[0]["fecha_comprobante"])[:6]

    def save_supplier_memory(self, rnc_cedula: str, tipo_bienes_servicios: str, forma_pago: str) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO supplier_memory (rnc_cedula, tipo_bienes_servicios, forma_pago)
                VALUES (?, ?, ?)
                ON CONFLICT(rnc_cedula) DO UPDATE SET
                    tipo_bienes_servicios=excluded.tipo_bienes_servicios,
                    forma_pago=excluded.forma_pago,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (rnc_cedula, tipo_bienes_servicios, forma_pago),
            )

    def get_supplier_memory(self, rnc_cedula: str) -> dict[str, str]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT tipo_bienes_servicios, forma_pago FROM supplier_memory WHERE rnc_cedula = ?",
                (rnc_cedula,),
            ).fetchone()
        if row:
            return {"tipo_bienes_servicios": row["tipo_bienes_servicios"], "forma_pago": row["forma_pago"]}
        return {}

    def clear_batch_and_session(self, chat_id: int) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM sessions WHERE chat_id = ?", (chat_id,))
            connection.execute("DELETE FROM batch_invoices WHERE chat_id = ?", (chat_id,))


# Importing the repository creates the database directory.  STATE_DB_PATH is
# intentionally read from the environment so the same module works locally and
# inside Docker without mutable process-global reconfiguration.
state_manager = StateManager(os.getenv("STATE_DB_PATH", "data/state/bot606.sqlite3"))
