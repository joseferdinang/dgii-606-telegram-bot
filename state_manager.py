"""SQLite-backed state, batch repository, and multi-company access management.

SQLite is sufficient for a single polling container and preserves both the
in-progress invoice, confirmed batch, and authorized company profiles after a restart.
The database ensures strict tenant isolation: each Telegram user only has access to
companies they have explicitly registered or unlocked with a valid secret PIN.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import secrets
import sqlite3
import threading
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


def normalize_rnc(rnc: str) -> str:
    """Extrae solo los dígitos de un RNC o Cédula (9 u 11 dígitos)."""
    return re.sub(r"\D", "", str(rnc or ""))


def _hash_pin(pin: str, salt: str) -> str:
    """Genera el hash seguro SHA-256 para el PIN de la empresa con sal."""
    return hashlib.sha256(f"{salt}:{pin}".encode("utf-8")).hexdigest()


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


@dataclass
class Company:
    rnc: str
    razon_social: str
    created_by_chat_id: int
    is_active: bool = False


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
                CREATE TABLE IF NOT EXISTS supplier_memory (
                    rnc_cedula TEXT PRIMARY KEY,
                    tipo_bienes_servicios TEXT,
                    forma_pago TEXT,
                    tipo_asignacion TEXT DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS companies (
                    rnc TEXT PRIMARY KEY,
                    razon_social TEXT NOT NULL,
                    pin_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_by_chat_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS user_companies (
                    chat_id INTEGER NOT NULL,
                    rnc TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    authenticated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (chat_id, rnc)
                );
                """
            )
            # Manejo y migración de batch_invoices con columna rnc
            cursor = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='batch_invoices'"
            )
            row = cursor.fetchone()
            if row is None:
                connection.execute(
                    """
                    CREATE TABLE batch_invoices (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chat_id INTEGER NOT NULL,
                        rnc TEXT NOT NULL DEFAULT '',
                        invoice_key TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(chat_id, rnc, invoice_key)
                    )
                    """
                )
            else:
                cols = [c["name"] for c in connection.execute("PRAGMA table_info(batch_invoices)").fetchall()]
                if "rnc" not in cols:
                    connection.executescript(
                        """
                        CREATE TABLE batch_invoices_new (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            chat_id INTEGER NOT NULL,
                            rnc TEXT NOT NULL DEFAULT '',
                            invoice_key TEXT NOT NULL,
                            payload TEXT NOT NULL,
                            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(chat_id, rnc, invoice_key)
                        );
                        INSERT INTO batch_invoices_new (id, chat_id, rnc, invoice_key, payload, created_at)
                        SELECT id, chat_id, '', invoice_key, payload, created_at FROM batch_invoices;
                        DROP TABLE batch_invoices;
                        ALTER TABLE batch_invoices_new RENAME TO batch_invoices;
                        """
                    )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_batch_invoices_chat_rnc ON batch_invoices(chat_id, rnc, id)"
            )
            sm_cols = [c["name"] for c in connection.execute("PRAGMA table_info(supplier_memory)").fetchall()]
            if "tipo_asignacion" not in sm_cols:
                connection.execute("ALTER TABLE supplier_memory ADD COLUMN tipo_asignacion TEXT DEFAULT ''")

    # -------------------------------------------------------------------------
    # GESTIÓN DE SESIONES DE ENTRADA (OCR Y CORRECCIONES)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # GESTIÓN MULTI-EMPRESA Y CONTROL DE ACCESO (AUTENTICACIÓN CON PIN)
    # -------------------------------------------------------------------------
    def register_company(self, chat_id: int, rnc: str, razon_social: str, pin: str) -> Company:
        """Registra una nueva empresa y la vincula como activa al usuario actual."""
        rnc_clean = normalize_rnc(rnc)
        if len(rnc_clean) not in (9, 11):
            raise ValueError("El RNC debe contener 9 dígitos (empresas) u 11 dígitos (cédula).")
        clean_name = (razon_social or "").strip()
        if not clean_name:
            raise ValueError("Debes ingresar la razón social o nombre de la empresa.")
        pin_str = str(pin or "").strip()
        if len(pin_str) < 4:
            raise ValueError("El PIN de acceso debe tener al menos 4 caracteres o dígitos.")

        salt = secrets.token_hex(16)
        pin_hash = _hash_pin(pin_str, salt)

        with self._connection() as connection:
            existing = connection.execute(
                "SELECT rnc FROM companies WHERE rnc = ?", (rnc_clean,)
            ).fetchone()
            if existing:
                raise ValueError("Este RNC ya está registrado. Si eres usuario autorizado, usa /login para acceder con su PIN.")

            connection.execute(
                "INSERT INTO companies(rnc, razon_social, pin_hash, salt, created_by_chat_id) VALUES (?, ?, ?, ?, ?)",
                (rnc_clean, clean_name, pin_hash, salt, chat_id),
            )
            # Desactivar otras empresas para este chat
            connection.execute("UPDATE user_companies SET is_active = 0 WHERE chat_id = ?", (chat_id,))
            # Activar la recién creada
            connection.execute(
                """
                INSERT INTO user_companies(chat_id, rnc, is_active) VALUES (?, ?, 1)
                ON CONFLICT(chat_id, rnc) DO UPDATE SET is_active = 1
                """,
                (chat_id, rnc_clean),
            )

        return Company(rnc=rnc_clean, razon_social=clean_name, created_by_chat_id=chat_id, is_active=True)

    def login_company(self, chat_id: int, rnc: str, pin: str) -> Company:
        """Autentica a un usuario con el RNC y PIN de una empresa y la activa para su chat."""
        rnc_clean = normalize_rnc(rnc)
        pin_str = str(pin or "").strip()
        if not rnc_clean or not pin_str:
            raise ValueError("Debes ingresar el RNC y el PIN de acceso.")

        with self._connection() as connection:
            row = connection.execute(
                "SELECT rnc, razon_social, pin_hash, salt, created_by_chat_id FROM companies WHERE rnc = ?",
                (rnc_clean,),
            ).fetchone()
            # Mensaje genérico para evitar enumeración de empresas registradas
            if not row or _hash_pin(pin_str, row["salt"]) != row["pin_hash"]:
                raise ValueError("RNC o PIN incorrecto.")

            # Desactivar otras empresas para este chat
            connection.execute("UPDATE user_companies SET is_active = 0 WHERE chat_id = ?", (chat_id,))
            # Vincular y marcar como activa
            connection.execute(
                """
                INSERT INTO user_companies(chat_id, rnc, is_active) VALUES (?, ?, 1)
                ON CONFLICT(chat_id, rnc) DO UPDATE SET is_active = 1
                """,
                (chat_id, rnc_clean),
            )

        return Company(
            rnc=row["rnc"],
            razon_social=row["razon_social"],
            created_by_chat_id=int(row["created_by_chat_id"]),
            is_active=True,
        )

    def get_active_company(self, chat_id: int) -> Optional[Company]:
        """Obtiene la empresa activa actual para este chat_id, o None si no hay ninguna."""
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT c.rnc, c.razon_social, c.created_by_chat_id
                FROM user_companies uc
                JOIN companies c ON uc.rnc = c.rnc
                WHERE uc.chat_id = ? AND uc.is_active = 1
                LIMIT 1
                """,
                (chat_id,),
            ).fetchone()
        if not row:
            return None
        return Company(
            rnc=row["rnc"],
            razon_social=row["razon_social"],
            created_by_chat_id=int(row["created_by_chat_id"]),
            is_active=True,
        )

    def get_user_companies(self, chat_id: int) -> list[Company]:
        """Devuelve ÚNICAMENTE las empresas a las que este usuario tiene acceso autenticado.

        Garantiza aislamiento estricto: un usuario no puede ver ni listar las empresas
        o RNCs de otros usuarios.
        """
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT c.rnc, c.razon_social, c.created_by_chat_id, uc.is_active
                FROM user_companies uc
                JOIN companies c ON uc.rnc = c.rnc
                WHERE uc.chat_id = ?
                ORDER BY uc.is_active DESC, c.razon_social ASC
                """,
                (chat_id,),
            ).fetchall()
        return [
            Company(
                rnc=row["rnc"],
                razon_social=row["razon_social"],
                created_by_chat_id=int(row["created_by_chat_id"]),
                is_active=bool(row["is_active"]),
            )
            for row in rows
        ]

    def switch_active_company(self, chat_id: int, rnc: str) -> Company:
        """Cambia la empresa activa de un usuario entre las empresas que ya tiene autorizadas."""
        rnc_clean = normalize_rnc(rnc)
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT c.rnc, c.razon_social, c.created_by_chat_id
                FROM user_companies uc
                JOIN companies c ON uc.rnc = c.rnc
                WHERE uc.chat_id = ? AND uc.rnc = ?
                """,
                (chat_id, rnc_clean),
            ).fetchone()
            if not row:
                raise PermissionError("No tienes acceso a esta empresa o no has iniciado sesión en ella con su PIN.")

            connection.execute("UPDATE user_companies SET is_active = 0 WHERE chat_id = ?", (chat_id,))
            connection.execute(
                "UPDATE user_companies SET is_active = 1 WHERE chat_id = ? AND rnc = ?",
                (chat_id, rnc_clean),
            )

        return Company(
            rnc=row["rnc"],
            razon_social=row["razon_social"],
            created_by_chat_id=int(row["created_by_chat_id"]),
            is_active=True,
        )

    def logout_company(self, chat_id: int, rnc: str | None = None) -> None:
        """Cierra sesión en una empresa específica o en todas las empresas para este chat."""
        with self._connection() as connection:
            if rnc:
                rnc_clean = normalize_rnc(rnc)
                connection.execute("DELETE FROM user_companies WHERE chat_id = ? AND rnc = ?", (chat_id, rnc_clean))
                # Si queda otra empresa vinculada, activar la primera restante
                remaining = connection.execute(
                    "SELECT rnc FROM user_companies WHERE chat_id = ? ORDER BY authenticated_at DESC LIMIT 1",
                    (chat_id,),
                ).fetchone()
                if remaining:
                    connection.execute(
                        "UPDATE user_companies SET is_active = 1 WHERE chat_id = ? AND rnc = ?",
                        (chat_id, remaining["rnc"]),
                    )
            else:
                connection.execute("DELETE FROM user_companies WHERE chat_id = ?", (chat_id,))

    # -------------------------------------------------------------------------
    # GESTIÓN DEL LOTE DE FACTURAS (AISLADO POR EMPRESA / RNC)
    # -------------------------------------------------------------------------
    def _resolve_rnc(self, chat_id: int, rnc: str | None) -> str:
        if rnc is not None:
            return normalize_rnc(rnc)
        active = self.get_active_company(chat_id)
        return active.rnc if active else ""

    @staticmethod
    def _invoice_key(data: dict) -> str:
        return f"{data['rnc_cedula']}:{data['ncf']}"

    def add_confirmed_invoice(self, chat_id: int, data: dict, rnc: str | None = None) -> int:
        rnc_scope = self._resolve_rnc(chat_id, rnc)
        invoice_key = self._invoice_key(data)
        with self._connection() as connection:
            try:
                cursor = connection.execute(
                    "INSERT INTO batch_invoices(chat_id, rnc, invoice_key, payload) VALUES (?, ?, ?, ?)",
                    (chat_id, rnc_scope, invoice_key, json.dumps(data, ensure_ascii=False)),
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateInvoiceError("Ese NCF ya está incluido en el lote actual.") from exc
            return int(cursor.lastrowid)

    def has_invoice(self, chat_id: int, extracted_data: dict, rnc: str | None = None) -> bool:
        ncf = extracted_data.get("ncf") or extracted_data.get("ncf_modificado")
        if not ncf:
            return False
        
        rnc_scope = self._resolve_rnc(chat_id, rnc)
        data_for_key = dict(extracted_data)
        data_for_key["ncf"] = ncf
        invoice_key = self._invoice_key(data_for_key)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM batch_invoices WHERE chat_id = ? AND rnc = ? AND invoice_key = ?",
                (chat_id, rnc_scope, invoice_key),
            ).fetchone()
        return row is not None

    def get_batch(self, chat_id: int, rnc: str | None = None) -> list[dict]:
        rnc_scope = self._resolve_rnc(chat_id, rnc)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT payload FROM batch_invoices WHERE chat_id = ? AND rnc = ? ORDER BY id",
                (chat_id, rnc_scope),
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def batch_count(self, chat_id: int, rnc: str | None = None) -> int:
        rnc_scope = self._resolve_rnc(chat_id, rnc)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM batch_invoices WHERE chat_id = ? AND rnc = ?",
                (chat_id, rnc_scope),
            ).fetchone()
        return int(row["total"])

    def batch_period(self, chat_id: int, rnc: str | None = None) -> str | None:
        invoices = self.get_batch(chat_id, rnc=rnc)
        if not invoices:
            return None
        return str(invoices[0]["fecha_comprobante"])[:6]

    def save_supplier_memory(
        self,
        rnc_cedula: str,
        tipo_bienes_servicios: str,
        forma_pago: str,
        tipo_asignacion: str = "",
    ) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO supplier_memory (rnc_cedula, tipo_bienes_servicios, forma_pago, tipo_asignacion)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(rnc_cedula) DO UPDATE SET
                    tipo_bienes_servicios=excluded.tipo_bienes_servicios,
                    forma_pago=excluded.forma_pago,
                    tipo_asignacion=CASE
                        WHEN excluded.tipo_asignacion != '' THEN excluded.tipo_asignacion
                        ELSE supplier_memory.tipo_asignacion
                    END,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (rnc_cedula, tipo_bienes_servicios, forma_pago, tipo_asignacion or ""),
            )

    def get_supplier_memory(self, rnc_cedula: str) -> dict[str, str]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT tipo_bienes_servicios, forma_pago, tipo_asignacion FROM supplier_memory WHERE rnc_cedula = ?",
                (rnc_cedula,),
            ).fetchone()
        if row:
            res = {"tipo_bienes_servicios": row["tipo_bienes_servicios"], "forma_pago": row["forma_pago"]}
            if row["tipo_asignacion"]:
                res["tipo_asignacion"] = row["tipo_asignacion"]
            return res
        return {}

    def clear_batch_and_session(self, chat_id: int, rnc: str | None = None) -> None:
        rnc_scope = self._resolve_rnc(chat_id, rnc)
        with self._connection() as connection:
            connection.execute("DELETE FROM sessions WHERE chat_id = ?", (chat_id,))
            connection.execute("DELETE FROM batch_invoices WHERE chat_id = ? AND rnc = ?", (chat_id, rnc_scope))


# Importing the repository creates the database directory.  STATE_DB_PATH is
# intentionally read from the environment so the same module works locally and
# inside Docker without mutable process-global reconfiguration.
state_manager = StateManager(os.getenv("STATE_DB_PATH", "data/state/bot606.sqlite3"))
