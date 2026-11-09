"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    google_vision_credentials_path: str | None
    state_db_path: Path
    export_directory: Path
    reporting_rnc: str | None
    openai_api_key: str | None = None
    openai_model: str = "gpt-6-luna"


def load_settings() -> Settings:
    missing: list[str] = []

    def get_var(name: str) -> str:
        return os.getenv(name, "").strip()

    token = get_var("TELEGRAM_BOT_TOKEN")
    if not token:
        missing.append("TELEGRAM_BOT_TOKEN")

    openai_key = get_var("OPENAI_API_KEY") or None
    vision_credentials = get_var("GOOGLE_VISION_CREDENTIALS") or None

    if not openai_key and not vision_credentials:
        missing.append("OPENAI_API_KEY o GOOGLE_VISION_CREDENTIALS (se requiere al menos uno para procesar facturas)")

    if missing:
        raise ValueError(
            "Faltan variables obligatorias: " + ", ".join(missing) + ". "
            "Revise .env.example antes de iniciar el bot."
        )

    credentials_path: str | None = None
    if vision_credentials:
        credentials_file = Path(vision_credentials).expanduser()
        if credentials_file.is_file():
            credentials_path = str(credentials_file)
        elif not openai_key:
            raise ValueError(
                f"No existe el archivo configurado en GOOGLE_VISION_CREDENTIALS: {credentials_file}"
            )

    data_directory = Path(os.getenv("APP_DATA_DIR", "data")).expanduser()
    state_db_path = Path(os.getenv("STATE_DB_PATH", str(data_directory / "state" / "bot606.sqlite3"))).expanduser()
    export_directory = Path(os.getenv("EXPORT_DIRECTORY", str(data_directory / "exports"))).expanduser()
    state_db_path.parent.mkdir(parents=True, exist_ok=True)
    export_directory.mkdir(parents=True, exist_ok=True)

    return Settings(
        telegram_bot_token=token,
        google_vision_credentials_path=credentials_path,
        state_db_path=state_db_path,
        export_directory=export_directory,
        reporting_rnc=os.getenv("DGII_REPORTING_RNC", "").strip() or None,
        openai_api_key=openai_key,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-6-luna").strip() or "gpt-6-luna",
    )
