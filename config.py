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
    google_vision_credentials_path: str
    state_db_path: Path
    export_directory: Path
    reporting_rnc: str | None
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"


def load_settings() -> Settings:
    missing: list[str] = []

    def required(name: str) -> str:
        value = os.getenv(name, "").strip()
        if not value:
            missing.append(name)
        return value

    token = required("TELEGRAM_BOT_TOKEN")
    vision_credentials = required("GOOGLE_VISION_CREDENTIALS")
    if missing:
        raise ValueError(
            "Faltan variables obligatorias: " + ", ".join(missing) + ". "
            "Revise .env.example antes de iniciar el bot."
        )

    credentials_file = Path(vision_credentials).expanduser()
    if not credentials_file.is_file():
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
        google_vision_credentials_path=str(credentials_file),
        state_db_path=state_db_path,
        export_directory=export_directory,
        reporting_rnc=os.getenv("DGII_REPORTING_RNC", "").strip() or None,
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip() or None,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
    )
