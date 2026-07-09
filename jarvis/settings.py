# jarvis/settings.py
"""Central application settings.

App-level settings (host, port, reload, admin credentials, secret key) are read
from the environment with the JARVIS_ prefix (see .env.example). Component and
device configuration continues to live in config/jarvis_config.json; this module
overlays the two for the web entrypoint.

Uses pydantic-settings when available, with a dependency-free fallback so the
package always imports (e.g. in a slim test environment).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _HAS_PYDANTIC_SETTINGS = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_PYDANTIC_SETTINGS = False


class _SettingsMixin:
    """Shared helpers for both the pydantic and fallback Settings classes."""

    config_path: str
    host: Optional[str]
    port: Optional[int]
    reload: bool
    log_level: str

    def web_config(self) -> Dict[str, Any]:
        """Return the 'web' section of jarvis_config.json (or {})."""
        try:
            with open(self.config_path) as f:
                return (json.load(f) or {}).get("web", {})
        except Exception:
            return {}

    def uvicorn_kwargs(self) -> Dict[str, Any]:
        """Build uvicorn.run kwargs from env settings overlaid on the JSON config."""
        web = self.web_config()
        host = self.host or web.get("host") or "127.0.0.1"
        port = self.port or web.get("port") or 8000
        kwargs: Dict[str, Any] = {
            "host": host,
            "port": int(port),
            "reload": bool(self.reload),
            "log_level": self.log_level,
        }
        ssl = web.get("ssl", {}) or {}
        if ssl.get("enabled"):
            cert, key = ssl.get("cert_file"), ssl.get("key_file")
            if cert and key and Path(cert).exists() and Path(key).exists():
                kwargs["ssl_certfile"] = cert
                kwargs["ssl_keyfile"] = key
        return kwargs


if _HAS_PYDANTIC_SETTINGS:

    class Settings(_SettingsMixin, BaseSettings):
        model_config = SettingsConfigDict(
            env_prefix="JARVIS_", env_file=".env",
            env_file_encoding="utf-8", extra="ignore",
        )

        config_path: str = "config/jarvis_config.json"
        host: Optional[str] = None
        port: Optional[int] = None
        reload: bool = False
        log_level: str = "info"
        admin_user: str = "admin"
        admin_password: Optional[str] = None
        secret_key: Optional[str] = None

else:

    class Settings(_SettingsMixin):  # type: ignore[no-redef]
        """Fallback settings reading os.environ directly."""

        def __init__(self):
            self.config_path = os.getenv("JARVIS_CONFIG_PATH", "config/jarvis_config.json")
            self.host = os.getenv("JARVIS_HOST")
            _port = os.getenv("JARVIS_PORT")
            self.port = int(_port) if _port else None
            self.reload = os.getenv("JARVIS_RELOAD", "false").lower() in ("1", "true", "yes")
            self.log_level = os.getenv("JARVIS_LOG_LEVEL", "info")
            self.admin_user = os.getenv("JARVIS_ADMIN_USER", "admin")
            self.admin_password = os.getenv("JARVIS_ADMIN_PASSWORD")
            self.secret_key = os.getenv("JARVIS_SECRET_KEY")


def get_settings() -> "Settings":
    return Settings()
