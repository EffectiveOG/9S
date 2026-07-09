# jarvis/__main__.py
"""Canonical entrypoint: `python -m jarvis`.

Host, port, reload and SSL are resolved from environment (JARVIS_*) overlaid on
the `web` section of config/jarvis_config.json — see jarvis/settings.py.
"""

import os
from pathlib import Path

import uvicorn


def main():
    # Run from the project root so relative paths (config/, data/) resolve.
    os.chdir(Path(__file__).parent.parent)

    # Load .env before importing settings / the app (they read env at import).
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from jarvis.settings import get_settings
    settings = get_settings()
    uvicorn.run("jarvis.web.server:app", **settings.uvicorn_kwargs())


if __name__ == "__main__":
    main()
