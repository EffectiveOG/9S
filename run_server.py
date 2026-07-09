# run_server.py
#
# Thin wrapper kept for convenience. The canonical way to start Jarvis is:
#     python -m jarvis
# Host/port/SSL/reload come from JARVIS_* env vars overlaid on
# config/jarvis_config.json (see jarvis/settings.py).

from jarvis.__main__ import main

if __name__ == "__main__":
    main()
