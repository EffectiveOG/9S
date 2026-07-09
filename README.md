# Jarvis

A modular, local-first AI home assistant. Jarvis combines real-time computer
vision, speech recognition and synthesis, persistent memory, and smart-home
automation behind a single asynchronous core and a web dashboard.

It is designed to run on a personal machine (developed and tuned for Apple
Silicon / macOS) and to control devices on your local network — TVs, smart
lights, and game consoles — via voice, gestures, and room occupancy.

> **Status:** early / work-in-progress. The core system, message bus, web API,
> and component scaffolding are functional. See [Known limitations](#known-limitations)
> before relying on it.

---

## Features

- **Vision** — webcam object detection (YOLOv8), face recognition, and hand-gesture
  detection, with an optional live preview window.
- **Audio** — microphone speech recognition (OpenAI Whisper) and text-to-speech.
- **Memory** — detections and events persisted to a local SQLite database with
  automatic retention/cleanup.
- **Automation** — scene orchestration and device control (smart TV, Philips
  Hue / LIFX lights, PS5 / Xbox) driven by voice commands, gestures, and
  occupancy rules.
- **Web dashboard** — FastAPI + WebSocket interface for status, metrics, scene
  control, and backups, protected by JWT authentication.

---

## Architecture

```
                         ┌──────────────┐
                         │  JarvisCore  │   coordinator + event/command loops
                         └──────┬───────┘
        message bus  ┌──────────┼───────────┐
                     ▼          ▼           ▼
              ┌──────────┐ ┌────────┐ ┌────────────┐ ┌────────────┐
              │  Vision  │ │ Audio  │ │  Memory    │ │ Automation │
              └────┬─────┘ └───┬────┘ └─────┬──────┘ └─────┬──────┘
                   │           │            │              │
             detectors     Whisper/TTS   SQLite      device controllers
          (YOLO/face/gesture)                         (TV / lights / console)
```

Each component subclasses `BaseComponent` and publishes `Message` objects. The
core wires producers (vision, audio) to consumers (memory, automation) and to
its own event loop, so detections and speech flow through the system
automatically. All work is asyncio-based; long-running loops (camera capture,
audio capture, command processing) run as background tasks.

---

## Project structure

```
jarvis/
├── core/            # JarvisCore, BaseComponent, Message
├── components/
│   ├── vision/      # VisionComponent + object/face/gesture processors
│   ├── audio/       # AudioComponent + Whisper ASR and TTS
│   ├── memory/      # MemoryComponent + SQLite schema
│   └── automation/  # AutomationComponent, SceneManager, device controllers
├── web/             # FastAPI server, security (JWT), metrics, backups
└── utils/           # logging helpers
config/              # jarvis_config.json, scenes.json, automation_rules.json
data/                # models, known faces, logs, jarvis.db (git-ignored)
scripts/             # model download / install helpers
tests/               # test suite
```

---

## Requirements

- **Python 3.10+**
- macOS on Apple Silicon is the primary target (`mediapipe-silicon`, Metal/`mps`
  device). Other platforms may require swapping some ML dependencies.
- A webcam and microphone for the vision/audio components.
- `ffmpeg` available on your `PATH` (used by Whisper).

---

## Installation

```bash
# 1. clone and enter the project
git clone <your-repo-url> 9S && cd 9S

# 2. create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. install dependencies
pip install -r requirements.txt

# 4. (optional) download model weights
python scripts/download_models.py

# 5. configure your environment
cp .env.example .env      # then edit values for your machine
```

---

## Configuration

Two layers of configuration:

- **`.env`** — machine-specific settings and secrets (device indices, paths,
  database URL, auth credentials). Copy it from `.env.example`. This file is
  git-ignored. See `.env.example` for every supported variable.
- **`config/jarvis_config.json`** — component behaviour (vision/audio tuning,
  automation devices, scenes, performance limits). `config/scenes.json` and
  `config/automation_rules.json` define scenes and occupancy rules.

> **Do not commit real device tokens or API keys** (Hue keys, PSN/Xbox tokens)
> into `config/jarvis_config.json`. Keep secrets in `.env`.

### Authentication

The dashboard uses JWT auth. On first start an admin account is seeded from
`JARVIS_ADMIN_USER` / `JARVIS_ADMIN_PASSWORD`. If no password is set, a default
`admin` / `admin` is created **with a warning** — set real credentials before
exposing the server. Set `JARVIS_SECRET_KEY` to a fixed value in production so
tokens survive restarts (otherwise a key is generated and stored in
`config/.secret_key`).

---

## Running

```bash
# Recommended — runs the web server on http://localhost:8000
python -m jarvis

# Alternative — binds all interfaces (0.0.0.0:8000); use only on trusted networks
python run_server.py
```

Then open the dashboard at **http://localhost:8000**.

Get an access token:

```bash
curl -X POST http://localhost:8000/token \
  -d "username=admin&password=admin"
```

### Selected API endpoints

| Method | Path                              | Auth | Description               |
|--------|-----------------------------------|------|---------------------------|
| GET    | `/api/status`                     | no   | System status            |
| GET    | `/api/components`                 | no   | Component health         |
| POST   | `/api/command`                    | no   | Send a command           |
| GET    | `/api/scenes`                     | no   | List / active scene      |
| POST   | `/api/scenes/{name}/activate`     | no   | Activate a scene         |
| GET    | `/api/devices`                    | no   | Device states            |
| GET    | `/api/metrics`                    | yes  | System/component metrics |
| POST   | `/api/backup`                     | yes  | Create a backup          |
| WS     | `/ws`                             | yes  | Live state + metrics     |

---

## Development

```bash
pytest              # run the test suite
black .             # format
pylint jarvis       # lint
```

---

## Known limitations

- The PyQt desktop GUI (`jarvis_gui.py`) is experimental. The component
  accessors it needs now exist, but it hasn't been fully validated end-to-end.
- `config/scenes.json` references a `blinds` device type that has no controller
  yet; those actions are skipped gracefully at runtime.
- The vision/audio stacks require host hardware (camera, microphone, and ideally
  a GPU/Apple Silicon), so they don't run in the base Docker image.

---

## License

Released under the [MIT License](LICENSE).
