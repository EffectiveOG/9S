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
- **Conversation** — free-form spoken (or REST) Q&A with a local LLM served by
  [Ollama](https://ollama.com) (default `qwen2.5:3b`), aware of what the camera
  sees and of who is in front of it; an optional multimodal model (`llava`)
  describes the live frame. Questions needing fresh information (news,
  weather, prices…) trigger a web search (Google Custom Search, DuckDuckGo
  fallback) whose results ground the answer. Ported from the Ciel project.
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
              ┌──────────┐ ┌────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐
              │  Vision  │ │ Audio  │ │  Memory    │ │ Automation │ │ Conversation │
              └────┬─────┘ └───┬────┘ └─────┬──────┘ └─────┬──────┘ └──────┬───────┘
                   │           │            │              │               │
             detectors     Whisper/TTS   SQLite      device controllers  Ollama LLM
          (YOLO/face/gesture)                     (TV / lights / console) (+ llava)
```

Speech routing: an utterance containing the wake word is first matched against
system commands (`stop`, `status`, `restart`) and scenes, then against device
commands (automation). Anything else goes to the conversation component, which
answers through TTS. Vision updates are also fed to the conversation component
so questions like *« Jarvis, qu'est-ce que tu vois ? »* use the current scene.

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
│   ├── automation/  # AutomationComponent, SceneManager, device controllers
│   └── conversation/# ConversationComponent, async Ollama client, web search
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

### Conversation (local LLM)

The conversation component needs an [Ollama](https://ollama.com) server
running locally (it is optional: if Ollama is unreachable Jarvis still starts,
canned answers keep working and an apology is spoken for other questions).

```bash
ollama pull qwen2.5:3b     # text model
ollama pull llava:7b       # optional: describes the camera frame
```

Settings live in the `conversation` section of `config/jarvis_config.json`:

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | `true` | Turn the component off entirely |
| `assistant_name` / `owner_name` | `Jarvis` / `null` | Names used in prompts; the owner is greeted personally when their face is recognised or name spoken |
| `language` | `fr` | `fr` or `en` — prompt, canned answers, fallback messages |
| `wake_word` / `require_wake_word` | `jarvis` / `true` | Only answer utterances that contain the wake word |
| `history_size` | `6` | Exchanges kept as chat context |
| `llm.model` / `llm.vision_model` | `qwen2.5:3b` / `null` | Ollama models (set `vision_model` to `null` to skip frame description) |
| `llm.base_url`, `temperature`, `max_tokens`, `timeout`, `auto_pull` | | Ollama connection and generation options |

#### Web search

Questions that look time-sensitive (*actualités, météo, prix, cours, résultats,
"cherche sur internet"…*, or a year such as 2026) are searched on the web and the
top results are given to the LLM, which summarises them; the source URLs are
returned by `/api/chat` as `last_sources`. Search answers are never cached.

- **Google Custom Search** is used when `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` are
  set in `.env` (create an engine at <https://programmablesearchengine.google.com/>).
- Otherwise, or if Google fails, **DuckDuckGo** is used (no key, needs the
  `ddgs` package from `requirements.txt`).
- With neither available, search is skipped and the LLM answers alone.

Options under `conversation.web_search`: `enabled`, `provider`
(`auto` | `google` | `duckduckgo`), `max_results` (3), `fetch_content` (false —
also download and extract each result page; richer but slower), `timeout`,
`region` (`fr-fr`), `time_limit` (`d` | `w` | `m` | `y`).

For French conversations, also set `"language": "fr"` in the `audio` section so
Whisper transcribes French.

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
| POST   | `/api/chat`                       | yes  | Ask the assistant (`{"text": "...", "speak": false}`) |
| DELETE | `/api/chat/history`               | yes  | Reset conversation memory |
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
