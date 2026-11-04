# Jarvis

A modular, local-first AI home assistant. Jarvis combines real-time computer
vision, speech recognition and synthesis, persistent memory, and smart-home
automation behind a single asynchronous core and a web dashboard.

It is designed to run on a personal machine (developed and tuned for Apple
Silicon / macOS). Rather than re-implementing a home-automation and NVR stack,
Jarvis builds on the ones already running in the house:

- **[Frigate](https://frigate.video)** watches the camera network (object
  detection, face recognition, license plates, zones) — Jarvis consumes its
  results over MQTT.
- **[Home Assistant](https://www.home-assistant.io)** controls the devices and
  knows who is home — Jarvis drives it over its WebSocket API.
- The computer's own **webcam** stays local, for low-latency interaction
  (hand gestures, who is sitting in front of the screen).

> **Status:** early / work-in-progress. The core system, message bus, web API,
> and component scaffolding are functional. See [Known limitations](#known-limitations)
> before relying on it.

---

## Features

- **Vision (local webcam)** — hand-gesture detection and face recognition of the
  person at the computer, with an optional live preview window. Webcam object
  detection (YOLOv8) is still available but off by default: Frigate covers it.
- **Awareness** — a live model of the house built from Frigate (who/what is on
  which camera and zone, recognized faces and plates, per-zone counts) and Home
  Assistant presence entities.
- **Audio** — microphone speech recognition (OpenAI Whisper) and text-to-speech.
- **Memory** — detections and events persisted to a local SQLite database with
  automatic retention/cleanup.
- **Automation** — scene orchestration and device control driven by voice
  commands, gestures, and per-zone occupancy rules. Every device is a Home
  Assistant entity: Jarvis has no device-specific code of its own.
- **Web dashboard** — FastAPI + WebSocket interface for status, metrics, scene
  control, and backups, protected by JWT authentication.

---

## Architecture

```
  IP cameras ──► Frigate ──── MQTT (Mosquitto) ────┐
                                                   ▼
                         ┌──────────────┐   ┌─────────────┐
  Mac webcam ──► Vision ─►  JarvisCore  ◄───┤  Awareness  │◄── HA presence
  Microphone ──► Audio  ─►  message bus │   └─────────────┘     (person.*)
                         └──┬────────┬──┘
                            ▼        ▼
                        Memory    Automation ── WebSocket ──► Home Assistant
                        (SQLite)  (scenes, rules)             (lights, TV, covers…)
```

Each component subclasses `BaseComponent` and publishes `Message` objects. The
core wires producers (vision, audio, awareness) to consumers (memory,
automation) and to its own event loop, so detections, speech and presence
changes flow through the system automatically. Everything is asyncio-based:
the camera network's heavy, parallel work happens inside Frigate, so Jarvis
itself only handles events and stays light.

Awareness publishes these messages on the bus:

| Message                | When                                                    |
|------------------------|---------------------------------------------------------|
| `presence_update`      | an object count changed in a Frigate camera or zone     |
| `object_detected` / `object_ended` | Frigate starts / stops tracking an object   |
| `person_identified`    | a face is recognized (once per tracked object)          |
| `person_moved`         | an identified person changes camera or zone             |
| `vehicle_identified`   | a license plate is read (once per tracked object)       |
| `home_presence_update` | a Home Assistant `person.*` entity changes state        |

Identifications and presence changes are stored in the `awareness_events`
table, so Jarvis can answer "when did I last see…".

---

## Project structure

```
jarvis/
├── core/            # JarvisCore, BaseComponent, Message
├── components/
│   ├── vision/      # VisionComponent + object/face/gesture processors
│   ├── audio/       # AudioComponent + Whisper ASR and TTS
│   ├── memory/      # MemoryComponent + SQLite schema
│   ├── awareness/   # AwarenessComponent (Frigate + HA presence world model)
│   └── automation/  # AutomationComponent, SceneManager, device controllers
├── integrations/    # Home Assistant WebSocket client, Frigate MQTT client
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

> **Do not commit real device tokens or API keys** (Home Assistant token, MQTT password)
> into `config/jarvis_config.json`. Keep secrets in `.env`.

### Home Assistant

Set `HA_URL` and `HA_TOKEN` (a long-lived access token: your HA profile →
Security) in `.env`. Leave either empty to disable the integration.

Declare each device Jarvis should drive in `automation.devices` with type
`home_assistant`. The `id` is the name used by `scenes.json`, rules and voice
commands; scenes keep using the same actions (`power_on`, `set_brightness`,
`set_position`, `set_input`, `set_scene`, …), which are translated to HA
services from the entity's domain:

```json
{"type": "home_assistant", "id": "living_room_lights", "entity_id": "light.salon",
 "scenes": {"movie": "scene.salon_cinema", "default": "scene.salon_normal"}},
{"type": "home_assistant", "id": "living_room_blinds", "entity_id": "cover.salon"},
{"type": "home_assistant", "id": "living_room_tv", "entity_id": "media_player.tv_salon",
 "inputs": {"hdmi1": "HDMI 1", "hdmi2": "HDMI 2"},
 "actions": {"set_game_mode": {"service": "switch.turn_on",
                               "target": {"entity_id": "switch.tv_salon_game_mode"}}}}
```

`config/jarvis_config.json` ships one device per `id` used by `scenes.json`
and the rules, with placeholder entity ids — replace them with yours. Devices
are only available when Home Assistant is configured.
`actions` maps any action to an explicit service call (`value_key` passes the
action's value as a service field). Locks and alarm panels only accept such
explicit `actions`, never the generic on/off/toggle mapping.

### Frigate

Set `MQTT_HOST` (plus `MQTT_USERNAME` / `MQTT_PASSWORD`) to the broker Frigate
publishes to; leave it empty to disable. `FRIGATE_URL` is optional and only
used to build snapshot links. If you changed Frigate's `mqtt.topic_prefix`,
change `frigate.topic_prefix` in `jarvis_config.json` to match. Face and plate
recognition need Frigate 0.16+ with `face_recognition` / `lpr` enabled.

Occupancy rules in `config/automation_rules.json` are bound to a Frigate
camera or zone name with `area` (and optionally `label`, default `person`).
They fire when a count crosses the threshold, not on every update:

```json
{"name": "Lights off when the living room empties", "area": "living_room",
 "conditions": {"max_occupancy": 0},
 "actions": [{"device_id": "living_room_lights", "type": "power_off"}]}
```

Rename `living_room` in the shipped rules to your own Frigate zone or camera.

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
| GET    | `/api/status`                     | yes  | System status            |
| GET    | `/api/components`                 | yes  | Component health         |
| POST   | `/api/command`                    | yes  | Send a command           |
| GET    | `/api/scenes`                     | yes  | List / active scene      |
| POST   | `/api/scenes/{name}/activate`     | yes  | Activate a scene         |
| GET    | `/api/devices`                    | yes  | Device states            |
| GET    | `/api/awareness`                  | yes  | Who/what is where        |
| GET    | `/api/awareness/history`          | yes  | Recent identifications (`?subject=Alice`) |
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
- The Home Assistant and Frigate integrations are covered by tests against a
  simulated HA WebSocket server and an MQTT broker, not yet against a live
  installation. Scene actions with no Home Assistant equivalent (e.g.
  `set_picture_mode`) need an `actions` override per device.
- The vision/audio stacks require host hardware (camera, microphone, and ideally
  a GPU/Apple Silicon), so they don't run in the base Docker image.

---

## License

Released under the [MIT License](LICENSE).
