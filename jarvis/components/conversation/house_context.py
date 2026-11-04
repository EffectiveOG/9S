# jarvis/components/conversation/house_context.py
"""Turns the awareness world model into text the LLM can answer from.

Questions like « qui est à la maison ? », « où est Alice ? » or « quelqu'un
est passé devant l'entrée ? » are answered from Frigate + Home Assistant data
(AwarenessComponent.snapshot) and the awareness history kept by memory, never
from the web or the model's imagination.
"""

import re
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

_TEXTS = {
    "fr": {
        # Phrases specific enough not to steal web questions ("il fait froid
        # dehors ?", "prix d'une maison") from the search component.
        "keywords": [
            "à la maison", "a la maison", "dans la maison", "chez nous", "chez moi",
            "qui est là", "quelqu'un", "il y a du monde",
            "devant la porte", "à la porte", "l'entrée", "dans l'allée", "le jardin",
            "le garage", "sont passés", "est venu",
            "est venue", "est arrivé", "est arrivée", "est parti", "est partie",
            "est sorti", "est sortie", "est rentré", "est rentrée", "dernière fois",
            "quelle voiture", "plaque", "les caméras",
        ],
        # "qui est passé" but not "qu'est-ce qui s'est passé" (a news question)
        "patterns": [r"(?<!s')(?<!s’)est passée?\b"],
        "header": (
            "ÉTAT DE LA MAISON (caméras Frigate et Home Assistant) :\n{house}\n"
            "Pour une question sur la maison, réponds uniquement d'après ces "
            "informations ; si elles ne suffisent pas, dis-le simplement."
        ),
        "cameras_off": "Les caméras de la maison ne sont pas disponibles en ce moment.",
        "occupied": "Personnes détectées en ce moment : {areas}.",
        "nobody": "Aucune personne détectée par les caméras en ce moment.",
        "other_objects": "Autres détections en cours : {items}.",
        "person_seen": "{name} : vu(e) {ago}, caméra {camera}{zones}.",
        "vehicle_seen": "Véhicule {plate}{name} : vu {ago}, caméra {camera}.",
        "presence": "{name} (Home Assistant) : {state}.",
        "home": "à la maison",
        "not_home": "pas à la maison",
        "zones": ", zone {zones}",
        "history": "Historique récent :",
        "event": "- {when} : {what}",
        "now": "à l'instant",
        "minutes": "il y a {n} min",
        "hours": "il y a {n} h",
        "days": "il y a {n} jour(s)",
        "events": {
            "person_identified": "{subject} reconnu(e) sur la caméra {camera}",
            "person_moved": "{subject} s'est déplacé(e) vers {camera}{zones}",
            "vehicle_identified": "plaque {subject} lue sur la caméra {camera}",
            "home_presence_update": "{name} : {state}",
            "presence_update": "{count} {label} dans {subject}",
        },
    },
    "en": {
        "keywords": [
            "at home", "in the house", "who's here", "who is here", "who's home",
            "who is home", "anyone", "anybody", "someone", "somebody", "front door",
            "at the door", "driveway", "the garden", "the garage", "came by",
            "has arrived", "arrived home", "has left", "last time", "last seen",
            "which car", "license plate", "the cameras",
        ],
        "patterns": [],
        "header": (
            "HOUSE STATUS (Frigate cameras and Home Assistant):\n{house}\n"
            "For questions about the house, answer only from this information; if "
            "it isn't enough, say so plainly."
        ),
        "cameras_off": "The house cameras are not available right now.",
        "occupied": "People detected right now: {areas}.",
        "nobody": "No person detected by the cameras right now.",
        "other_objects": "Other current detections: {items}.",
        "person_seen": "{name}: seen {ago}, camera {camera}{zones}.",
        "vehicle_seen": "Vehicle {plate}{name}: seen {ago}, camera {camera}.",
        "presence": "{name} (Home Assistant): {state}.",
        "home": "home",
        "not_home": "away",
        "zones": ", zone {zones}",
        "history": "Recent history:",
        "event": "- {when}: {what}",
        "now": "just now",
        "minutes": "{n} min ago",
        "hours": "{n} h ago",
        "days": "{n} day(s) ago",
        "events": {
            "person_identified": "{subject} recognized on camera {camera}",
            "person_moved": "{subject} moved to {camera}{zones}",
            "vehicle_identified": "plate {subject} read on camera {camera}",
            "home_presence_update": "{name}: {state}",
            "presence_update": "{count} {label} in {subject}",
        },
    },
}


def texts(language: str) -> Dict[str, Any]:
    return _TEXTS.get(language, _TEXTS["en"])


def known_subjects(snapshot: Dict[str, Any]) -> Dict[str, str]:
    """Names a question may mention -> history subject to look up.

    People and vehicles map to themselves (their awareness history subject),
    Home Assistant persons map their friendly name to the entity id, and
    Frigate areas map their spoken form ("living room") to the area id.
    """
    subjects: Dict[str, str] = {}
    for name in snapshot.get("people", {}) or {}:
        subjects[name.lower()] = name
    for plate, vehicle in (snapshot.get("vehicles", {}) or {}).items():
        subjects[plate.lower()] = plate
        if vehicle.get("name"):
            subjects[str(vehicle["name"]).lower()] = plate
    for entity_id, presence in (snapshot.get("home_presence", {}) or {}).items():
        name = presence.get("name")
        if name:
            subjects[str(name).lower()] = entity_id
    for area in snapshot.get("areas", {}) or {}:
        subjects[area.lower().replace("_", " ")] = area
    return subjects


def mentioned_subjects(query: str, snapshot: Dict[str, Any]) -> List[str]:
    q = query.lower()
    return list(dict.fromkeys(
        subject for spoken, subject in known_subjects(snapshot).items() if spoken and spoken in q
    ))


def is_house_query(query: str, snapshot: Dict[str, Any], language: str) -> bool:
    q = query.lower()
    t = texts(language)
    if any(k in q for k in t["keywords"]) or any(re.search(p, q) for p in t["patterns"]):
        return True
    return bool(mentioned_subjects(query, snapshot))


def _ago(timestamp: Optional[float], t: Dict[str, Any], now: float) -> str:
    if not timestamp:
        return "?"
    seconds = max(0, now - float(timestamp))
    if seconds < 60:
        return t["now"]
    if seconds < 3600:
        return t["minutes"].format(n=int(seconds // 60))
    if seconds < 86400:
        return t["hours"].format(n=int(seconds // 3600))
    return t["days"].format(n=int(seconds // 86400))


def _presence_state(state: Optional[str], t: Dict[str, Any]) -> str:
    if state == "home":
        return t["home"]
    if state == "not_home":
        return t["not_home"]
    return str(state)


def _zones(zones: Iterable[str], t: Dict[str, Any]) -> str:
    zones = [z for z in zones or [] if z]
    return t["zones"].format(zones=", ".join(zones)) if zones else ""


def describe_house(snapshot: Dict[str, Any], history: List[Dict[str, Any]],
                   language: str = "fr", now: Optional[float] = None) -> str:
    """Compact, speakable description of the house for the system prompt."""
    t = texts(language)
    now = now if now is not None else time.time()
    lines: List[str] = []

    cameras_up = snapshot.get("mqtt_connected") and snapshot.get("frigate_online") is not False
    if not cameras_up:
        lines.append(t["cameras_off"])
    else:
        areas = snapshot.get("areas", {}) or {}
        occupied = [f"{area.replace('_', ' ')} ({counts['person']})"
                    for area, counts in areas.items() if counts.get("person", 0) > 0]
        lines.append(t["occupied"].format(areas=", ".join(occupied)) if occupied else t["nobody"])
        others = [f"{label} — {area.replace('_', ' ')} ({n})"
                  for area, counts in areas.items()
                  for label, n in counts.items() if label not in ("person", "all") and n > 0]
        if others:
            lines.append(t["other_objects"].format(items=", ".join(others)))

    people = sorted((snapshot.get("people", {}) or {}).values(),
                    key=lambda p: p.get("last_seen") or 0, reverse=True)
    for person in people:
        lines.append(t["person_seen"].format(
            name=person.get("name"), ago=_ago(person.get("last_seen"), t, now),
            camera=person.get("camera"), zones=_zones(person.get("zones"), t)))

    vehicles = sorted((snapshot.get("vehicles", {}) or {}).values(),
                      key=lambda v: v.get("last_seen") or 0, reverse=True)
    for vehicle in vehicles:
        lines.append(t["vehicle_seen"].format(
            plate=vehicle.get("plate"),
            name=f" ({vehicle['name']})" if vehicle.get("name") else "",
            ago=_ago(vehicle.get("last_seen"), t, now), camera=vehicle.get("camera")))

    for presence in (snapshot.get("home_presence", {}) or {}).values():
        lines.append(t["presence"].format(name=presence.get("name"),
                                          state=_presence_state(presence.get("state"), t)))

    if history:
        lines.append(t["history"])
        for event in history:
            what = _describe_event(event, t)
            if what:
                lines.append(t["event"].format(when=_when(event.get("timestamp"), t, now), what=what))

    return "\n".join(lines)


def _when(timestamp: Optional[str], t: Dict[str, Any], now: float) -> str:
    try:
        dt = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError):
        return "?"
    return f"{dt.strftime('%d/%m %H:%M')} ({_ago(dt.timestamp(), t, now)})"


def _describe_event(event: Dict[str, Any], t: Dict[str, Any]) -> Optional[str]:
    template = t["events"].get(event.get("event_type"))
    if template is None:
        return None
    data = event.get("data") or {}
    return template.format(
        subject=str(event.get("subject") or "").replace("_", " "),
        camera=event.get("camera") or data.get("camera") or "?",
        zones=_zones(data.get("zones"), t),
        name=data.get("name") or event.get("subject"),
        state=_presence_state(data.get("state"), t),
        count=data.get("count", "?"),
        label=data.get("label", ""),
    )
