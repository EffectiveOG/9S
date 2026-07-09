# jarvis_gui.py
"""PyQt6 desktop front-end for Jarvis.

Run with:  python jarvis_gui.py
Requires the optional GUI extras:  pip install ".[gui]"  (PyQt6, qasync).

The GUI starts a JarvisCore in the background (via the qasync event loop),
displays the camera feed and recognized speech, and lets you activate scenes
and speak text. It degrades gracefully if the vision/audio components fail to
start (e.g. no camera/microphone available).
"""

import asyncio
import os
import sys

import cv2
import qasync
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QImage, QPalette, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

# Ensure the project root (this file's directory) is importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from jarvis.core.jarvis_core import JarvisCore
from jarvis.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CameraThread(QThread):
    """Polls the vision component's latest frame and emits it as a QImage."""

    frame_ready = pyqtSignal(QImage)

    def __init__(self, vision_component):
        super().__init__()
        self.vision_component = vision_component
        self.running = False

    def run(self):
        self.running = True
        while self.running and self.vision_component:
            frame = self.vision_component.get_frame()
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                # .copy() so the QImage owns its buffer (rgb is freed each loop).
                qt_image = QImage(rgb.data, w, h, ch * w,
                                  QImage.Format.Format_RGB888).copy()
                self.frame_ready.emit(qt_image)
            self.msleep(33)  # ~30 fps

    def stop(self):
        self.running = False
        self.wait()


class AudioThread(QThread):
    """Polls the audio component for newly recognized speech."""

    text_ready = pyqtSignal(str)

    def __init__(self, audio_component):
        super().__init__()
        self.audio_component = audio_component
        self.running = False

    def run(self):
        self.running = True
        while self.running and self.audio_component:
            if getattr(self.audio_component, "is_listening", False):
                text = self.audio_component.process_audio()
                if text:
                    self.text_ready.emit(text)
            self.msleep(100)

    def stop(self):
        self.running = False
        self.wait()


STYLE = """
QMainWindow, QWidget { background-color: #0b0b0b; color: #ffffff;
    font-family: 'SF Pro Display', Arial; }
QPushButton { background-color: #333333; border: none; border-radius: 12px;
    padding: 12px; color: white; font-size: 15px; min-height: 20px; }
QPushButton:hover { background-color: #444444; }
QPushButton:pressed, QPushButton:checked { background-color: #2980b9; }
QGroupBox { border: 2px solid #333333; border-radius: 12px; margin-top: 10px;
    padding: 12px; font-size: 16px; }
QGroupBox::title { color: #ffffff; subcontrol-position: top center; padding: 4px; }
QComboBox { background-color: #333333; border: none; border-radius: 8px;
    padding: 6px; color: white; min-height: 26px; }
QTextEdit { background-color: #1a1a1a; border: none; border-radius: 8px;
    padding: 8px; color: white; }
"""


class JarvisGUI(QMainWindow):
    def __init__(self, jarvis: JarvisCore):
        super().__init__()
        self.jarvis = jarvis
        self.camera_thread = None
        self.audio_thread = None
        self.setWindowTitle("JARVIS")
        self.setMinimumSize(1100, 720)
        self.setStyleSheet(STYLE)
        self._build_ui()
        self._setup_threads()

    # ---- UI --------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Left: controls
        left = QVBoxLayout()
        left.setSpacing(16)

        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout(status_group)
        self.status_label = QLabel("Starting…")
        status_layout.addWidget(self.status_label)
        left.addWidget(status_group)

        scenes_group = QGroupBox("Scenes")
        scenes_layout = QVBoxLayout(scenes_group)
        self.scenes_combo = QComboBox()
        scenes_layout.addWidget(self.scenes_combo)
        activate_btn = QPushButton("Activate scene")
        activate_btn.clicked.connect(self.activate_scene)
        scenes_layout.addWidget(activate_btn)
        left.addWidget(scenes_group)

        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)
        self.camera_toggle = QPushButton("Start camera preview")
        self.camera_toggle.setCheckable(True)
        self.camera_toggle.toggled.connect(self.toggle_camera)
        controls_layout.addWidget(self.camera_toggle)
        self.mic_toggle = QPushButton("Start listening")
        self.mic_toggle.setCheckable(True)
        self.mic_toggle.toggled.connect(self.toggle_microphone)
        controls_layout.addWidget(self.mic_toggle)
        left.addWidget(controls_group)
        left.addStretch()

        # Right: camera + voice + TTS
        right = QVBoxLayout()
        self.camera_view = QLabel("Camera preview")
        self.camera_view.setStyleSheet(
            "QLabel { background-color: #111111; border-radius: 12px; padding: 8px; }")
        self.camera_view.setMinimumSize(640, 480)
        self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(self.camera_view)

        self.voice_display = QTextEdit()
        self.voice_display.setReadOnly(True)
        self.voice_display.setPlaceholderText("Recognized speech will appear here…")
        self.voice_display.setMaximumHeight(120)
        right.addWidget(self.voice_display)

        tts_row = QHBoxLayout()
        self.tts_input = QTextEdit()
        self.tts_input.setPlaceholderText("Enter text to speak…")
        self.tts_input.setMaximumHeight(60)
        tts_row.addWidget(self.tts_input)
        speak_btn = QPushButton("Speak")
        speak_btn.clicked.connect(self.speak_text)
        tts_row.addWidget(speak_btn)
        right.addLayout(tts_row)

        root.addLayout(left, 1)
        root.addLayout(right, 2)

        self.statusBar().showMessage("Jarvis desktop")
        self._populate_scenes()

    def _populate_scenes(self):
        automation = self.jarvis.components.get("automation") if self.jarvis else None
        scene_manager = getattr(automation, "scene_manager", None)
        if scene_manager is not None and scene_manager.scenes:
            self.scenes_combo.addItems(sorted(scene_manager.scenes.keys()))

    def _setup_threads(self):
        vision = self.jarvis.components.get("vision") if self.jarvis else None
        if vision is not None:
            self.camera_thread = CameraThread(vision)
            self.camera_thread.frame_ready.connect(self.update_camera_feed)
        audio = self.jarvis.components.get("audio") if self.jarvis else None
        if audio is not None:
            self.audio_thread = AudioThread(audio)
            self.audio_thread.text_ready.connect(self.update_speech_text)

    # ---- slots -----------------------------------------------------------
    @pyqtSlot(QImage)
    def update_camera_feed(self, image):
        scaled = image.scaled(self.camera_view.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.camera_view.setPixmap(QPixmap.fromImage(scaled))

    @pyqtSlot(str)
    def update_speech_text(self, text):
        self.voice_display.append(text)

    def toggle_camera(self, enabled):
        if not self.camera_thread:
            self.statusBar().showMessage("Vision component not available")
            return
        if enabled:
            self.camera_thread.start()
            self.camera_toggle.setText("Stop camera preview")
        else:
            self.camera_thread.stop()
            self.camera_toggle.setText("Start camera preview")
            self.camera_view.clear()
            self.camera_view.setText("Camera preview")

    def toggle_microphone(self, enabled):
        if not self.audio_thread:
            self.statusBar().showMessage("Audio component not available")
            return
        if enabled:
            self.audio_thread.start()
            self.mic_toggle.setText("Stop listening")
        else:
            self.audio_thread.stop()
            self.mic_toggle.setText("Start listening")

    def activate_scene(self):
        scene = self.scenes_combo.currentText()
        if not scene or not self.jarvis:
            return
        asyncio.ensure_future(self.jarvis.command_queue.put(
            {"type": "scene_control", "data": {"action": "activate", "scene": scene}}
        ))
        self.statusBar().showMessage(f"Activating scene: {scene}")

    def speak_text(self):
        text = self.tts_input.toPlainText().strip()
        audio = self.jarvis.components.get("audio") if self.jarvis else None
        if text and audio is not None:
            asyncio.ensure_future(audio.speak(text))
            self.tts_input.clear()

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
        if self.audio_thread:
            self.audio_thread.stop()
        if self.jarvis:
            asyncio.ensure_future(self.jarvis.stop())
        event.accept()


def _apply_dark_palette(app):
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(11, 11, 11))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    _apply_dark_palette(app)

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    jarvis = JarvisCore("config/jarvis_config.json")
    # The GUI renders the camera itself; disable the component's cv2 preview window.
    jarvis.config.setdefault("vision", {})["show_preview"] = False

    window = JarvisGUI(jarvis)
    window.show()

    with loop:
        loop.create_task(jarvis.start())
        loop.run_forever()


if __name__ == "__main__":
    main()
