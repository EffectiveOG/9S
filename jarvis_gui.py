import sys
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import qtmodern.styles
import qtmodern.windows
import cv2
import numpy as np
import asyncio
import qasync
from pathlib import Path
import sounddevice as sd
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jarvis.core.jarvis_core import JarvisCore

class CameraThread(QThread):
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
                # Convert frame to QImage
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.frame_ready.emit(qt_image)
            self.msleep(30)  # ~30 fps

    def stop(self):
        self.running = False
        self.wait()

class AudioThread(QThread):
    text_ready = pyqtSignal(str)

    def __init__(self, audio_component):
        super().__init__()
        self.audio_component = audio_component
        self.running = False

    def run(self):
        self.running = True
        while self.running and self.audio_component:
            if self.audio_component.is_listening:
                text = self.audio_component.process_audio()
                if text:
                    self.text_ready.emit(text)
            self.msleep(100)

    def stop(self):
        self.running = False
        self.wait()

class JarvisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JARVIS")
        self.setup_ui()
        
    def setup_ui(self):
        self.setMinimumSize(1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #000000;
                color: #FFFFFF;
            }
            QWidget {
                background-color: #000000;
                color: #FFFFFF;
                font-family: 'SF Pro Display', Arial;
            }
            QPushButton {
                background-color: #333333;
                border: none;
                border-radius: 20px;
                padding: 15px;
                color: white;
                font-size: 16px;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #444444;
            }
            QPushButton:pressed {
                background-color: #2980b9;
            }
            QLabel {
                font-size: 16px;
                color: #FFFFFF;
            }
            QGroupBox {
                border: 2px solid #333333;
                border-radius: 15px;
                margin-top: 10px;
                padding: 15px;
                font-size: 18px;
            }
            QGroupBox::title {
                color: #FFFFFF;
                subcontrol-position: top center;
                padding: 5px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #333333;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2980b9;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QComboBox {
                background-color: #333333;
                border: none;
                border-radius: 10px;
                padding: 8px;
                color: white;
                min-height: 30px;
            }
            QTextEdit {
                background-color: #222222;
                border: none;
                border-radius: 10px;
                padding: 10px;
                color: white;
            }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left Panel (Controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(20)
        
        # Quick Controls
        quick_controls = QGroupBox("Quick Controls")
        quick_layout = QGridLayout()
        controls = [
            ("CAMERA", "🎥"), ("VOICE", "🎤"),
            ("LIGHTS", "💡"), ("MUSIC", "🎵"),
            ("TV", "📺"), ("GAMING", "🎮")
        ]
        row = 0
        col = 0
        for text, icon in controls:
            btn = QPushButton(f"{icon}\n{text}")
            btn.setFixedSize(150, 100)
            quick_layout.addWidget(btn, row, col)
            col += 1
            if col > 1:
                col = 0
                row += 1
        quick_controls.setLayout(quick_layout)
        left_layout.addWidget(quick_controls)

        # System Status
        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("All Systems Operational")
        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)
        left_layout.addWidget(status_group)

        main_layout.addWidget(left_panel, 1)

        # Right Panel (Display/Preview)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Camera Preview
        self.camera_view = QLabel()
        self.camera_view.setStyleSheet("""
            QLabel {
                background-color: #111111;
                border-radius: 20px;
                padding: 10px;
            }
        """)
        self.camera_view.setMinimumSize(640, 480)
        self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.camera_view)

        # Voice Recognition Display
        self.voice_display = QTextEdit()
        self.voice_display.setPlaceholderText("Voice commands will appear here...")
        self.voice_display.setMaximumHeight(150)
        right_layout.addWidget(self.voice_display)

        main_layout.addWidget(right_panel, 2)

        # Environment Controls
        env_controls = QGroupBox("Environment")
        env_layout = QVBoxLayout()
        
        # Temperature Control
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature"))
        temp_slider = QSlider(Qt.Orientation.Horizontal)
        temp_layout.addWidget(temp_slider)
        env_layout.addLayout(temp_layout)
        
        # Lighting Control
        light_layout = QHBoxLayout()
        light_layout.addWidget(QLabel("Lighting"))
        light_slider = QSlider(Qt.Orientation.Horizontal)
        light_layout.addWidget(light_slider)
        env_layout.addLayout(light_layout)
        
        env_controls.setLayout(env_layout)
        left_layout.addWidget(env_controls)

        # Bottom Status Bar
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #111111;
                color: #FFFFFF;
                padding: 8px;
            }
        """)
        self.statusBar().showMessage("Connected to Home Network")

    def create_sidebar(self):
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setStyleSheet("""
            #sidebar {
                background-color: #2c3e50;
                min-width: 200px;
                padding: 20px;
            }
            QPushButton {
                background-color: transparent;
                border: none;
                color: white;
                text-align: left;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #34495e;
            }
            QPushButton:checked {
                background-color: #3498db;
            }
        """)
        
        layout = QVBoxLayout(sidebar)
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)

        # Navigation buttons
        for text, index in [
            ("Dashboard", 0),
            ("Vision", 1),
            ("Audio", 2),
            ("Automation", 3)
        ]:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, i=index: self.content_stack.setCurrentIndex(i))
            layout.addWidget(btn)

        layout.addStretch()
        return sidebar

    def create_dashboard_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        # System status card
        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout(status_group)
        self.status_label = QLabel("Initializing...")
        status_layout.addWidget(self.status_label)
        layout.addWidget(status_group)

        # Quick actions card
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QHBoxLayout(actions_group)
        for name in ["Movie Mode", "Gaming Mode", "Evening Mode"]:
            btn = QPushButton(name)
            actions_layout.addWidget(btn)
        layout.addWidget(actions_group)

        self.content_stack.addWidget(page)

    def create_vision_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        # Camera preview
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.camera_label)

        # Controls
        controls = QHBoxLayout()
        self.camera_toggle = QPushButton("Start Camera")
        self.camera_toggle.setCheckable(True)
        self.camera_toggle.toggled.connect(self.toggle_camera)
        controls.addWidget(self.camera_toggle)

        # Detection toggles
        for text in ["Object Detection", "Face Recognition", "Gesture Detection"]:
            toggle = QCheckBox(text)
            controls.addWidget(toggle)

        layout.addLayout(controls)
        self.content_stack.addWidget(page)

    def create_audio_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        # Speech recognition area
        recognition_group = QGroupBox("Speech Recognition")
        recognition_layout = QVBoxLayout(recognition_group)
        
        self.speech_text = QTextEdit()
        self.speech_text.setReadOnly(True)
        recognition_layout.addWidget(self.speech_text)

        # Audio controls
        controls = QHBoxLayout()
        self.mic_toggle = QPushButton("Start Listening")
        self.mic_toggle.setCheckable(True)
        self.mic_toggle.toggled.connect(self.toggle_microphone)
        controls.addWidget(self.mic_toggle)

        # TTS section
        tts_group = QGroupBox("Text to Speech")
        tts_layout = QVBoxLayout(tts_group)
        
        self.tts_input = QTextEdit()
        self.tts_input.setPlaceholderText("Enter text to speak...")
        tts_layout.addWidget(self.tts_input)

        speak_btn = QPushButton("Speak")
        speak_btn.clicked.connect(self.speak_text)
        tts_layout.addWidget(speak_btn)

        recognition_layout.addLayout(controls)
        layout.addWidget(recognition_group)
        layout.addWidget(tts_group)
        
        self.content_stack.addWidget(page)

    def create_automation_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        # Scenes control
        scenes_group = QGroupBox("Scenes")
        scenes_layout = QVBoxLayout(scenes_group)
        
        self.scenes_combo = QComboBox()
        self.scenes_combo.addItems(["Default", "Movie Night", "Gaming", "Evening"])
        scenes_layout.addWidget(self.scenes_combo)

        activate_btn = QPushButton("Activate Scene")
        scenes_layout.addWidget(activate_btn)
        layout.addWidget(scenes_group)

        # Devices control
        devices_group = QGroupBox("Devices")
        devices_layout = QGridLayout(devices_group)
        
        # Add device controls
        devices = [
            ("TV", ["Power", "Input Source", "Volume"]),
            ("Lights", ["Power", "Brightness", "Color"]),
            ("Game Console", ["Power", "Game Mode"])
        ]

        for i, (device, controls) in enumerate(devices):
            devices_layout.addWidget(QLabel(device), i, 0)
            for j, control in enumerate(controls):
                if control == "Power":
                    btn = QPushButton("ON/OFF")
                    devices_layout.addWidget(btn, i, j+1)
                else:
                    combo = QComboBox()
                    combo.addItems([f"{control} {n}" for n in range(1, 4)])
                    devices_layout.addWidget(combo, i, j+1)

        layout.addWidget(devices_group)
        self.content_stack.addWidget(page)

    def setup_threads(self):
        if self.jarvis and "vision" in self.jarvis.components:
            self.camera_thread = CameraThread(self.jarvis.components["vision"])
            self.camera_thread.frame_ready.connect(self.update_camera_feed)

        if self.jarvis and "audio" in self.jarvis.components:
            self.audio_thread = AudioThread(self.jarvis.components["audio"])
            self.audio_thread.text_ready.connect(self.update_speech_text)

    @pyqtSlot(QImage)
    def update_camera_feed(self, image):
        scaled_image = image.scaled(self.camera_label.size(), 
                                  Qt.AspectRatioMode.KeepAspectRatio)
        self.camera_label.setPixmap(QPixmap.fromImage(scaled_image))

    @pyqtSlot(str)
    def update_speech_text(self, text):
        self.speech_text.append(text)

    def toggle_camera(self, enabled):
        if enabled:
            if self.camera_thread:
                self.camera_thread.start()
                self.camera_toggle.setText("Stop Camera")
        else:
            if self.camera_thread:
                self.camera_thread.stop()
                self.camera_toggle.setText("Start Camera")
                self.camera_label.clear()

    def toggle_microphone(self, enabled):
        if enabled:
            if self.audio_thread:
                self.audio_thread.start()
                self.mic_toggle.setText("Stop Listening")
        else:
            if self.audio_thread:
                self.audio_thread.stop()
                self.mic_toggle.setText("Start Listening")

    def speak_text(self):
        text = self.tts_input.toPlainText()
        if text and self.jarvis and "audio" in self.jarvis.components:
            asyncio.create_task(self.jarvis.components["audio"].speak(text))

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
        if self.audio_thread:
            self.audio_thread.stop()
        event.accept()

async def main():
    app = QApplication(sys.argv)
    
    # Set Fusion style as base
    app.setStyle("Fusion")
    
    # Set dark palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(0, 0, 0))
    dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(dark_palette)
    
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    window = JarvisGUI()
    window.show()
    
    with loop:
        await loop.run_forever()

if __name__ == "__main__":
    asyncio.run(main())