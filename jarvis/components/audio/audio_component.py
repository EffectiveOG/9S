# jarvis/components/audio/audio_component.py

import asyncio
import numpy as np
import pyaudio
import queue
import time
import torch
import threading
from typing import Dict, Any, Optional, Callable
from ...core.base_component import BaseComponent
from ...core.message import Message
from .processors.speech_recognition import WhisperProcessor
from .processors.text_to_speech import TTSProcessor
from ...utils.logging_utils import get_logger

logger = get_logger(__name__)

class AudioComponent(BaseComponent):
    """Main audio component handling speech recognition and synthesis."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize audio component."""
        super().__init__("audio")
        self.config = config
        
        # Audio settings
        self.sample_rate = config.get("sample_rate", 16000)
        self.chunk_size = config.get("chunk_size", 1024)
        self.channels = config.get("channels", 1)
        
        # Initialize audio interfaces
        self.audio = pyaudio.PyAudio()
        self.input_queue = queue.Queue()
        self.is_listening = False
        self.stream = None
        self.output_stream = None
        self._loop = None            # main event loop, captured in start()
        self.is_speech_detected = False  # read by MetricsCollector
        self._last_transcript = None     # last recognized text (polled by the GUI)
        
        # Voice activity detection
        self.vad_threshold = config.get("vad_threshold", 0.1)  # Adjusted for higher sensitivity
        self.silence_threshold = config.get("silence_threshold", 1.0)
        self.speech_buffer = []
        self.last_speech_time = 0
        
        # Determine best available device for TTS
        self.tts_device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using {self.tts_device} for TTS")
        
        # Initialize processors
        self.asr = WhisperProcessor(
            model_size=config.get("whisper_model", "base"),
            device="cpu",
            language=config.get("language", "en")
        )
        self.tts = TTSProcessor(
            model_config=config.get("tts_model", {}),
            device=self.tts_device
        )
        
        # Callback registry
        self.speech_callbacks = []
        self._initialize_audio_devices()
    
    def _initialize_audio_devices(self):
        """Get and log default audio devices."""
        try:
            self.input_device = self.audio.get_default_input_device_info()
            self.output_device = self.audio.get_default_output_device_info()
            logger.info(f"Input device: {self.input_device['name']}")
            logger.info(f"Output device: {self.output_device['name']}")
        except Exception as e:
            logger.error(f"Error initializing audio devices: {e}")
    
    async def start(self):
        """Start audio processing."""
        try:
            await super().start()  # marks component running so publish() works
            # Capture the running loop so the worker thread can schedule
            # coroutines back onto it safely.
            self._loop = asyncio.get_running_loop()

            # Device indices default to the system default (None) unless the
            # config overrides them - hardcoding 0/1 broke on other machines.
            input_index = self.config.get("input_device_index")
            output_index = self.config.get("output_device_index")

            # Open input stream in callback mode so audio is actually captured
            # into input_queue (previously no callback was registered).
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                output=False,
                frames_per_buffer=self.chunk_size,
                input_device_index=input_index,
                stream_callback=self._audio_callback
            )
            self.output_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=False,
                output=True,
                frames_per_buffer=self.chunk_size,
                output_device_index=output_index
            )
            
            # Start processing
            self.stream.start_stream()
            self.output_stream.start_stream()
            self.is_listening = True
            self.listen_thread = threading.Thread(target=self._process_audio)
            self.listen_thread.start()
            logger.info("Audio component started successfully")
        except Exception as e:
            logger.error(f"Error starting audio component: {e}")
            raise
    
    async def stop(self):
        """Stop audio processing and cleanup."""
        self.is_listening = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        if self.audio:
            self.audio.terminate()
        listen_thread = getattr(self, "listen_thread", None)
        if listen_thread:
            listen_thread.join(timeout=2.0)
        await super().stop()
        logger.info("Audio component stopped")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Handle audio input data."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        logger.debug(f"Audio callback received {len(audio_data)} samples")
        self.input_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def _process_audio(self):
        """Main audio processing loop."""
        while self.is_listening:
            try:
                audio_chunk = self.input_queue.get(timeout=0.1)
                if self._detect_speech(audio_chunk):
                    self.speech_buffer.append(audio_chunk)
                    self.last_speech_time = time.monotonic()
                elif self.speech_buffer:
                    silence_duration = time.monotonic() - self.last_speech_time
                    if silence_duration > self.silence_threshold:
                        if self._loop is not None:
                            asyncio.run_coroutine_threadsafe(
                                self._process_utterance(),
                                self._loop
                            )
                        self.speech_buffer = []
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
    
    def _detect_speech(self, audio_chunk: np.ndarray) -> bool:
        """Detect voice activity in audio chunk."""
        energy = np.mean(np.abs(audio_chunk))
        detected = bool(energy > self.vad_threshold)
        self.is_speech_detected = detected
        return detected
    
    async def _process_utterance(self):
        """Process complete speech utterance."""
        if not self.speech_buffer:
            return
        try:
            audio_data = np.concatenate(self.speech_buffer)
            text = await self.asr.transcribe(audio_data)
            if text:
                logger.info(f"Recognized speech: {text}")
                self._last_transcript = text
                message = Message(
                    sender=self.name,
                    message_type="speech_recognized",
                    data={"text": text}
                )
                await self.publish(message)
                for callback in self.speech_callbacks:
                    callback(text)
        except Exception as e:
            logger.error(f"Error processing utterance: {e}")
    
    async def speak(self, text: str, voice: Optional[str] = None) -> bool:
        """Convert text to speech and play it."""
        try:
            audio_data = await self.tts.synthesize(text, voice)
            if audio_data is not None:
                self.output_stream.write(audio_data.astype(np.float32).tobytes())
                return True
            return False
        except Exception as e:
            logger.error(f"Error during text-to-speech: {e}")
            return False

    async def test_tts(self, text="Hello, this is a test."):
        """Test TTS by synthesizing and playing a test message."""
        await self.speak(text)

    def process_audio(self):
        """Return and clear the last recognized transcript (polled by the GUI)."""
        text = self._last_transcript
        self._last_transcript = None
        return text

    def add_speech_callback(self, callback: Callable[[str], None]):
        """Add callback for speech recognition events."""
        self.speech_callbacks.append(callback)
    
    def remove_speech_callback(self, callback: Callable[[str], None]):
        """Remove speech recognition callback."""
        if callback in self.speech_callbacks:
            self.speech_callbacks.remove(callback)