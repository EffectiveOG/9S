from TTS.api import TTS
from scipy.signal import butter, lfilter, lfilter_zi
import torch
import numpy as np
from typing import Optional, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
from jarvis.utils.logging_utils import get_logger

logger = get_logger(__name__)

class TTSProcessor:
    """Optimized text-to-speech processor for M1 Mac."""
    
    def __init__(self, model_config: Dict[str, Any], device: str = "cpu"):
        self.config = model_config
        self.device = self._get_optimal_device(device)
        
        try:
            logger.info(f"Initializing TTS on device: {self.device}")
            self.tts = TTS(
                model_name="tts_models/en/ljspeech/fast_pitch",
                progress_bar=False,
                gpu=False
            )
            
            if self.device == "mps":
                self._setup_mps()
                
            self._load_voice_config()
            self._init_cache()
            logger.info("TTS initialized successfully")
            
        except Exception as e:
            logger.error(f"TTS initialization failed: {e}")
            raise

    def _get_optimal_device(self, device: str) -> str:
        """Select optimal processing device for M1."""
        if device == "auto" or device == "mps":
            if torch.backends.mps.is_available():
                return "mps"
        return "cpu"

    def _setup_mps(self):
        """Configure TTS for M1 GPU."""
        try:
            if hasattr(self.tts, 'model'):
                self.tts.model = self.tts.model.to("mps")
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception as e:
            logger.warning(f"MPS setup failed, falling back to CPU: {e}")
            self.device = "cpu"

    async def synthesize(self, text: str, voice: Optional[str] = None) -> Optional[np.ndarray]:
        """Synthesize text to speech."""
        try:
            # Use default voice if none specified
            if not voice and hasattr(self, 'default_speaker'):
                voice = self.default_speaker
                
            # Check cache
            cache_key = f"{text}_{voice}"
            if hasattr(self, 'cache') and cache_key in self.cache:
                return np.array(self.cache[cache_key])
            
            # Generate audio
            with torch.inference_mode():
                wav = self.tts.tts(
                    text=text,
                    speaker=voice,
                    language="en"
                )
                
                if wav is not None:
                    # Process audio
                    audio_data = self._enhance_audio(np.array(wav))
                    
                    # Update cache
                    if hasattr(self, 'cache'):
                        self.cache[cache_key] = audio_data.tolist()
                        self._save_cache()
                        
                    return audio_data
                return None
                
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return None

    def _load_voice_config(self):
        """Load and validate voice configurations."""
        try:
            self.available_speakers = self.tts.speakers
            if self.available_speakers:
                self.default_speaker = self.config.get(
                    "default_voice", 
                    self.available_speakers[0]
                )
                # Voice profiles
                self.voice_profiles = {
                    "default": {
                        "speed": 1.0,
                        "pitch": 1.0
                    },
                    "fast": {
                        "speed": 1.2,
                        "pitch": 1.05
                    },
                    "slow": {
                        "speed": 0.8,
                        "pitch": 0.95
                    }
                }
            else:
                logger.warning("No speakers available")
                self.default_speaker = None
                
        except Exception as e:
            logger.error(f"Error loading voice config: {e}")
            self.default_speaker = None

    def _init_cache(self):
        """Initialize TTS cache."""
        cache_file = Path("data/tts_cache.json")
        try:
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached entries")
        except Exception as e:
            logger.error(f"Cache loading failed: {e}")
            self.cache = {}

    def _save_cache(self):
        """Save TTS cache."""
        try:
            cache_file = Path("data/tts_cache.json")
            with open(cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Cache saving failed: {e}")


    def _synthesize_sync(self, 
                        text: str, 
                        voice: Optional[str],
                        profile: Dict[str, float]) -> Optional[np.ndarray]:
        """Synchronous synthesis with optimizations."""
        try:
            with torch.no_grad():
                wav = self.tts.tts(
                    text=text,
                    speaker=voice,
                    speed=profile["speed"],
                    language="en"
                )
                return np.array(wav)
        except Exception as e:
            logger.error(f"Sync synthesis error: {e}")
            return None

    def _enhance_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Enhance audio quality."""
        try:
            # Normalize
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Apply gentle compression
            threshold = 0.5
            ratio = 0.7
            audio_data = np.where(
                np.abs(audio_data) > threshold,
                threshold + (np.abs(audio_data) - threshold) * ratio * np.sign(audio_data),
                audio_data
            )
            
            # Apply subtle EQ
            if len(audio_data) > 0:
                # High-pass filter to reduce low frequency noise
                b, a = self._butter_highpass(cutoff=100, fs=22050, order=2)
                audio_data = self._apply_filter(b, a, audio_data)
                
                # Presence boost around 3kHz
                b, a = self._butter_bandpass(lowcut=2000, highcut=4000, fs=22050, order=2)
                presence = self._apply_filter(b, a, audio_data)
                audio_data += 0.2 * presence
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Audio enhancement error: {e}")
            return audio_data

    @staticmethod
    def _butter_highpass(cutoff: float, fs: float, order: int = 2):
        """Design Butterworth high-pass filter."""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    @staticmethod
    def _butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 2):
        """Design Butterworth band-pass filter."""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band', analog=False)
        return b, a

    @staticmethod
    def _apply_filter(b: np.ndarray, a: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Apply filter avoiding state blowup."""
        zi = lfilter_zi(b, a)
        filtered_data, _ = lfilter(b, a, data, zi=zi*data[0])
        return filtered_data

    async def validate_voice(self, voice: str) -> bool:
        """Validate voice identifier."""
        return voice in self.available_speakers if self.available_speakers else False

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.device == "mps":
                torch.mps.empty_cache()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")