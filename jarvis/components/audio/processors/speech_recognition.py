import torch
import whisper
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
import asyncio
from queue import Queue
from threading import Thread
from ....utils.logging_utils import get_logger

logger = get_logger(__name__)

@dataclass
class TranscriptionResult:
    """Structured transcription result."""
    text: str
    language: str
    confidence: float
    segments: list
    words: Optional[list] = None

class WhisperProcessor:
    """Optimized Whisper-based speech recognition."""
    
    def __init__(self, 
                 model: str = "base",
                 device: Optional[str] = None,
                 language: str = "en",
                 compute_type: str = "float16",
                 num_workers: int = 2):
        """Initialize Whisper processor with optimizations."""
        self.language = language
        self.compute_type = compute_type
        self.num_workers = num_workers
        self.transcription_queue = Queue()
        self.batch_size = 16 if device == "mps" else 8

        # Select optimal device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        try:
            logger.info(f"Loading Whisper model '{model}' on {device}")
            self.model = whisper.load_model(model)
            
            # Optimize model for inference
            if device == "mps":
                self.model = self._optimize_for_mps(self.model)
            else:
                self.model = self.model.to(device)
                
            if compute_type == "int8":
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                
            self.model.eval()
            logger.info("Model loaded successfully")
            
            # Start worker threads
            self.workers = []
            for _ in range(num_workers):
                worker = Thread(target=self._process_queue, daemon=True)
                worker.start()
                self.workers.append(worker)
                
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def _optimize_for_mps(self, model: whisper.Whisper) -> whisper.Whisper:
        """Optimize model for M1 GPU."""
        try:
            # Convert to MPS format
            model = model.to("mps")
            
            # Enable memory efficient optimizations
            if hasattr(model, "encoder"):
                model.encoder = torch.compile(model.encoder)
            if hasattr(model, "decoder"):
                model.decoder = torch.compile(model.decoder)
                
            return model
            
        except Exception as e:
            logger.warning(f"MPS optimization failed, falling back to CPU: {e}")
            return model.to("cpu")

    async def transcribe(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio data asynchronously."""
        try:
            # Normalize and prepare audio
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
                
            # Add to queue and await result
            future = asyncio.Future()
            self.transcription_queue.put((audio_data, future))
            result = await future
            
            if isinstance(result, TranscriptionResult):
                return result.text
            return None
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    def _process_queue(self):
        """Process queued transcription requests."""
        while True:
            try:
                audio_data, future = self.transcription_queue.get()
                result = self._transcribe_sync(audio_data)
                
                if not future.cancelled():
                    asyncio.run_coroutine_threadsafe(
                        self._set_result(future, result),
                        asyncio.get_event_loop()
                    )
                    
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                
            finally:
                self.transcription_queue.task_done()

    async def _set_result(self, future: asyncio.Future, result: Any):
        """Set future result safely."""
        if not future.cancelled():
            future.set_result(result)

    def _transcribe_sync(self, audio_data: np.ndarray) -> Optional[TranscriptionResult]:
        """Synchronous transcription with optimizations."""
        try:
            with torch.no_grad():
                # Prepare audio features
                mel = self._prepare_audio(audio_data)
                
                # Transcribe with language detection
                result = self.model.transcribe(
                    mel,
                    language=self.language,
                    task="transcribe",
                    fp16=(self.compute_type == "float16"),
                    batch_size=self.batch_size
                )
                
                return TranscriptionResult(
                    text=result["text"],
                    language=result["language"],
                    confidence=self._calculate_confidence(result),
                    segments=result["segments"],
                    words=result.get("words")
                )
                
        except Exception as e:
            logger.error(f"Sync transcription error: {e}")
            return None

    def _prepare_audio(self, audio_data: np.ndarray) -> torch.Tensor:
        """Prepare audio features efficiently."""
        try:
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_data)
            
            # Pad if needed
            if audio_tensor.shape[0] < self.model.dims.n_samples:
                pad_length = self.model.dims.n_samples - audio_tensor.shape[0]
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_length))
                
            # Move to device
            audio_tensor = audio_tensor.to(self.device)
            
            # Get mel spectrogram
            mel = self.model.mel_filters(audio_tensor)
            
            return mel
            
        except Exception as e:
            logger.error(f"Audio preparation error: {e}")
            return None

    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate overall transcription confidence."""
        if "segments" not in result:
            return 0.0
            
        confidences = []
        for segment in result["segments"]:
            if "confidence" in segment:
                confidences.append(segment["confidence"])
                
        return np.mean(confidences) if confidences else 0.0

    def cleanup(self):
        """Clean up resources."""
        try:
            # Clear queue
            while not self.transcription_queue.empty():
                audio_data, future = self.transcription_queue.get_nowait()
                if not future.cancelled():
                    future.cancel()
                self.transcription_queue.task_done()
            
            # Stop workers
            for worker in self.workers:
                worker.join(timeout=1.0)
                
            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}")