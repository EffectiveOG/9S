import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from jarvis.components.audio.audio_component import AudioComponent
from jarvis.components.audio.processors.speech_recognition import WhisperProcessor
from jarvis.components.audio.processors.text_to_speech import TTSProcessor

@pytest.fixture
def audio_config():
    return {
        "sample_rate": 22050,
        "chunk_size": 1024,
        "channels": 1,
        "whisper_model": "base",
        "language": "en",
        "vad_threshold": 0.005,
        "silence_threshold": 1.0
    }

@pytest.fixture
async def mock_pyaudio():
    with patch('pyaudio.PyAudio') as mock:
        # Mock audio devices
        mock.return_value.get_default_input_device_info.return_value = {
            'index': 0,
            'name': 'Mock Input Device'
        }
        mock.return_value.get_default_output_device_info.return_value = {
            'index': 1,
            'name': 'Mock Output Device'
        }
        mock.return_value.open.return_value = Mock()
        yield mock

@pytest.mark.asyncio
async def test_audio_initialization(audio_config, mock_pyaudio):
    """Test audio component initialization."""
    with patch('jarvis.components.audio.audio_component.pyaudio.PyAudio', return_value=mock_pyaudio.return_value):
        component = AudioComponent(audio_config)
        await component.start()
        try:
            assert component.sample_rate == 22050
            assert component.chunk_size == 1024
            assert component.channels == 1
            assert isinstance(component.asr, WhisperProcessor)
            assert isinstance(component.tts, TTSProcessor)
        finally:
            await component.stop()

@pytest.mark.asyncio
async def test_speech_detection(audio_config, mock_pyaudio):
    """Test voice activity detection."""
    with patch('jarvis.components.audio.audio_component.pyaudio.PyAudio', return_value=mock_pyaudio.return_value):
        component = AudioComponent(audio_config)
        await component.start()
        try:
            # Test with silence
            silent_audio = np.zeros(1024, dtype=np.float32)
            assert not component._detect_speech(silent_audio)
            
            # Test with speech
            speech_audio = np.random.normal(0, 0.1, 1024).astype(np.float32)
            assert component._detect_speech(speech_audio)
        finally:
            await component.stop()

@pytest.mark.asyncio
async def test_tts_synthesis(audio_config, mock_pyaudio):
    """Test text-to-speech synthesis."""
    with patch('jarvis.components.audio.audio_component.pyaudio.PyAudio', return_value=mock_pyaudio.return_value):
        component = AudioComponent(audio_config)
        await component.start()
        try:
            # Mock TTS synthesize method
            mock_audio = np.random.rand(22050).astype(np.float32)
            with patch.object(component.tts, 'synthesize', new_callable=AsyncMock) as mock_synth:
                mock_synth.return_value = mock_audio
                
                # Test TTS
                test_text = "Hello, this is a test."
                result = await component.speak(test_text)
                
                assert result is True
                mock_synth.assert_called_once_with(test_text, None)
        finally:
            await component.stop()

@pytest.mark.asyncio
async def test_speech_recognition_callback(audio_config, mock_pyaudio):
    """Test speech recognition callback system."""
    with patch('jarvis.components.audio.audio_component.pyaudio.PyAudio', return_value=mock_pyaudio.return_value):
        component = AudioComponent(audio_config)
        await component.start()
        try:
            # Create mock callback
            mock_callback = Mock()
            component.add_speech_callback(mock_callback)
            
            # Mock ASR transcribe method
            test_text = "Hello, this is a test."
            with patch.object(component.asr, 'transcribe', return_value=test_text):
                # Create test audio
                test_audio = np.random.normal(0, 0.1, 22050).astype(np.float32)
                component.speech_buffer = [test_audio]
                
                # Process utterance
                await component._process_utterance()
                
                # Verify callback
                mock_callback.assert_called_once_with(test_text)
        finally:
            await component.stop()

@pytest.mark.asyncio
async def test_audio_processing_pipeline(audio_config, mock_pyaudio):
    """Test the complete audio processing pipeline."""
    with patch('jarvis.components.audio.audio_component.pyaudio.PyAudio', return_value=mock_pyaudio.return_value):
        component = AudioComponent(audio_config)
        await component.start()
        try:
            # Generate test audio
            test_audio = np.random.normal(0, 0.1, component.sample_rate).astype(np.float32)
            
            # Process chunks
            chunk_size = component.chunk_size
            for i in range(0, len(test_audio), chunk_size):
                chunk = test_audio[i:i + chunk_size]
                component.input_queue.put(chunk)
            
            await asyncio.sleep(0.1)
            assert not component.input_queue.empty()
        finally:
            await component.stop()

@pytest.mark.asyncio
async def test_audio_device_initialization(audio_config, mock_pyaudio):
    """Test audio device initialization."""
    with patch('jarvis.components.audio.audio_component.pyaudio.PyAudio', return_value=mock_pyaudio.return_value):
        component = AudioComponent(audio_config)
        await component.start()
        try:
            assert component.input_device is not None
            assert component.output_device is not None
            assert component.input_device_index == 0
            assert component.output_device_index == 1
        finally:
            await component.stop()

@pytest.mark.asyncio
async def test_error_handling(audio_config, mock_pyaudio):
    """Test error handling in audio processing."""
    with patch('jarvis.components.audio.audio_component.pyaudio.PyAudio', return_value=mock_pyaudio.return_value):
        component = AudioComponent(audio_config)
        await component.start()
        try:
            # Test empty speech buffer
            component.speech_buffer = []
            await component._process_utterance()
            
            # Test TTS with empty text
            result = await component.speak("")
            assert result is False
            
            # Test invalid callback
            with pytest.raises(ValueError):
                component.add_speech_callback(None)
            
            # Test invalid audio data
            component.speech_buffer = [np.array([])]
            await component._process_utterance()
        finally:
            await component.stop()

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 