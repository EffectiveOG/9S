import asyncio
from jarvis.components.audio.audio_component import AudioComponent

async def main():
    # Configuration
    config = {
        "sample_rate": 22050,
        "chunk_size": 1024,
        "channels": 1,
        "whisper_model": "base",
        "language": "en",
        "vad_threshold": 0.005,
        "silence_threshold": 1.0
    }
    
    # Initialize audio component
    audio = AudioComponent(config)
    
    try:
        # Start audio processing
        await audio.start()
        print("Audio component started. Running tests...")
        
        # Test TTS
        print("\nTesting text-to-speech...")
        await audio.speak("Hello, this is a test of the text to speech system.")
        
        # Test microphone input
        print("\nTesting microphone input (speak for 5 seconds)...")
        await audio.test_audio_system()
        
        # Wait for a moment
        await asyncio.sleep(10)
        
    except Exception as e:
        print(f"Error during test: {e}")
    
    finally:
        # Cleanup
        await audio.stop()
        print("\nTest completed.")

if __name__ == "__main__":
    asyncio.run(main()) 