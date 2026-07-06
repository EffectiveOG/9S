import os
from pathlib import Path
import torch
from tqdm import tqdm
import whisper
import ssl
import certifi

def setup_directories():
    """Create necessary directories for model storage."""
    dirs = [
        Path("data/models"),
        Path("data/models/whisper"),
        Path("data/models/yolo"),
        Path("data/known_faces")
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def setup_ssl_context():
    """Setup SSL context with proper certificate verification."""
    try:
        # Create SSL context using certifi's certificates
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        return ssl_context
    except Exception as e:
        print(f"Warning: Could not create SSL context: {e}")
        return None

def download_whisper_model(model_size: str = "base"):
    """Download Whisper model using built-in functionality."""
    print(f"Downloading Whisper {model_size} model...")
    try:
        # Set environment variables for SSL certificate verification
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        os.environ['SSL_CERT_FILE'] = certifi.where()
        
        # This will automatically download and cache the model
        model = whisper.load_model(model_size)
        print(f"Whisper {model_size} model downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading Whisper model: {e}")
        print("\nTrying alternative download method...")
        try:
            # Alternative: Install certificates for Python
            import subprocess
            subprocess.run([
                "/Applications/Python 3.11/Install Certificates.command"
            ], shell=True)
            # Try downloading again
            model = whisper.load_model(model_size)
            print(f"Whisper {model_size} model downloaded successfully!")
            return True
        except Exception as e2:
            print(f"Alternative method failed: {e2}")
            print("\nPlease try running the following command in Terminal:")
            print("/Applications/Python 3.11/Install Certificates.command")
            return False

def download_yolo_model():
    """Download YOLOv8 model."""
    print("Downloading YOLOv8 nano model...")
    try:
        from ultralytics import YOLO
        # Set SSL environment variables
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        os.environ['SSL_CERT_FILE'] = certifi.where()
        # This will download the model if it's not already present
        model = YOLO("yolov8n.pt")
        print("YOLOv8 model downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading YOLO model: {e}")
        return False

def check_device_compatibility():
    """Check and print device compatibility information."""
    print("\nDevice Compatibility:")
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.backends.mps.is_available():
        print("✓ M1/M2 GPU (MPS) is available")
        device = "mps"
    elif torch.cuda.is_available():
        print("✓ CUDA GPU is available")
        device = "cuda"
    else:
        print("! Using CPU only")
        device = "cpu"
    
    print(f"Selected device: {device}")
    return device

def main():
    """Main function to download all required models."""
    print("Starting model download process...")
    
    # Create necessary directories
    setup_directories()
    
    # Setup SSL context
    ssl_context = setup_ssl_context()
    if ssl_context:
        print("✓ SSL context created successfully")
    
    # Check device compatibility
    device = check_device_compatibility()
    
    # First ensure certifi is installed
    try:
        import pip
        pip.main(['install', '--upgrade', 'certifi'])
    except Exception as e:
        print(f"Warning: Could not upgrade certifi: {e}")
    
    # Download models
    success = True
    if not download_whisper_model("base"):
        success = False
    if not download_yolo_model():
        success = False
    
    if success:
        print("\nAll models downloaded successfully!")
        print("\nNext steps:")
        print("1. Start Jarvis using 'python -m jarvis'")
        print("2. Check the logs in data/logs for any issues")
        print("3. Configure your devices in config/jarvis_config.json")
    else:
        print("\nSome models failed to download. Please check the errors above and try again.")

if __name__ == "__main__":
    main()