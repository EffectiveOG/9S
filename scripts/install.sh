#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
BLUE='\033[0;34m'

# Function to print colored status messages
print_status() {
    echo -e "${BLUE}[*]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[+]${NC} $1"
}

print_error() {
    echo -e "${RED}[!]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if running on MacOS
if [[ "$(uname)" != "Darwin" ]]; then
    print_error "This script is designed for MacOS. Exiting..."
    exit 1
fi

# Check if running on Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    print_warning "This script is optimized for Apple Silicon (M1/M2). Some features may not work optimally on Intel Macs."
fi

# Create project structure
create_project_structure() {
    print_status "Creating project structure..."
    
    directories=(
        "config"
        "jarvis/core"
        "jarvis/components/vision/processors"
        "jarvis/components/vision/models"
        "jarvis/components/audio/processors"
        "jarvis/components/audio/models"
        "jarvis/components/memory/database"
        "jarvis/components/automation/controllers"
        "jarvis/utils"
        "jarvis/plugins"
        "data/known_faces"
        "data/models"
        "data/logs"
        "tests"
        "scripts"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        touch "$dir/__init__.py"
    done
    
    print_success "Project structure created successfully"
}

# Install Homebrew if not present
install_homebrew() {
    if ! command -v brew &> /dev/null; then
        print_status "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
        print_success "Homebrew installed successfully"
    else
        print_status "Homebrew already installed"
    fi
}

# Create requirements.txt with specific versions
create_requirements() {
    print_status "Creating requirements.txt..."
    
    cat > requirements.txt <<EOF
# Core dependencies
numpy==1.24.3
pandas==2.0.3
pillow==10.0.0

# Machine Learning
torch==2.1.0  # will be installed with M1 optimizations
torchvision==0.16.0
ultralytics==8.0.227  # YOLOv8
mediapipe-silicon==0.10.5  # M1 optimized version
face-recognition==1.3.0
opencv-python==4.8.1.78
scikit-learn==1.3.0

# Audio Processing
whisper==1.1.10
TTS==0.21.1
sounddevice==0.4.6
PyAudio==0.2.13

# API and Web
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
websockets==11.0.3

# Database
sqlalchemy==2.0.23
alembic==1.12.1

# Utilities
python-dotenv==1.0.0
pydantic==2.4.2
rich==13.6.0  # For better console output
typer==0.9.0  # For CLI interfaces

# Development
pytest==7.4.3
black==23.10.1
pylint==3.0.2
EOF
    
    print_success "requirements.txt created successfully"
}

# Install system dependencies
install_system_dependencies() {
    print_status "Installing minimal system dependencies..."
    
    # Install only essential system packages
    brew_packages=(
        "cmake"         # Required for building some Python packages
        "ffmpeg"        # Required for audio processing
        "portaudio"     # Required for PyAudio
    )
    
    for package in "${brew_packages[@]}"; do
        if ! brew list "$package" &>/dev/null; then
            brew install "$package"
        else
            print_status "$package already installed"
        fi
    done
    
    print_success "System dependencies installed successfully"
}

# Setup Python environment
setup_python_environment() {
    print_status "Setting up Python environment..."
    
    # Check if Python 3.10 is installed
    if ! command -v python3.10 &> /dev/null; then
        print_status "Installing Python 3.10..."
        brew install python@3.10
    fi
    
    # Create virtual environment
    python3.10 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install PyTorch with M1 optimizations
    pip install --default-timeout=100 torch torchvision torchaudio
    
    # Install other requirements
    pip install -r requirements.txt
    
    print_success "Python environment setup completed"
}

# Download ML models
download_ml_models() {
    print_status "Downloading essential ML models..."
    
    source venv/bin/activate
    
    # Create models directory
    mkdir -p data/models
    
    # Download minimal YOLOv8 model
    python3 - <<EOF
from ultralytics import YOLO
YOLO('yolov8n.pt')  # Download nano model only
EOF
    
    # Download small Whisper model
    python3 - <<EOF
import whisper
model = whisper.load_model("base")
EOF
    
    print_success "Essential ML models downloaded successfully"
}

# Create basic configuration
create_config_files() {
    print_status "Creating configuration files..."
    
    # Create .env file
    cat > .env <<EOF
# Jarvis Configuration
DEBUG=True
LOG_LEVEL=INFO

# Paths
DATA_DIR=data
MODELS_DIR=data/models
KNOWN_FACES_DIR=data/known_faces
LOGS_DIR=data/logs

# Vision
CAMERA_INDEX=0
FRAME_WIDTH=1280
FRAME_HEIGHT=720
FPS=30

# Audio
AUDIO_DEVICE_INDEX=0
SAMPLE_RATE=16000
CHUNK_SIZE=1024

# Database
DATABASE_URL=sqlite:///data/jarvis.db
EOF
    
    print_success "Configuration files created successfully"
}

# Setup git repository
setup_git() {
    print_status "Setting up git repository..."
    
    git init
    
    # Create .gitignore
    cat > .gitignore <<EOF
# Python
__pycache__/
*.py[cod]
*$py.class
venv/
.env

# Data
data/models/*
data/known_faces/*
data/logs/*
*.db

# IDE
.idea/
.vscode/

# OS
.DS_Store
EOF
    
    git add .
    git commit -m "Initial commit"
    
    print_success "Git repository setup completed"
}

# Main installation process
main() {
    print_status "Starting Jarvis installation..."
    
    create_project_structure
    install_homebrew
    install_system_dependencies
    create_requirements
    setup_python_environment
    download_ml_models
    create_config_files
    setup_git
    
    print_success "Jarvis installation completed successfully!"
    print_status "Next steps:"
    echo "1. source venv/bin/activate"
    echo "2. python -m jarvis"
}

# Run main installation
main