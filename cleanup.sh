#!/bin/bash

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[*]${NC} $1"; }
success() { echo -e "${GREEN}[+]${NC} $1"; }
error() { echo -e "${RED}[!]${NC} $1"; }

# Function to clean Python packages
clean_python_packages() {
    log "Cleaning Python packages..."
    
    if [ -n "$VIRTUAL_ENV" ]; then
        # Get list of all installed packages
        pip freeze > packages_to_remove.txt
        
        # Core ML packages
        pip uninstall -y torch torchvision torchaudio
        pip uninstall -y ultralytics mediapipe-silicon face-recognition
        pip uninstall -y opencv-python scikit-learn numpy pandas
        
        # Audio packages
        pip uninstall -y whisper TTS sounddevice PyAudio
        
        # Web packages
        pip uninstall -y fastapi uvicorn python-multipart websockets
        
        # Database packages
        pip uninstall -y sqlalchemy alembic
        
        # Utility packages
        pip uninstall -y python-dotenv pydantic rich typer
        
        # Development packages
        pip uninstall -y pytest black pylint mypy
        
        # Remove remaining packages
        pip uninstall -y -r packages_to_remove.txt
        rm packages_to_remove.txt
        
        # Clean pip cache
        pip cache purge
        success "Python packages removed"
    else
        error "No virtual environment active"
        exit 1
    fi
}

# Function to clean Homebrew packages
clean_homebrew() {
    log "Cleaning Homebrew packages..."
    
    BREW_PACKAGES=(
        "cmake"
        "ffmpeg"
        "portaudio"
        "python@3.10"
    )
    
    for package in "${BREW_PACKAGES[@]}"; do
        if brew list "$package" &>/dev/null; then
            brew uninstall "$package"
            success "Removed $package"
        fi
    done
    
    # Clean Homebrew cache
    brew cleanup --prune=all
    success "Homebrew cleanup complete"
}

# Function to clean system artifacts
clean_system_artifacts() {
    log "Cleaning system artifacts..."
    
    # Clean Python cache
    find . -type f -name "*.pyc" -delete
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
    
    # Clean model caches
    rm -rf ~/.cache/torch
    rm -rf ~/.cache/huggingface
    rm -rf ~/.cache/pip
    
    success "System artifacts cleaned"
}

# Function to clean project files
clean_project_files() {
    log "Cleaning project files..."
    
    # Deactivate virtual environment
    deactivate 2>/dev/null || true
    
    # Remove virtual environment
    rm -rf venv
    
    # Remove project directories
    rm -rf config jarvis data tests scripts build dist
    
    # Remove configuration files
    rm -f .env requirements.txt setup.py .gitignore
    rm -f *.log *.db
    
    success "Project files cleaned"
}

# Main cleanup process
main() {
    log "Starting comprehensive cleanup..."
    
    if [ -n "$VIRTUAL_ENV" ]; then
        clean_python_packages
    else
        error "Please activate virtual environment first"
        exit 1
    fi
    
    clean_homebrew
    clean_system_artifacts
    clean_project_files
    
    success "Cleanup complete!"
}

# Run script
main