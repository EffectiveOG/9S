#!/bin/bash

# Colors for messages
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Starting Jarvis uninstallation...${NC}"

# Deactivate virtual environment if it's active
if [[ -n "$VIRTUAL_ENV" ]]; then
    deactivate
    echo -e "${GREEN}Deactivated virtual environment${NC}"
fi

# Remove virtual environment
if [ -d "venv" ]; then
    rm -rf "venv"
    echo -e "${GREEN}Removed virtual environment${NC}"
fi

# Remove project directories
directories=(
    "config"
    "jarvis"
    "data"
    "tests"
    "scripts"
    "__pycache__"
    ".pytest_cache"
    "build"
    "dist"
)

for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        rm -rf "$dir"
        echo -e "${GREEN}Removed $dir${NC}"
    fi
done

# Remove project files
files=(
    ".env"
    "requirements.txt"
    "setup.py"
    ".gitignore"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        rm -f "$file"
        echo -e "${GREEN}Removed $file${NC}"
    fi
done

# Remove Python cache files
find . -type f -name "*.pyc" -delete
find . -type f -name "*.log" -delete
find . -type d -name "*.egg-info" -exec rm -rf {} +

echo -e "${GREEN}Removed Python cache and log files${NC}"

# Remove git repository if it exists
if [ -d ".git" ]; then
    rm -rf ".git"
    echo -e "${GREEN}Removed Git repository${NC}"
fi

echo -e "${GREEN}Uninstallation complete!${NC}"

# Offer to remove the current directory
read -p "Do you want to remove the current directory? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    parent_dir="$(dirname "$(pwd)")"
    project_dir="$(basename "$(pwd)")"
    cd "$parent_dir" || exit
    rm -rf "$project_dir"
    echo -e "${GREEN}Removed project directory${NC}"
fi