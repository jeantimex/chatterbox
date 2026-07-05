#!/bin/bash
set -e

echo "=== Chatterbox TTS Mac Setup ==="

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Install it from https://brew.sh"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Detected Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" < "3.10" ]]; then
    echo "Error: Python 3.10+ required. Please install via: brew install python@3.11"
    exit 1
fi

# Install system dependencies
echo "Checking system dependencies..."
for pkg in ffmpeg yt-dlp pipx; do
    if ! command -v $pkg &> /dev/null; then
        echo "Installing $pkg..."
        brew install $pkg
    else
        echo "  $pkg: OK"
    fi
done

# Create virtual environment
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install chatterbox in editable mode
echo "Installing chatterbox-tts and dependencies..."
pip install -e .

# Create samples directory for output
mkdir -p samples

# Install audio-separator via pipx (isolated, avoids dependency conflicts)
echo ""
echo "Installing audio-separator for voice isolation..."
pipx install "audio-separator[cpu]" 2>/dev/null || pipx upgrade audio-separator 2>/dev/null || true

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test TTS, run:"
echo "  python test_turbo.py"
echo "  python test_multilingual.py"
echo ""
echo "To download a reference voice from YouTube:"
echo "  python download_reference.py 'https://youtube.com/...' -s 0:30 -e 0:40 -o speaker.wav -i"
echo ""
echo "To test voice cloning:"
echo "  python test_voice_cloning.py speaker.wav --lang en,zh"
echo ""
echo "Output files will be saved to the 'samples/' directory."
