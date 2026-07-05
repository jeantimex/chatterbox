#!/bin/bash
set -e

echo "=== Chatterbox TTS Mac Setup ==="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Detected Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" < "3.10" ]]; then
    echo "Error: Python 3.10+ required. Please install via: brew install python@3.11"
    exit 1
fi

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
echo "Output files will be saved to the 'samples/' directory."
