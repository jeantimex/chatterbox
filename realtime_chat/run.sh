#!/bin/bash
# Realtime Voice Chat with Chatterbox
# Usage: ./run.sh [reference_audio.wav]

set -e

cd "$(dirname "$0")"

# Check if we're in the venv
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "../venv/bin/activate" ]; then
        echo "Activating virtual environment..."
        source ../venv/bin/activate
    else
        echo "Warning: Virtual environment not found. Run mac-setup.sh first."
    fi
fi

# Install additional dependencies if needed
echo "Checking dependencies..."
pip install -q faster-whisper ollama 2>/dev/null || true

# Reference audio
REFERENCE=""
if [ -n "$1" ]; then
    REFERENCE="--reference $1"
fi

echo ""
echo "=== Chatterbox Realtime Voice Chat ==="
echo ""
echo "Starting server at http://localhost:8000"
echo ""
echo "Prerequisites:"
echo "  - Ollama running with a model (e.g., ollama run llama3.2)"
echo "  - Reference audio for voice cloning (optional)"
echo ""

# Start server
python server.py --port 8000 $REFERENCE
