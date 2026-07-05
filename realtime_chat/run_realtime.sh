#!/bin/bash
# Realtime Voice Chat - Full Duplex Mode
# Usage: ./run_realtime.sh [reference_audio.wav]

cd "$(dirname "$0")"

if [ -z "$VIRTUAL_ENV" ] && [ -f "../venv/bin/activate" ]; then
    source ../venv/bin/activate
fi

REFERENCE=""
if [ -n "$1" ]; then
    REFERENCE="--reference $1"
fi

echo ""
echo "=== Chatterbox Realtime Voice Chat ==="
echo ""
echo "Features:"
echo "  - Always listening (no button needed)"
echo "  - Interrupt the AI by speaking"
echo "  - Voice cloning with your reference audio"
echo ""
echo "Starting at http://localhost:8000"
echo ""

python server_realtime.py --port 8000 $REFERENCE
