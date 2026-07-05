# Chatterbox Realtime Voice Chat

A realtime voice chat interface using Chatterbox TTS for voice cloning.

## Architecture

```
Browser (mic) → WebSocket → Server
                              ↓
                         faster-whisper (STT)
                              ↓
                         Ollama/OpenAI (LLM)
                              ↓
                         Chatterbox TTS (voice cloning)
                              ↓
                         WebSocket → Browser (speaker)
```

## Features

- **Push-to-talk** voice input
- **Speech-to-text** with faster-whisper
- **LLM responses** via Ollama or OpenAI
- **Voice cloning** with Chatterbox
- **Paralinguistic tags** support (Turbo model)
- **Web-based UI** - no app installation needed

## Prerequisites

1. **Ollama** running locally with a model:
   ```bash
   # Install Ollama
   brew install ollama

   # Pull a model
   ollama pull llama3.2

   # Start Ollama (runs in background)
   ollama serve
   ```

2. **Reference audio** for voice cloning (optional):
   - 6-15 seconds of clear speech
   - WAV format
   - Use `download_reference.py` to get one from YouTube

## Quick Start

```bash
# From the chatterbox directory
cd realtime_chat

# Run with default voice
./run.sh

# Run with cloned voice
./run.sh ../xiaozhan.wav
```

Then open http://localhost:8000 in your browser.

## Usage

1. **Hold the microphone button** to speak
2. **Release** when done - your speech is transcribed
3. **Wait** for the AI response (spoken in cloned voice)
4. **Repeat** for conversation

## Command Line Options

```bash
python server.py --help

Options:
  --host HOST           Host to bind (default: 0.0.0.0)
  --port PORT           Port to bind (default: 8000)
  --reference PATH      Reference audio for voice cloning
  --turbo / --no-turbo  Use Turbo model (default: turbo)
  --llm-provider        LLM provider: ollama or openai
  --llm-model           LLM model name
```

## Examples

```bash
# Basic usage with Ollama
python server.py --reference ../speaker.wav

# Use OpenAI instead
export OPENAI_API_KEY="your-key"
python server.py --reference ../speaker.wav --llm-provider openai --llm-model gpt-4o-mini

# Use Multilingual model (for non-English)
python server.py --reference ../speaker.wav --no-turbo
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REFERENCE_AUDIO` | Path to reference audio | None |
| `USE_TURBO` | Use Turbo model | true |
| `LLM_PROVIDER` | ollama or openai | ollama |
| `LLM_MODEL` | Model name | llama3.2 |
| `OPENAI_API_KEY` | OpenAI API key | None |

## How It Works

Since Chatterbox doesn't support true streaming TTS, we use **sentence-level streaming**:

1. Buffer LLM output until sentence boundary (`.`, `!`, `?`)
2. Generate audio for complete sentence with Chatterbox
3. Chunk audio into small pieces for streaming playback
4. Client plays chunks as they arrive

This provides a good user experience with typical latency of 300-500ms after each sentence.

## Troubleshooting

### "Ollama not running"
```bash
ollama serve
```

### "No audio from microphone"
- Check browser permissions
- Try Chrome (best WebAudio support)

### "Voice sounds different"
- Ensure reference audio is 6-15 seconds
- Use clear speech with minimal background noise
- Try voice isolation: `python download_reference.py ... --isolate-voice`

### "Slow response time"
- Use Turbo model (faster than Multilingual)
- Use a faster LLM model
- Ensure GPU is being used

## Limitations

- **Not true streaming** - generates audio sentence by sentence
- **Turbo is English-only** - use Multilingual for other languages
- **Requires good reference audio** for quality voice cloning
