"""
Realtime Voice Chat Server with Chatterbox TTS

A WebSocket server that:
1. Receives audio from browser
2. Transcribes with Whisper
3. Sends to LLM (Ollama/OpenAI)
4. Generates speech with Chatterbox (voice cloning)
5. Streams audio back to browser
"""

import asyncio
import base64
import json
import logging
import os
import sys
import time
import struct
import threading
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatterbox_engine import ChatterboxEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
REFERENCE_AUDIO = os.getenv("REFERENCE_AUDIO", None)
USE_TURBO = os.getenv("USE_TURBO", "true").lower() == "true"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # ollama or openai
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Sample rate for input audio from browser
INPUT_SAMPLE_RATE = 16000

# System prompt
SYSTEM_PROMPT = """You are a helpful voice assistant. Keep your responses concise and conversational.
Respond naturally as if speaking to someone. Avoid long lists or complex formatting.
Use short sentences that are easy to listen to."""


class SpeechToText:
    """Speech-to-text using faster-whisper."""

    def __init__(self, model_size: str = "base"):
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading Whisper model: {model_size}")
            self.model = WhisperModel(model_size, device="auto", compute_type="auto")
            logger.info("Whisper model loaded")
        except ImportError:
            logger.warning("faster-whisper not installed, using mock STT")
            self.model = None

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio to text."""
        if self.model is None:
            return "[STT not available]"

        # Ensure audio is float32 and normalized
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0

        segments, _ = self.model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=False,  # Disable VAD to avoid filtering out speech
        )

        text = " ".join([seg.text for seg in segments]).strip()
        return text


class LLMClient:
    """LLM client supporting Ollama and OpenAI."""

    def __init__(self, provider: str = "ollama", model: str = "llama3.2"):
        self.provider = provider
        self.model = model
        self.history = []

        if provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=OPENAI_API_KEY)
            except ImportError:
                logger.error("openai package not installed")
                self.client = None
        elif provider == "ollama":
            try:
                import ollama
                self.client = ollama
            except ImportError:
                logger.error("ollama package not installed")
                self.client = None

    def chat(self, user_message: str) -> str:
        """Get response from LLM (non-streaming)."""
        self.history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        try:
            if self.provider == "openai" and self.client:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=150,
                )
                assistant_message = response.choices[0].message.content
            elif self.provider == "ollama" and self.client:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                )
                assistant_message = response['message']['content']
            else:
                assistant_message = "I'm sorry, the LLM is not available."

        except Exception as e:
            logger.error(f"LLM error: {e}")
            assistant_message = "I'm sorry, I encountered an error."

        self.history.append({"role": "assistant", "content": assistant_message})

        # Keep history limited
        if len(self.history) > 20:
            self.history = self.history[-20:]

        return assistant_message

    def chat_stream(self, user_message: str):
        """Get streaming response from LLM."""
        self.history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        full_response = ""

        try:
            if self.provider == "openai" and self.client:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=150,
                    stream=True,
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        full_response += text
                        yield text

            elif self.provider == "ollama" and self.client:
                stream = self.client.chat(
                    model=self.model,
                    messages=messages,
                    stream=True,
                )
                for chunk in stream:
                    if 'message' in chunk and 'content' in chunk['message']:
                        text = chunk['message']['content']
                        full_response += text
                        yield text
            else:
                full_response = "LLM not available."
                yield full_response

        except Exception as e:
            logger.error(f"LLM stream error: {e}")
            full_response = "I encountered an error."
            yield full_response

        self.history.append({"role": "assistant", "content": full_response})

        if len(self.history) > 20:
            self.history = self.history[-20:]

    def clear_history(self):
        """Clear conversation history."""
        self.history = []


# Global instances
stt: Optional[SpeechToText] = None
llm: Optional[LLMClient] = None
tts: Optional[ChatterboxEngine] = None
components_ready = False


def init_components():
    """Initialize global components."""
    global stt, llm, tts, components_ready

    logger.info("=" * 50)
    logger.info("Initializing components...")
    logger.info("=" * 50)

    # Initialize STT
    logger.info("Loading Whisper STT model...")
    stt = SpeechToText(model_size="base")
    logger.info("STT ready!")

    # Initialize LLM
    logger.info(f"Initializing LLM ({LLM_PROVIDER}/{LLM_MODEL})...")
    llm = LLMClient(provider=LLM_PROVIDER, model=LLM_MODEL)
    logger.info("LLM ready!")

    # Initialize TTS
    logger.info(f"Loading Chatterbox TTS (reference={REFERENCE_AUDIO})...")
    tts = ChatterboxEngine(
        reference_audio=REFERENCE_AUDIO,
        use_turbo=USE_TURBO,
    )
    logger.info("TTS ready!")

    components_ready = True
    logger.info("=" * 50)
    logger.info("All components initialized!")
    logger.info("=" * 50)


# Serve static files
static_dir = Path(__file__).parent / "static"

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    init_components()
    yield
    # Shutdown
    logger.info("Server shutting down")

# FastAPI app
app = FastAPI(title="Chatterbox Realtime Voice Chat", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def get_index():
    """Serve the main page."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h1>Chatterbox Realtime Voice Chat</h1><p>Static files not found.</p>")


@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    return {
        "reference_audio": REFERENCE_AUDIO,
        "use_turbo": USE_TURBO,
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "has_reference": REFERENCE_AUDIO is not None and Path(REFERENCE_AUDIO).exists() if REFERENCE_AUDIO else False,
    }


@app.post("/api/set_reference")
async def set_reference(data: dict):
    """Set reference audio for voice cloning."""
    path = data.get("path")
    if path and Path(path).exists():
        if tts:
            tts.set_reference_audio(path)
        return {"status": "ok", "path": path}
    return {"status": "error", "message": "File not found"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Handle WebSocket connections for realtime voice chat."""
    await ws.accept()
    logger.info("Client connected")

    # Check if components are ready
    if not components_ready:
        logger.error("Components not ready!")
        await ws.send_json({"type": "error", "message": "Server not ready, please wait..."})
        await ws.close()
        return

    # Audio buffer for incoming audio
    audio_buffer = []
    is_recording = False

    try:
        while True:
            try:
                message = await ws.receive()
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                break

            if "bytes" in message and message["bytes"]:
                # Binary audio data
                raw = message["bytes"]

                # Parse header if present (8 bytes: timestamp + flags)
                if len(raw) >= 8:
                    timestamp_ms, flags = struct.unpack("!II", raw[:8])
                    pcm_data = raw[8:]
                else:
                    pcm_data = raw

                # Convert to numpy array (16-bit PCM)
                audio_chunk = np.frombuffer(pcm_data, dtype=np.int16)
                audio_buffer.append(audio_chunk)

            elif "text" in message and message["text"]:
                # JSON message
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")

                if msg_type == "start_recording":
                    audio_buffer = []
                    is_recording = True
                    logger.info("Recording started")

                elif msg_type == "stop_recording":
                    is_recording = False
                    logger.info("Recording stopped, waiting for audio data...")

                    # Wait for the audio blob that comes after stop_recording
                    try:
                        audio_msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
                        if "bytes" in audio_msg and audio_msg["bytes"]:
                            raw = audio_msg["bytes"]
                            audio_data = np.frombuffer(raw, dtype=np.int16)
                            logger.info(f"Received {len(audio_data)} audio samples")
                            await process_audio(ws, [audio_data])
                        else:
                            logger.warning("No audio data received after stop_recording")
                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for audio data")

                    audio_buffer = []

                elif msg_type == "clear_history":
                    if llm:
                        llm.clear_history()
                    await ws.send_json({"type": "history_cleared"})

                elif msg_type == "set_reference":
                    path = data.get("path")
                    if path and tts:
                        tts.set_reference_audio(path)
                        await ws.send_json({"type": "reference_set", "path": path})

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket session ended")


async def process_audio(ws: WebSocket, audio_buffer: list):
    """Process recorded audio: STT -> LLM -> TTS -> stream back."""
    if not audio_buffer:
        return

    # Combine audio chunks
    audio = np.concatenate(audio_buffer)
    logger.info(f"Processing {len(audio)} samples of audio")

    # Step 1: Speech to Text
    await ws.send_json({"type": "status", "message": "Transcribing..."})

    if stt:
        transcription = stt.transcribe(audio, INPUT_SAMPLE_RATE)
    else:
        transcription = "[STT not available]"

    logger.info(f"Transcription: {transcription}")
    await ws.send_json({"type": "transcription", "text": transcription})

    if not transcription.strip() or transcription == "[STT not available]":
        return

    # Step 2: LLM Response
    await ws.send_json({"type": "status", "message": "Thinking..."})

    if llm:
        response = llm.chat(transcription)
    else:
        response = "LLM not available."

    logger.info(f"LLM Response: {response}")
    await ws.send_json({"type": "llm_response", "text": response})

    # Step 3: Text to Speech
    await ws.send_json({"type": "status", "message": "Speaking..."})

    if tts:
        audio_queue = Queue()

        # Run TTS synchronously (Chatterbox generates all audio at once)
        logger.info("Starting TTS synthesis...")

        # Run in thread to not block event loop, but wait for completion
        def run_tts():
            tts.synthesize(response, audio_queue)
            audio_queue.put(None)  # End marker

        tts_thread = threading.Thread(target=run_tts, daemon=True)
        tts_thread.start()

        # Wait for synthesis to complete (with timeout)
        tts_thread.join(timeout=60.0)

        if tts_thread.is_alive():
            logger.error("TTS synthesis timeout!")
            await ws.send_json({"type": "error", "message": "TTS timeout"})
        else:
            # Now stream the audio chunks to client
            await ws.send_json({"type": "tts_start"})
            logger.info("TTS streaming started")

            chunk_count = 0
            while True:
                try:
                    chunk = audio_queue.get(timeout=1.0)
                    if chunk is None:
                        logger.info(f"TTS streaming complete, sent {chunk_count} chunks")
                        break

                    chunk_count += 1
                    # Send as base64 encoded audio
                    chunk_b64 = base64.b64encode(chunk).decode('utf-8')
                    await ws.send_json({
                        "type": "tts_chunk",
                        "audio": chunk_b64,
                    })

                    if chunk_count == 1:
                        logger.info(f"First TTS chunk sent ({len(chunk)} bytes)")

                except Empty:
                    logger.info(f"TTS queue empty, sent {chunk_count} chunks")
                    break

            await ws.send_json({"type": "tts_end"})
            logger.info("TTS streaming ended")

    await ws.send_json({"type": "status", "message": "Ready"})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chatterbox Realtime Voice Chat Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reference", "-r", help="Reference audio for voice cloning")
    parser.add_argument("--turbo", action="store_true", default=True, help="Use Turbo model")
    parser.add_argument("--no-turbo", action="store_false", dest="turbo", help="Use Multilingual model")
    parser.add_argument("--llm-provider", default="ollama", choices=["ollama", "openai"])
    parser.add_argument("--llm-model", default="llama3.2")

    args = parser.parse_args()

    # Set environment variables from args
    if args.reference:
        os.environ["REFERENCE_AUDIO"] = args.reference
        REFERENCE_AUDIO = args.reference
    os.environ["USE_TURBO"] = str(args.turbo).lower()
    os.environ["LLM_PROVIDER"] = args.llm_provider
    os.environ["LLM_MODEL"] = args.llm_model

    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Reference audio: {args.reference}")
    logger.info(f"Use Turbo: {args.turbo}")
    logger.info(f"LLM: {args.llm_provider}/{args.llm_model}")

    uvicorn.run(app, host=args.host, port=args.port)
