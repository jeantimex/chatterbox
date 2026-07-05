"""
Realtime Voice Chat Server with Chatterbox TTS - Full Duplex Mode

Features:
- Always listening (no push-to-talk)
- Voice Activity Detection (VAD)
- Interruption support (user can interrupt AI)
- Streaming conversation
"""

import asyncio
import base64
import json
import logging
import os
import sys
import time
import threading
from pathlib import Path
from queue import Queue, Empty
from typing import Optional
from collections import deque

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

sys.path.insert(0, str(Path(__file__).parent.parent))

from chatterbox_engine import ChatterboxEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
REFERENCE_AUDIO = os.getenv("REFERENCE_AUDIO", None)
USE_TURBO = os.getenv("USE_TURBO", "true").lower() == "true"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

# Audio settings
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 100
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

# VAD settings
SILENCE_THRESHOLD = 0.01  # RMS threshold for silence
SPEECH_MIN_DURATION = 0.3  # Minimum speech duration (seconds)
SILENCE_DURATION_END = 1.0  # Silence duration to end turn (seconds)

SYSTEM_PROMPT = """You are a helpful voice assistant having a natural conversation.
Keep responses SHORT - 1-2 sentences max. Be conversational and friendly.
Don't use lists, bullet points, or long explanations."""


class SimpleVAD:
    """Simple Voice Activity Detection based on RMS energy."""

    def __init__(self, threshold=SILENCE_THRESHOLD, sample_rate=SAMPLE_RATE):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None

    def process(self, audio_chunk: np.ndarray) -> dict:
        """Process audio chunk and return VAD state."""
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2)) / 32768.0

        current_time = time.time()
        is_speech = rms > self.threshold

        result = {
            "is_speech": is_speech,
            "rms": rms,
            "speech_started": False,
            "speech_ended": False,
        }

        if is_speech:
            self.last_speech_time = current_time
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = current_time
                result["speech_started"] = True
        else:
            if self.is_speaking and self.last_speech_time:
                silence_duration = current_time - self.last_speech_time
                speech_duration = self.last_speech_time - self.speech_start_time

                if silence_duration >= SILENCE_DURATION_END and speech_duration >= SPEECH_MIN_DURATION:
                    self.is_speaking = False
                    result["speech_ended"] = True
                    result["speech_duration"] = speech_duration

        return result


class SpeechToText:
    """Speech-to-text using faster-whisper."""

    def __init__(self, model_size: str = "base", language: str = "en"):
        self.language = language if language != "auto" else None
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading Whisper model: {model_size}, language: {language}")
            self.model = WhisperModel(model_size, device="auto", compute_type="auto")
            logger.info("Whisper model loaded")
        except ImportError:
            logger.warning("faster-whisper not installed")
            self.model = None

    def transcribe(self, audio: np.ndarray) -> str:
        if self.model is None:
            return ""

        audio_float = audio.astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(audio_float, language=self.language, beam_size=5)
        return " ".join([seg.text for seg in segments]).strip()


class LLMClient:
    """LLM client for Ollama with streaming support."""

    def __init__(self, provider: str = "ollama", model: str = "llama3.2"):
        self.provider = provider
        self.model = model
        self.history = []

        try:
            import ollama
            self.client = ollama
        except ImportError:
            logger.error("ollama not installed")
            self.client = None

    def chat(self, user_message: str) -> str:
        if not user_message.strip():
            return ""

        self.history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history[-10:]

        try:
            response = self.client.chat(model=self.model, messages=messages)
            assistant_message = response['message']['content']
            self.history.append({"role": "assistant", "content": assistant_message})
            return assistant_message
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "Sorry, I had trouble understanding."

    def chat_stream(self, user_message: str):
        """Stream LLM response, yielding chunks."""
        if not user_message.strip():
            return

        self.history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history[-10:]

        full_response = ""
        try:
            for chunk in self.client.chat(model=self.model, messages=messages, stream=True):
                content = chunk['message']['content']
                full_response += content
                yield content

            self.history.append({"role": "assistant", "content": full_response})
        except Exception as e:
            logger.error(f"LLM stream error: {e}")
            yield "Sorry, I had trouble understanding."

    def clear(self):
        self.history = []


# Global components
stt: Optional[SpeechToText] = None
llm: Optional[LLMClient] = None
tts: Optional[ChatterboxEngine] = None


def init_components():
    global stt, llm, tts

    # Re-read config from env (may have been set after import)
    reference_audio = os.getenv("REFERENCE_AUDIO", None)
    use_turbo = os.getenv("USE_TURBO", "true").lower() == "true"
    llm_provider = os.getenv("LLM_PROVIDER", "ollama")
    llm_model = os.getenv("LLM_MODEL", "llama3.2")

    # Turbo = English only, Multilingual = auto-detect language
    stt_language = "en" if use_turbo else "auto"

    logger.info("=" * 50)
    logger.info("Initializing Realtime Voice Chat...")
    logger.info(f"Reference audio: {reference_audio}")
    logger.info(f"Use Turbo: {use_turbo}")
    logger.info(f"STT Language: {stt_language}")
    logger.info("=" * 50)

    stt = SpeechToText("base", language=stt_language)
    llm = LLMClient(llm_provider, llm_model)
    tts = ChatterboxEngine(reference_audio=reference_audio, use_turbo=use_turbo)

    logger.info("All components ready!")
    logger.info("=" * 50)


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_components()
    yield
    logger.info("Shutting down")


app = FastAPI(title="Chatterbox Realtime Voice Chat", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def index():
    # Serve realtime.html instead of index.html
    realtime_path = static_dir / "realtime.html"
    if realtime_path.exists():
        return FileResponse(str(realtime_path))
    return HTMLResponse("<h1>Realtime mode - realtime.html not found</h1>")


class ConversationSession:
    """Manages a single conversation session."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.vad = SimpleVAD()
        self.audio_buffer = []
        self.is_ai_speaking = False
        self.should_stop_tts = False
        self.processing_speech = False
        # Client reports when it's playing TTS audio
        self.client_tts_playing = False
        # Thread-safe queue for TTS chunks
        self.tts_chunks = Queue()
        self.tts_done = threading.Event()

    def check_interrupt(self, audio_data: bytes) -> bool:
        """Check if audio contains speech (for interruption). Non-async for speed."""
        audio = np.frombuffer(audio_data, dtype=np.int16)
        vad_result = self.vad.process(audio)

        # If user is speaking while AI is talking, trigger interrupt
        if vad_result["is_speech"] and self.is_ai_speaking:
            if not self.should_stop_tts:
                logger.info("User interrupted AI!")
                self.should_stop_tts = True
                return True
            # Collect for transcription
            self.audio_buffer.append(audio)
        elif not self.processing_speech:
            # Collect audio when user is speaking (and not processing)
            if vad_result["is_speech"] or self.vad.is_speaking:
                self.audio_buffer.append(audio)

        return False

    def should_process_speech(self) -> bool:
        """Check if we should process accumulated speech."""
        if self.processing_speech or not self.audio_buffer:
            return False
        # Check if VAD says speech ended
        return not self.vad.is_speaking and self.vad.last_speech_time is not None

    async def handle_audio(self, audio_data: bytes):
        """Process incoming audio chunk."""
        audio = np.frombuffer(audio_data, dtype=np.int16)
        vad_result = self.vad.process(audio)

        # Log when speech detected during TTS
        if vad_result["is_speech"] and (self.client_tts_playing or self.is_ai_speaking):
            logger.info(f"Speech during TTS! client_tts={self.client_tts_playing}, ai_speaking={self.is_ai_speaking}, should_stop={self.should_stop_tts}")

        # If user starts speaking while TTS is playing on client, interrupt immediately
        if vad_result["is_speech"] and (self.client_tts_playing or self.is_ai_speaking):
            if not self.should_stop_tts:
                logger.info(f">>> INTERRUPT! client_tts={self.client_tts_playing}, ai_speaking={self.is_ai_speaking}")
                self.should_stop_tts = True
                self.client_tts_playing = False
                # Send stop command to client
                await self.ws.send_json({"type": "stop_tts"})
            self.audio_buffer.append(audio)
            return

        if self.processing_speech:
            return

        if vad_result["is_speech"] or self.vad.is_speaking:
            self.audio_buffer.append(audio)

        if vad_result["speech_ended"] and self.audio_buffer:
            asyncio.create_task(self.process_speech())

    async def process_speech(self):
        """Process accumulated speech."""
        if self.processing_speech:
            return

        self.processing_speech = True

        try:
            if not self.audio_buffer:
                return

            audio = np.concatenate(self.audio_buffer)
            self.audio_buffer = []

            logger.info(f"Processing {len(audio)} samples")

            await self.ws.send_json({"type": "status", "text": "Transcribing..."})

            # Run transcription in executor to not block
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, lambda: stt.transcribe(audio) if stt else "")

            if not text.strip():
                logger.info("No speech detected")
                await self.ws.send_json({"type": "status", "text": "Listening..."})
                return

            logger.info(f"User: {text}")
            await self.ws.send_json({"type": "user_text", "text": text})

            # Show typing indicator while thinking + preparing TTS
            await self.ws.send_json({"type": "typing", "show": True})
            await self.ws.send_json({"type": "status", "text": "Thinking..."})

            # Use streaming LLM with sentence-level TTS
            await self.speak_streaming(text)

        finally:
            self.processing_speech = False

    async def speak_streaming(self, user_text: str):
        """Stream LLM response and synthesize sentences in parallel."""
        if not llm or not tts:
            await self.ws.send_json({"type": "status", "text": "Listening..."})
            return

        self.is_ai_speaking = True
        self.should_stop_tts = False
        self._tts_started = False
        self._text_shown = False

        loop = asyncio.get_event_loop()

        # First, collect full LLM response
        def get_llm_chunks():
            return list(llm.chat_stream(user_text))

        chunks = await loop.run_in_executor(None, get_llm_chunks)
        full_response = "".join(chunks)

        logger.info(f"AI: {full_response}")

        # Extract all sentences
        sentences_to_speak = [s.strip() for s in self._extract_sentences(full_response) if s.strip()]

        if not sentences_to_speak:
            await self.ws.send_json({"type": "typing", "show": False})
            await self.ws.send_json({"type": "ai_text", "text": full_response})
            self.is_ai_speaking = False
            await self.ws.send_json({"type": "status", "text": "Listening..."})
            return

        # Parallel synthesis: synthesize next sentence while current plays
        await self.speak_parallel(sentences_to_speak, full_response)

        # Send tts_end if we started
        if self._tts_started:
            await self.ws.send_json({"type": "tts_end"})

        # If text wasn't shown yet, show it now
        if not self._text_shown:
            await self.ws.send_json({"type": "typing", "show": False})
            await self.ws.send_json({"type": "ai_text", "text": full_response})

        self.is_ai_speaking = False
        self._tts_started = False
        self._text_shown = False
        await self.ws.send_json({"type": "status", "text": "Listening..."})

    async def speak_parallel(self, sentences: list, full_response: str):
        """Synthesize sentences in parallel - next sentence while current plays."""
        from concurrent.futures import ThreadPoolExecutor

        # Pre-synthesized audio storage: {index: [chunks]}
        synthesized = {}

        def synthesize_sentence(idx, sentence):
            """Synthesize a sentence in background thread."""
            if self.should_stop_tts:
                return []

            logger.info(f"[Parallel] Starting synthesis {idx+1}: {sentence[:40]}...")
            chunks = []
            q = Queue()
            tts.synthesize(sentence, q)
            while True:
                try:
                    chunk = q.get_nowait()
                    if chunk is None:
                        break
                    chunks.append(chunk)
                except Empty:
                    break
            logger.info(f"[Parallel] Finished synthesis {idx+1}: {len(chunks)} chunks")
            return chunks

        executor = ThreadPoolExecutor(max_workers=2)
        futures = {}

        # Submit first sentence immediately
        if len(sentences) > 0:
            futures[0] = executor.submit(synthesize_sentence, 0, sentences[0])

        # Submit second sentence in parallel with first
        if len(sentences) > 1:
            futures[1] = executor.submit(synthesize_sentence, 1, sentences[1])

        # Process sentences in order
        for i in range(len(sentences)):
            if self.should_stop_tts:
                break

            # Wait for this sentence's synthesis to complete
            if i in futures:
                synthesized[i] = futures[i].result()

            # Start synthesizing sentence i+2 while we stream sentence i
            next_idx = i + 2
            if next_idx < len(sentences) and next_idx not in futures:
                futures[next_idx] = executor.submit(synthesize_sentence, next_idx, sentences[next_idx])

            # Stream current sentence's audio
            if i in synthesized and synthesized[i]:
                chunks = synthesized[i]

                if not self._tts_started:
                    await self.ws.send_json({"type": "tts_start"})
                    self._tts_started = True

                    # Show full response when first audio starts
                    if not self._text_shown:
                        await self.ws.send_json({"type": "typing", "show": False})
                        await self.ws.send_json({"type": "ai_text", "text": full_response})
                        self._text_shown = True

                for chunk in chunks:
                    if self.should_stop_tts:
                        break
                    chunk_b64 = base64.b64encode(chunk).decode('utf-8')
                    await self.ws.send_json({"type": "tts_chunk", "audio": chunk_b64})
                    await asyncio.sleep(0.005)

        executor.shutdown(wait=False)

    def _extract_sentences(self, text: str) -> list:
        """Split text into sentences."""
        import re
        # Split on sentence endings but keep the delimiter
        parts = re.split(r'([.!?]+\s*)', text)
        sentences = []
        current = ""
        for part in parts:
            current += part
            if re.match(r'[.!?]+\s*$', part):
                sentences.append(current)
                current = ""
        if current:
            sentences.append(current)
        return sentences

    async def speak_sentence(self, sentence: str, show_text: str = None):
        """Synthesize and stream a single sentence."""
        if not sentence.strip():
            return

        audio_chunks = []
        self.tts_done.clear()

        def run_synthesis():
            q = Queue()
            tts.synthesize(sentence, q)
            while True:
                try:
                    chunk = q.get_nowait()
                    if chunk is None:
                        break
                    audio_chunks.append(chunk)
                except Empty:
                    break
            self.tts_done.set()

        # Run TTS
        thread = threading.Thread(target=run_synthesis, daemon=True)
        thread.start()

        # Wait with interrupt checks
        while not self.tts_done.is_set():
            if self.should_stop_tts:
                return
            await asyncio.sleep(0.05)

        if self.should_stop_tts:
            return

        # Stream chunks
        if audio_chunks:
            if not self._tts_started:
                await self.ws.send_json({"type": "tts_start"})
                self._tts_started = True

                # Show full AI response when first audio starts
                if show_text and not self._text_shown:
                    await self.ws.send_json({"type": "typing", "show": False})
                    await self.ws.send_json({"type": "ai_text", "text": show_text})
                    self._text_shown = True

            for chunk in audio_chunks:
                if self.should_stop_tts:
                    break
                chunk_b64 = base64.b64encode(chunk).decode('utf-8')
                await self.ws.send_json({"type": "tts_chunk", "audio": chunk_b64})
                await asyncio.sleep(0.005)

    async def speak(self, text: str):
        """Convert text to speech and stream to client."""
        if not tts or not text.strip():
            await self.ws.send_json({"type": "status", "text": "Listening..."})
            return

        self.is_ai_speaking = True
        self.should_stop_tts = False
        self.tts_done.clear()

        audio_chunks = []

        def run_synthesis():
            """Run TTS in background thread."""
            q = Queue()
            logger.info("Starting TTS synthesis in thread...")
            tts.synthesize(text, q)
            logger.info("TTS synthesis complete, collecting chunks...")
            # Collect all chunks from queue
            while True:
                try:
                    chunk = q.get_nowait()
                    if chunk is None:
                        break
                    audio_chunks.append(chunk)
                except Empty:
                    break
            logger.info(f"Collected {len(audio_chunks)} audio chunks")
            self.tts_done.set()

        # Start TTS in thread
        tts_thread = threading.Thread(target=run_synthesis, daemon=True)
        tts_thread.start()

        # Wait for TTS with interrupt checks
        while not self.tts_done.is_set():
            if self.should_stop_tts:
                logger.info("TTS interrupted during synthesis")
                self.is_ai_speaking = False
                await self.ws.send_json({"type": "status", "text": "Listening..."})
                return
            await asyncio.sleep(0.05)

        if self.should_stop_tts:
            logger.info("TTS interrupted before streaming")
            self.is_ai_speaking = False
            await self.ws.send_json({"type": "status", "text": "Listening..."})
            return

        # Stream chunks
        logger.info(f"Streaming {len(audio_chunks)} TTS chunks to client")
        await self.ws.send_json({"type": "tts_start"})

        chunks_sent = 0
        for i, chunk in enumerate(audio_chunks):
            if self.should_stop_tts:
                logger.info(f"TTS interrupted during stream at chunk {i}")
                break

            chunk_b64 = base64.b64encode(chunk).decode('utf-8')
            await self.ws.send_json({"type": "tts_chunk", "audio": chunk_b64})
            chunks_sent += 1
            await asyncio.sleep(0.005)

        logger.info(f"TTS streaming complete, sent {chunks_sent} chunks")
        await self.ws.send_json({"type": "tts_end"})
        self.is_ai_speaking = False
        await self.ws.send_json({"type": "status", "text": "Listening..."})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected")

    session = ConversationSession(ws)

    # Queue for incoming audio
    audio_queue = asyncio.Queue()

    async def receive_audio():
        """Receive audio from websocket."""
        try:
            while True:
                message = await ws.receive()
                if "bytes" in message and message["bytes"]:
                    await audio_queue.put(("audio", message["bytes"]))
                elif "text" in message and message["text"]:
                    await audio_queue.put(("text", message["text"]))
        except:
            pass

    async def process_audio():
        """Process audio from queue."""
        try:
            await ws.send_json({"type": "status", "text": "Listening..."})
            while True:
                msg_type, data = await audio_queue.get()

                if msg_type == "audio":
                    await session.handle_audio(data)
                elif msg_type == "text":
                    try:
                        parsed = json.loads(data)
                        msg_type_inner = parsed.get("type")

                        if msg_type_inner == "clear":
                            if llm:
                                llm.clear()
                            await ws.send_json({"type": "cleared"})

                        elif msg_type_inner == "tts_playing":
                            session.client_tts_playing = parsed.get("playing", False)
                            logger.info(f"Client TTS playing: {session.client_tts_playing}")

                    except:
                        pass
        except:
            pass

    try:
        # Run both tasks concurrently
        await asyncio.gather(
            receive_audio(),
            process_audio(),
        )
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chatterbox Realtime Voice Chat")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reference", "-r", help="Reference audio for voice cloning")
    parser.add_argument("--no-turbo", action="store_true",
                        help="Use Multilingual model (slower but supports all languages)")

    args = parser.parse_args()

    if args.reference:
        os.environ["REFERENCE_AUDIO"] = args.reference
    if args.no_turbo:
        os.environ["USE_TURBO"] = "false"

    mode = "Multilingual (all languages)" if args.no_turbo else "Turbo (English only)"
    logger.info(f"Starting realtime server on {args.host}:{args.port}")
    logger.info(f"Mode: {mode}")
    uvicorn.run(app, host=args.host, port=args.port)
