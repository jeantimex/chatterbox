"""
ChatterboxEngine - A wrapper for Chatterbox TTS that provides sentence-level streaming.

Since Chatterbox generates complete audio at once (not true streaming), we:
1. Buffer text until sentence boundaries
2. Generate audio for each sentence
3. Chunk and stream the audio
"""

import logging
import threading
import time
import re
from queue import Queue
from pathlib import Path
from typing import Optional, Callable, Generator
import numpy as np

import torch
import torchaudio as ta

logger = logging.getLogger(__name__)

# Detect device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Patch torch.load for MPS/CPU compatibility
map_location = torch.device(DEVICE)
_torch_load_original = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return _torch_load_original(*args, **kwargs)
torch.load = _patched_torch_load


class ChatterboxEngine:
    """
    Chatterbox TTS engine wrapper for realtime voice chat.

    Provides sentence-level streaming by:
    - Buffering incoming text until sentence boundaries
    - Generating audio for each sentence with Chatterbox
    - Chunking audio for streaming playback
    """

    SAMPLE_RATE = 24000  # Output sample rate for streaming
    CHUNK_DURATION_MS = 100  # Audio chunk duration in milliseconds

    def __init__(
        self,
        reference_audio: Optional[str] = None,
        use_turbo: bool = True,
        device: str = DEVICE,
    ):
        """
        Initialize the Chatterbox engine.

        Args:
            reference_audio: Path to reference audio for voice cloning
            use_turbo: Use Turbo model (faster, English only, paralinguistic tags)
            device: Device to run on (cuda/mps/cpu)
        """
        self.device = device
        self.use_turbo = use_turbo
        self.model = None

        # Resolve reference audio path
        if reference_audio:
            self.reference_audio = str(Path(reference_audio).resolve())
            if not Path(self.reference_audio).exists():
                logger.warning(f"Reference audio not found: {self.reference_audio}")
        else:
            self.reference_audio = None
        self._lock = threading.Lock()

        # Audio chunking parameters
        self.chunk_size = int(self.SAMPLE_RATE * self.CHUNK_DURATION_MS / 1000) * 2  # bytes (16-bit)

        # Callbacks
        self.on_audio_chunk: Optional[Callable[[bytes], None]] = None
        self.on_synthesis_complete: Optional[Callable[[], None]] = None

        # State
        self._stop_event = threading.Event()
        self._text_buffer = ""

        logger.info(f"ChatterboxEngine initializing on {device}, turbo={use_turbo}")
        self._load_model()

    def _load_model(self):
        """Load the Chatterbox model."""
        if self.use_turbo:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            logger.info("Loading Chatterbox Turbo model...")
            self.model = ChatterboxTurboTTS.from_pretrained(device=self.device)
        else:
            from chatterbox.tts import ChatterboxTTS
            logger.info("Loading Chatterbox TTS model...")
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
        logger.info("Chatterbox model loaded!")

        # Prewarm with a short synthesis
        self._prewarm()

    def _prewarm(self):
        """Prewarm the model and cache reference audio embedding."""
        logger.info("Prewarming Chatterbox model...")
        start = time.time()
        try:
            # Cache reference audio embedding once
            if self.reference_audio and hasattr(self.model, 'prepare_conditionals'):
                logger.info(f"Caching reference audio embedding: {self.reference_audio}")
                self.model.prepare_conditionals(self.reference_audio)
                self._conditionals_cached = True
            else:
                self._conditionals_cached = False

            # Prewarm with a short synthesis (no audio_prompt_path since we cached it)
            if self._conditionals_cached:
                _ = self.model.generate("Hello.")
            else:
                _ = self.model.generate("Hello.", audio_prompt_path=self.reference_audio)

            elapsed = time.time() - start
            logger.info(f"Prewarm complete in {elapsed:.2f}s")
        except Exception as e:
            logger.warning(f"Prewarm failed: {e}")
            self._conditionals_cached = False

    def set_reference_audio(self, path: str):
        """Set the reference audio for voice cloning and cache its embedding."""
        # Resolve to absolute path
        path = str(Path(path).resolve())
        if not Path(path).exists():
            logger.warning(f"Reference audio not found: {path}")
            return

        self.reference_audio = path
        logger.info(f"Reference audio set to: {path}")

        # Re-cache the conditionals for the new reference audio
        if hasattr(self.model, 'prepare_conditionals'):
            logger.info("Caching new reference audio embedding...")
            try:
                self.model.prepare_conditionals(path)
                self._conditionals_cached = True
                logger.info("Reference audio embedding cached!")
            except Exception as e:
                logger.warning(f"Failed to cache reference audio: {e}")
                self._conditionals_cached = False

    def _split_sentences(self, text: str) -> list[tuple[str, str]]:
        """
        Split text into sentences, returning (sentence, remainder) tuples.

        Returns list of complete sentences found, plus any remaining text.
        """
        # Sentence boundary pattern
        pattern = r'([.!?]+[\s]*)'

        sentences = []
        parts = re.split(pattern, text)

        current = ""
        for i, part in enumerate(parts):
            current += part
            # If this part is a sentence-ending punctuation
            if re.match(pattern, part):
                sentences.append(current.strip())
                current = ""

        # Return sentences and remainder
        return sentences, current.strip()

    def _audio_to_chunks(self, audio_tensor: torch.Tensor) -> Generator[bytes, None, None]:
        """Convert audio tensor to PCM16 byte chunks for streaming."""
        # Ensure audio is on CPU and convert to numpy
        audio = audio_tensor.squeeze().cpu().numpy()

        # Resample if needed (Chatterbox outputs at model.sr)
        if self.model.sr != self.SAMPLE_RATE:
            import scipy.signal
            num_samples = int(len(audio) * self.SAMPLE_RATE / self.model.sr)
            audio = scipy.signal.resample(audio, num_samples)

        # Normalize to int16 range
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        # Convert to bytes
        audio_bytes = audio_int16.tobytes()

        # Yield chunks
        chunk_bytes = self.chunk_size
        for i in range(0, len(audio_bytes), chunk_bytes):
            if self._stop_event.is_set():
                break
            yield audio_bytes[i:i + chunk_bytes]

    def synthesize(self, text: str, audio_queue: Queue) -> bool:
        """
        Synthesize audio from text and put chunks into queue.

        Args:
            text: Text to synthesize
            audio_queue: Queue to put audio chunks into

        Returns:
            True if completed, False if stopped
        """
        if not text.strip():
            return True

        logger.info(f"Synthesizing: {text[:50]}...")
        logger.info(f"Using cached conditionals: {getattr(self, '_conditionals_cached', False)}")
        start_time = time.time()

        try:
            with self._lock:
                # Generate audio - use cached conditionals if available
                if getattr(self, '_conditionals_cached', False):
                    # Conditionals already cached, no need to pass audio_prompt_path
                    if self.use_turbo:
                        audio = self.model.generate(
                            text,
                            temperature=0.8,
                        )
                    else:
                        audio = self.model.generate(
                            text,
                            exaggeration=0.5,
                            cfg_weight=0.5,
                        )
                else:
                    # No cached conditionals, pass reference audio each time
                    if self.use_turbo:
                        audio = self.model.generate(
                            text,
                            audio_prompt_path=self.reference_audio,
                            temperature=0.8,
                        )
                    else:
                        audio = self.model.generate(
                            text,
                            audio_prompt_path=self.reference_audio,
                            exaggeration=0.5,
                            cfg_weight=0.5,
                        )

            ttfa = time.time() - start_time
            logger.info(f"TTFA: {ttfa:.2f}s for '{text[:30]}...'")

            # Stream chunks
            first_chunk = True
            for chunk in self._audio_to_chunks(audio):
                if self._stop_event.is_set():
                    logger.info("Synthesis stopped by event")
                    return False

                audio_queue.put_nowait(chunk)

                if first_chunk and self.on_audio_chunk:
                    self.on_audio_chunk(chunk)
                    first_chunk = False

            if self.on_synthesis_complete:
                self.on_synthesis_complete()

            return True

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return False

    def synthesize_stream(
        self,
        text_generator: Generator[str, None, None],
        audio_queue: Queue,
    ) -> bool:
        """
        Synthesize audio from streaming text input.

        Buffers text until sentence boundaries, then synthesizes each sentence.

        Args:
            text_generator: Generator yielding text chunks
            audio_queue: Queue to put audio chunks into

        Returns:
            True if completed, False if stopped
        """
        self._text_buffer = ""
        self._stop_event.clear()

        for text_chunk in text_generator:
            if self._stop_event.is_set():
                return False

            self._text_buffer += text_chunk

            # Check for complete sentences
            sentences, remainder = self._split_sentences(self._text_buffer)

            for sentence in sentences:
                if self._stop_event.is_set():
                    return False
                if sentence:
                    success = self.synthesize(sentence, audio_queue)
                    if not success:
                        return False

            self._text_buffer = remainder

        # Synthesize any remaining text
        if self._text_buffer.strip():
            return self.synthesize(self._text_buffer, audio_queue)

        return True

    def stop(self):
        """Stop ongoing synthesis."""
        self._stop_event.set()
        logger.info("Synthesis stop requested")

    def reset(self):
        """Reset the engine state."""
        self._stop_event.clear()
        self._text_buffer = ""


class ChatterboxTTSStream:
    """
    Streaming wrapper that buffers text and synthesizes on sentence boundaries.
    Compatible with the RealtimeTTS TextToAudioStream pattern.
    """

    def __init__(self, engine: ChatterboxEngine):
        self.engine = engine
        self._text_queue: Queue = Queue()
        self._audio_queue: Queue = Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_playing = False

        # Callbacks
        self.on_audio_chunk: Optional[Callable[[bytes], None]] = None
        self.on_stream_stop: Optional[Callable[[], None]] = None

    def feed(self, text):
        """Feed text to the stream (can be string or generator)."""
        if isinstance(text, str):
            self._text_queue.put(text)
        else:
            # Assume generator
            for chunk in text:
                self._text_queue.put(chunk)
        self._text_queue.put(None)  # End marker

    def _worker(self):
        """Background worker that processes text and generates audio."""
        self._is_playing = True
        text_buffer = ""

        try:
            while not self._stop_event.is_set():
                try:
                    text = self._text_queue.get(timeout=0.1)
                except:
                    continue

                if text is None:  # End marker
                    # Synthesize remaining buffer
                    if text_buffer.strip():
                        self.engine.synthesize(text_buffer, self._audio_queue)
                    break

                text_buffer += text

                # Check for complete sentences
                sentences, remainder = self.engine._split_sentences(text_buffer)

                for sentence in sentences:
                    if self._stop_event.is_set():
                        break
                    if sentence:
                        self.engine.synthesize(sentence, self._audio_queue)

                text_buffer = remainder

        finally:
            self._is_playing = False
            if self.on_stream_stop:
                self.on_stream_stop()

    def play_async(self, **kwargs):
        """Start playing asynchronously."""
        self.on_audio_chunk = kwargs.get('on_audio_chunk')
        self.engine.on_audio_chunk = self.on_audio_chunk

        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def play(self, **kwargs):
        """Play synchronously (blocking)."""
        self.play_async(**kwargs)
        if self._worker_thread:
            self._worker_thread.join()

    def stop(self):
        """Stop playback."""
        self._stop_event.set()
        self.engine.stop()
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)

    def is_playing(self) -> bool:
        """Check if currently playing."""
        return self._is_playing

    def get_audio_queue(self) -> Queue:
        """Get the audio output queue."""
        return self._audio_queue
