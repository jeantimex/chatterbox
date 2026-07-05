#!/usr/bin/env python3
"""
Test script for Chatterbox Turbo TTS (English, low-latency)
Supports Mac MPS, CUDA, and CPU devices.
"""

import torch
import torchaudio as ta
from pathlib import Path

# Detect best available device for Mac
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Patch torch.load for MPS/CPU compatibility
map_location = torch.device(device)
torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)
torch.load = patched_torch_load

from chatterbox.tts_turbo import ChatterboxTurboTTS

print("Loading Chatterbox Turbo model...")
model = ChatterboxTurboTTS.from_pretrained(device=device)
print("Model loaded successfully!")

# Output directory
output_dir = Path("samples")
output_dir.mkdir(exist_ok=True)

# Test 1: Basic TTS
print("\n--- Test 1: Basic TTS ---")
text1 = "Hello! This is a test of the Chatterbox Turbo text to speech system. How does it sound?"
wav1 = model.generate(text1)
output_path1 = output_dir / "turbo_basic.wav"
ta.save(str(output_path1), wav1, model.sr)
print(f"Saved: {output_path1}")

# Test 2: TTS with paralinguistic tags
print("\n--- Test 2: TTS with Paralinguistic Tags ---")
text2 = "Oh wow, that's amazing! [laugh] I can't believe it actually works. [chuckle] Let me try something else."
wav2 = model.generate(text2)
output_path2 = output_dir / "turbo_paralinguistic.wav"
ta.save(str(output_path2), wav2, model.sr)
print(f"Saved: {output_path2}")

# Test 3: Voice cloning (if reference audio exists)
print("\n--- Test 3: Voice Cloning ---")
ref_audio = Path("reference_voice.wav")
if ref_audio.exists():
    text3 = "This is my cloned voice speaking. Pretty cool, right?"
    wav3 = model.generate(text3, audio_prompt_path=str(ref_audio))
    output_path3 = output_dir / "turbo_cloned.wav"
    ta.save(str(output_path3), wav3, model.sr)
    print(f"Saved: {output_path3}")
else:
    print(f"Skipped: No reference audio found at '{ref_audio}'")
    print("To test voice cloning, provide a ~10 second WAV file named 'reference_voice.wav'")

print("\n=== Turbo TTS Tests Complete ===")
print(f"Check the '{output_dir}/' directory for output files.")
