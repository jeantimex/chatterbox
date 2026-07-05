#!/usr/bin/env python3
"""
Test script for Chatterbox Multilingual TTS (23+ languages)
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

from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Output directory
output_dir = Path("samples")
output_dir.mkdir(exist_ok=True)

# Test 1: English TTS (base model)
print("\n--- Test 1: English TTS (ChatterboxTTS) ---")
print("Loading English model...")
model_en = ChatterboxTTS.from_pretrained(device=device)
print("Model loaded!")

text_en = "The quick brown fox jumps over the lazy dog. This is a test of the English text to speech system."
wav_en = model_en.generate(text_en)
output_en = output_dir / "multilingual_english.wav"
ta.save(str(output_en), wav_en, model_en.sr)
print(f"Saved: {output_en}")

# Test 2: Multilingual TTS (V3)
print("\n--- Test 2: Multilingual TTS (V3) ---")
print("Loading Multilingual V3 model...")
model_mtl = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
print("Model loaded!")

# French
print("\nGenerating French...")
text_fr = "Bonjour, comment ça va? Ceci est un test du système de synthèse vocale multilingue."
wav_fr = model_mtl.generate(text_fr, language_id="fr")
output_fr = output_dir / "multilingual_french.wav"
ta.save(str(output_fr), wav_fr, model_mtl.sr)
print(f"Saved: {output_fr}")

# Chinese
print("\nGenerating Chinese...")
text_zh = "你好，这是一个多语言文字转语音系统的测试。希望你喜欢这个声音。"
wav_zh = model_mtl.generate(text_zh, language_id="zh")
output_zh = output_dir / "multilingual_chinese.wav"
ta.save(str(output_zh), wav_zh, model_mtl.sr)
print(f"Saved: {output_zh}")

# Japanese
print("\nGenerating Japanese...")
text_ja = "こんにちは、これは多言語テキスト読み上げシステムのテストです。"
wav_ja = model_mtl.generate(text_ja, language_id="ja")
output_ja = output_dir / "multilingual_japanese.wav"
ta.save(str(output_ja), wav_ja, model_mtl.sr)
print(f"Saved: {output_ja}")

# Spanish
print("\nGenerating Spanish...")
text_es = "Hola, esto es una prueba del sistema de síntesis de voz multilingüe."
wav_es = model_mtl.generate(text_es, language_id="es")
output_es = output_dir / "multilingual_spanish.wav"
ta.save(str(output_es), wav_es, model_mtl.sr)
print(f"Saved: {output_es}")

# German
print("\nGenerating German...")
text_de = "Hallo, dies ist ein Test des mehrsprachigen Text-zu-Sprache-Systems."
wav_de = model_mtl.generate(text_de, language_id="de")
output_de = output_dir / "multilingual_german.wav"
ta.save(str(output_de), wav_de, model_mtl.sr)
print(f"Saved: {output_de}")

# Test 3: Voice cloning with multilingual
print("\n--- Test 3: Voice Cloning (Multilingual) ---")
ref_audio = Path("reference_voice.wav")
if ref_audio.exists():
    text_clone = "This is my cloned voice speaking in English using the multilingual model."
    wav_clone = model_mtl.generate(text_clone, language_id="en", audio_prompt_path=str(ref_audio))
    output_clone = output_dir / "multilingual_cloned.wav"
    ta.save(str(output_clone), wav_clone, model_mtl.sr)
    print(f"Saved: {output_clone}")
else:
    print(f"Skipped: No reference audio found at '{ref_audio}'")
    print("To test voice cloning, provide a ~10 second WAV file named 'reference_voice.wav'")

print("\n=== Multilingual TTS Tests Complete ===")
print(f"Check the '{output_dir}/' directory for output files.")
print("\nSupported languages: ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh")
