#!/usr/bin/env python3
"""
Test voice cloning with a reference audio.

Usage:
    python test_voice_cloning.py                              # English + Chinese, uses reference_voice.wav
    python test_voice_cloning.py my_voice.wav                 # English + Chinese with custom file
    python test_voice_cloning.py my_voice.wav --lang en       # English only
    python test_voice_cloning.py my_voice.wav --lang zh       # Chinese only
    python test_voice_cloning.py my_voice.wav --lang en,zh,ja # Multiple languages

Supported languages: ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh
"""

import argparse
import torch
import torchaudio as ta
from pathlib import Path
import sys

# Sample texts for each language
SAMPLE_TEXTS = {
    "en": ("English", "Hello! This is a test of voice cloning. The model is trying to match my voice characteristics from the reference audio."),
    "zh": ("Chinese", "你好！这是一个语音克隆的测试。模型正在尝试从参考音频中匹配我的声音特征。"),
    "ja": ("Japanese", "こんにちは！これは音声クローニングのテストです。モデルは参考音声から私の声の特徴を再現しようとしています。"),
    "ko": ("Korean", "안녕하세요! 이것은 음성 복제 테스트입니다. 모델이 참조 오디오에서 제 목소리 특성을 일치시키려고 합니다."),
    "fr": ("French", "Bonjour! Ceci est un test de clonage vocal. Le modèle essaie de reproduire les caractéristiques de ma voix."),
    "de": ("German", "Hallo! Dies ist ein Test der Stimmklonung. Das Modell versucht, meine Stimmmerkmale nachzuahmen."),
    "es": ("Spanish", "¡Hola! Esta es una prueba de clonación de voz. El modelo está tratando de imitar las características de mi voz."),
    "pt": ("Portuguese", "Olá! Este é um teste de clonagem de voz. O modelo está tentando corresponder às características da minha voz."),
    "it": ("Italian", "Ciao! Questo è un test di clonazione vocale. Il modello sta cercando di riprodurre le caratteristiche della mia voce."),
    "ru": ("Russian", "Привет! Это тест клонирования голоса. Модель пытается воспроизвести характеристики моего голоса."),
    "ar": ("Arabic", "مرحبا! هذا اختبار لاستنساخ الصوت. يحاول النموذج مطابقة خصائص صوتي من الصوت المرجعي."),
    "hi": ("Hindi", "नमस्ते! यह वॉयस क्लोनिंग का परीक्षण है। मॉडल संदर्भ ऑडियो से मेरी आवाज़ की विशेषताओं का मिलान करने की कोशिश कर रहा है।"),
    "nl": ("Dutch", "Hallo! Dit is een test van stemklonen. Het model probeert mijn stemkenmerken na te bootsen."),
    "pl": ("Polish", "Cześć! To jest test klonowania głosu. Model próbuje odtworzyć cechy mojego głosu."),
    "tr": ("Turkish", "Merhaba! Bu bir ses klonlama testidir. Model, referans sesinden ses özelliklerimi eşleştirmeye çalışıyor."),
    "sv": ("Swedish", "Hej! Detta är ett test av röstkloning. Modellen försöker matcha mina röstegenskaper."),
    "da": ("Danish", "Hej! Dette er en test af stemmekloning. Modellen forsøger at matche mine stemmekarakteristika."),
    "no": ("Norwegian", "Hei! Dette er en test av stemmekloning. Modellen prøver å matche mine stemmeegenskaper."),
    "fi": ("Finnish", "Hei! Tämä on äänikloonauksen testi. Malli yrittää jäljitellä ääneni ominaisuuksia."),
    "el": ("Greek", "Γεια σας! Αυτή είναι μια δοκιμή κλωνοποίησης φωνής. Το μοντέλο προσπαθεί να αντιστοιχίσει τα χαρακτηριστικά της φωνής μου."),
    "he": ("Hebrew", "שלום! זהו מבחן לשכפול קול. המודל מנסה להתאים את מאפייני הקול שלי מהשמע המקורי."),
    "ms": ("Malay", "Halo! Ini adalah ujian pengklonan suara. Model cuba memadankan ciri-ciri suara saya."),
    "sw": ("Swahili", "Habari! Hii ni jaribio la kunakili sauti. Mfano unajaribu kulinganisha sifa za sauti yangu."),
}

parser = argparse.ArgumentParser(description="Test voice cloning with different languages")
parser.add_argument("reference", nargs="?", default="reference_voice.wav", help="Reference audio file")
parser.add_argument("--lang", "-l", default="en,zh", help="Languages to test (comma-separated, e.g., en,zh,ja)")
parser.add_argument("--exaggeration", "-e", type=float, default=0.5, help="Expressiveness (0.0-1.0+, lower=closer to original, default=0.5)")
parser.add_argument("--cfg", "-c", type=float, default=0.5, help="CFG weight (0.0-1.0, lower=better similarity, default=0.5)")
parser.add_argument("--temperature", "-t", type=float, default=0.8, help="Temperature (0.1-1.0, lower=more consistent, default=0.8)")
args = parser.parse_args()

ref_audio = Path(args.reference)
languages = [l.strip() for l in args.lang.split(",")]

# Validate languages
for lang in languages:
    if lang not in SAMPLE_TEXTS:
        print(f"Error: Unsupported language '{lang}'")
        print(f"Supported: {', '.join(sorted(SAMPLE_TEXTS.keys()))}")
        sys.exit(1)

if not ref_audio.exists():
    print(f"Error: {ref_audio} not found!")
    print("\nDownload a reference voice first:")
    print('  python download_reference.py "https://youtube.com/watch?v=..." --start 0:10 --end 0:20')
    print("\nOr specify a different file:")
    print('  python test_voice_cloning.py your_file.wav')
    sys.exit(1)

# Detect best available device for Mac
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")
print(f"Reference audio: {ref_audio}")

# Patch torch.load for MPS/CPU compatibility
map_location = torch.device(device)
torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)
torch.load = patched_torch_load

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Output directory
output_dir = Path("samples")
output_dir.mkdir(exist_ok=True)

# Output prefix based on reference filename
prefix = ref_audio.stem  # e.g., "speaker_john" from "speaker_john.wav"

# Load multilingual model
print("\nLoading Chatterbox Multilingual V3 model...")
model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
print("Model loaded!")

# Show settings
print(f"\nSettings: exaggeration={args.exaggeration}, cfg_weight={args.cfg}, temperature={args.temperature}")

# Generate for each requested language
output_files = []
for i, lang in enumerate(languages, 1):
    lang_name, text = SAMPLE_TEXTS[lang]
    print(f"\n--- Test {i}: {lang_name} ({lang}) Voice Cloning ---")

    wav = model.generate(
        text,
        language_id=lang,
        audio_prompt_path=str(ref_audio),
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg,
        temperature=args.temperature,
    )
    output_file = output_dir / f"{prefix}_{lang}.wav"
    ta.save(str(output_file), wav, model.sr)
    print(f"Saved: {output_file}")
    output_files.append(output_file.name)

print("\n=== Voice Cloning Tests Complete ===")
print(f"Check the '{output_dir}/' directory for output files:")
for f in output_files:
    print(f"  - {f}")
