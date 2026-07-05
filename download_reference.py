#!/usr/bin/env python3
"""
Download audio from YouTube and extract a clip for voice cloning reference.

Usage:
    python download_reference.py <youtube_url> [--start 0:30] [--end 0:40] [--output reference_voice.wav]

Examples:
    # Download full audio
    python download_reference.py "https://www.youtube.com/watch?v=VIDEO_ID"

    # Extract 10 seconds starting at 1:30
    python download_reference.py "https://www.youtube.com/watch?v=VIDEO_ID" --start 1:30 --end 1:40

    # Custom output filename
    python download_reference.py "https://www.youtube.com/watch?v=VIDEO_ID" --start 0:10 --end 0:20 --output my_voice.wav

    # Remove background music and isolate voice
    python download_reference.py "https://www.youtube.com/watch?v=VIDEO_ID" --start 0:10 --end 0:20 --isolate-voice
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_timestamp(ts: str) -> float:
    """Parse timestamp string (e.g., '1:30' or '90') to seconds."""
    if ts is None:
        return None
    parts = ts.split(':')
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    else:
        raise ValueError(f"Invalid timestamp format: {ts}")


def main():
    parser = argparse.ArgumentParser(
        description="Download audio from YouTube for voice cloning reference"
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--start", "-s", help="Start time (e.g., 1:30 or 90)")
    parser.add_argument("--end", "-e", help="End time (e.g., 1:40 or 100)")
    parser.add_argument(
        "--output", "-o",
        default="reference_voice.wav",
        help="Output filename (default: reference_voice.wav)"
    )
    parser.add_argument(
        "--isolate-voice", "-i",
        action="store_true",
        help="Remove background music and isolate voice using demucs"
    )
    args = parser.parse_args()

    # Check for yt-dlp
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("Error: yt-dlp not found. Install it with:")
        print("  brew install yt-dlp")
        print("  # or: pip install yt-dlp")
        sys.exit(1)

    # Check for ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("Error: ffmpeg not found. Install it with:")
        print("  brew install ffmpeg")
        sys.exit(1)

    output_path = Path(args.output)
    temp_audio = Path("_temp_audio.wav")

    print(f"Downloading audio from: {args.url}")

    # Download audio with yt-dlp
    download_cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format", "wav",
        "--audio-quality", "0",  # Best quality
        "-o", str(temp_audio.with_suffix("")),  # yt-dlp adds extension
        args.url
    ]

    try:
        subprocess.run(download_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading audio: {e}")
        sys.exit(1)

    # Find the downloaded file (yt-dlp may add extension)
    temp_files = list(Path(".").glob("_temp_audio.*"))
    if not temp_files:
        print("Error: Downloaded file not found")
        sys.exit(1)
    temp_audio = temp_files[0]

    # Cut audio if start/end specified
    start_sec = parse_timestamp(args.start)
    end_sec = parse_timestamp(args.end)

    if start_sec is not None or end_sec is not None:
        print(f"Extracting clip: {args.start or '0:00'} to {args.end or 'end'}")

        ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(temp_audio)]

        if start_sec is not None:
            ffmpeg_cmd.extend(["-ss", str(start_sec)])
        if end_sec is not None:
            if start_sec is not None:
                duration = end_sec - start_sec
                ffmpeg_cmd.extend(["-t", str(duration)])
            else:
                ffmpeg_cmd.extend(["-to", str(end_sec)])

        # Convert to mono 22050Hz WAV (good for voice cloning)
        ffmpeg_cmd.extend([
            "-ac", "1",  # Mono
            "-ar", "22050",  # Sample rate
            str(output_path)
        ])

        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing audio: {e}")
            sys.exit(1)

        # Clean up temp file
        temp_audio.unlink()
    else:
        # Just convert to proper format
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", str(temp_audio),
            "-ac", "1",
            "-ar", "22050",
            str(output_path)
        ]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        temp_audio.unlink()

    # Isolate voice if requested
    if args.isolate_voice:
        temp_with_music = output_path.with_stem(output_path.stem + "_with_music")
        output_path.rename(temp_with_music)
        if isolate_voice(temp_with_music, output_path):
            temp_with_music.unlink()
            print("Voice isolated successfully!")
        else:
            temp_with_music.rename(output_path)
            print("Voice isolation failed, keeping original audio.")

    duration = get_duration(output_path)
    print(f"\nSaved: {output_path}")
    print(f"Duration: {duration:.1f} seconds")

    if duration < 5:
        print("\n⚠️  Warning: Audio is shorter than 5 seconds. Turbo model requires >5 sec.")
    elif duration < 6:
        print("\n⚠️  Warning: Audio is shorter than 6 seconds. Consider a longer clip for best results.")
    elif duration > 15:
        print(f"\nNote: Only first 15 sec used by Turbo, first 6 sec by Multilingual.")
    else:
        print("\n✓ Duration is good for voice cloning!")

    print("\nReady for voice cloning! Run:")
    print(f"  python test_voice_cloning.py {output_path}")


def get_duration(filepath: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(filepath)],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def isolate_voice(input_path: Path, output_path: Path) -> bool:
    """Use demucs to separate vocals from background music."""
    try:
        subprocess.run(["python", "-c", "import demucs"], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        print("Installing demucs for voice isolation...")
        subprocess.run([sys.executable, "-m", "pip", "install", "demucs"], check=True)

    print("Isolating voice (this may take a minute)...")

    # Run demucs to separate vocals
    result = subprocess.run(
        [sys.executable, "-m", "demucs", "--two-stems=vocals", "-o", "separated", str(input_path)],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"Demucs error: {result.stderr}")
        return False

    # Find the vocals file
    stem_name = input_path.stem
    vocals_path = Path("separated") / "htdemucs" / stem_name / "vocals.wav"

    if not vocals_path.exists():
        # Try alternative path structure
        for vp in Path("separated").rglob("vocals.wav"):
            vocals_path = vp
            break

    if not vocals_path.exists():
        print("Could not find separated vocals file")
        return False

    # Convert to proper format
    subprocess.run([
        "ffmpeg", "-y", "-i", str(vocals_path),
        "-ac", "1", "-ar", "22050",
        str(output_path)
    ], capture_output=True, check=True)

    # Cleanup
    import shutil
    shutil.rmtree("separated", ignore_errors=True)

    return True


if __name__ == "__main__":
    main()
