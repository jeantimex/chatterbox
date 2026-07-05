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

    # Download audio with yt-dlp (best audio format, we'll convert later)
    download_cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-quality", "0",  # Best quality
        "--no-playlist",  # Only download single video, not entire playlist
        "-o", "_temp_audio.%(ext)s",
        args.url
    ]

    try:
        result = subprocess.run(download_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"yt-dlp error:\n{result.stderr}")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading audio: {e}")
        sys.exit(1)

    # Find the downloaded file (extension varies based on source)
    temp_files = list(Path(".").glob("_temp_audio.*"))
    if not temp_files:
        print("Error: Downloaded file not found")
        print("yt-dlp output:", result.stdout)
        sys.exit(1)
    temp_audio = temp_files[0]
    print(f"Downloaded: {temp_audio}")

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
    """Use audio-separator CLI with UVR models to separate vocals from background music."""
    import shutil

    # Find audio-separator command (prefer pipx installation)
    audio_sep_cmd = None
    for cmd_path in [
        Path.home() / ".local/pipx/venvs/audio-separator/bin/audio-separator",
        Path.home() / ".local/bin/audio-separator",
        "audio-separator",  # fallback to PATH
    ]:
        result = subprocess.run([str(cmd_path), "--help"], capture_output=True)
        if result.returncode == 0:
            audio_sep_cmd = str(cmd_path)
            break

    if not audio_sep_cmd:
        print("audio-separator not found. Install it with:")
        print("  pipx install 'audio-separator[cpu]'")
        return False

    print("Isolating voice (this may take a minute)...")

    output_dir = Path("_separated")
    output_dir.mkdir(exist_ok=True)

    try:
        # Stage 1: Separate vocals from instrumental
        print("  Stage 1: Separating vocals from instrumental...")
        result = subprocess.run([
            audio_sep_cmd,
            str(input_path),
            "-m", "UVR-MDX-NET-Voc_FT.onnx",
            "--output_dir", str(output_dir),
            "--output_format", "WAV",
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Stage 1 error: {result.stderr}")
            return False

        # Find vocals file from stage 1
        vocals_file = None
        for f in output_dir.glob("*Vocals*"):
            vocals_file = f
            break
        if not vocals_file:
            for f in output_dir.glob("*vocal*"):
                vocals_file = f
                break

        if not vocals_file or not vocals_file.exists():
            print("Could not find vocals in stage 1 output")
            print(f"Files in output dir: {list(output_dir.glob('*'))}")
            return False

        # Stage 2: De-reverb the vocals for cleaner output
        print("  Stage 2: Removing reverb...")
        result = subprocess.run([
            audio_sep_cmd,
            str(vocals_file),
            "-m", "Reverb_HQ_By_FoxJoy.onnx",
            "--output_dir", str(output_dir),
            "--output_format", "WAV",
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Stage 2 error: {result.stderr}")
            # Continue with stage 1 output
            dry_vocals = vocals_file
        else:
            # Find dry vocals (no reverb)
            dry_vocals = None
            for f in output_dir.glob("*No Reverb*"):
                dry_vocals = f
                break
            if not dry_vocals:
                dry_vocals = vocals_file

        # Convert to proper format for voice cloning
        subprocess.run([
            "ffmpeg", "-y", "-i", str(dry_vocals),
            "-ac", "1", "-ar", "22050",
            str(output_path)
        ], capture_output=True, check=True)

        return True

    except Exception as e:
        print(f"Voice isolation error: {e}")
        return False

    finally:
        # Cleanup
        shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
