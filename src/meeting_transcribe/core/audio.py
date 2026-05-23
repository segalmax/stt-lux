import math
import os
import subprocess
import sys
import tempfile

from meeting_transcribe.core.config import CHUNK_MINUTES, MAX_DURATION_S, MAX_FILE_MB


def chunk_audio(path: str, chunk_minutes: int = CHUNK_MINUTES) -> list[str]:
    """Split audio into chunks under 25MB. Returns list of temp file paths."""
    try:
        from pydub import AudioSegment
    except ImportError:
        print("ERROR: pydub not installed. Run: pip install pydub", file=sys.stderr)
        print("Also ensure ffmpeg is installed: brew install ffmpeg", file=sys.stderr)
        sys.exit(1)

    audio = AudioSegment.from_file(path)
    chunk_ms = chunk_minutes * 60 * 1000
    num_chunks = math.ceil(len(audio) / chunk_ms)

    print(f"  Splitting into {num_chunks} chunks of {chunk_minutes} min each...")

    chunks = []
    tmpdir = tempfile.mkdtemp(prefix="transcribe_")
    for i in range(num_chunks):
        chunk = audio[i * chunk_ms : (i + 1) * chunk_ms]
        chunk_path = os.path.join(tmpdir, f"chunk_{i:03d}.m4a")
        chunk.export(chunk_path, format="ipod")  # ipod = .m4a container
        chunks.append(chunk_path)

    return chunks


def audio_duration_seconds(path: str) -> float:
    """Get audio duration in seconds. Tries pydub, falls back to ffprobe."""
    try:
        from pydub import AudioSegment
        duration = len(AudioSegment.from_file(path)) / 1000.0
        if duration > 0:
            return duration
    except Exception:
        pass
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except FileNotFoundError:
        pass
    return 0.0
