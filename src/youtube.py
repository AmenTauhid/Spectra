import re
import tempfile
from pathlib import Path

import yt_dlp

YOUTUBE_PATTERNS = [
    re.compile(r'(https?://)?(www\.)?youtube\.com/watch\?v=[\w-]+'),
    re.compile(r'(https?://)?(www\.)?youtu\.be/[\w-]+'),
    re.compile(r'(https?://)?(music\.)?youtube\.com/watch\?v=[\w-]+'),
]


def is_youtube_url(text: str) -> bool:
    """Check if the given string looks like a YouTube URL."""
    return any(p.match(text) for p in YOUTUBE_PATTERNS)


def download_audio(url: str, output_dir: Path | None = None) -> Path:
    """Download audio from a YouTube URL as MP3. Returns path to the MP3 file."""
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="spectra_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    output_template = str(output_dir / "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "outtmpl": output_template,
        "quiet": False,
        "no_warnings": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "audio")
        # Sanitize filename the same way yt-dlp does
        filename = yt_dlp.utils.sanitize_filename(title)
        mp3_path = output_dir / f"{filename}.mp3"

    if not mp3_path.exists():
        # Fallback: find any mp3 in output_dir
        mp3_files = list(output_dir.glob("*.mp3"))
        if mp3_files:
            mp3_path = mp3_files[0]
        else:
            raise RuntimeError(f"Download succeeded but MP3 not found in {output_dir}")

    return mp3_path
