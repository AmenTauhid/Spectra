# Spectra

Generate unique abstract art from audio files. Feed in a song, get a one-of-a-kind piece of wall-worthy art shaped by its frequency, rhythm, and energy.

## Art Styles

| Style | Description |
|-------|-------------|
| **Radial Burst** | Time maps to angle, frequency to radius. Looks like an exploding star. |
| **Terrain Flow** | Stacked frequency ridges displaced by amplitude. Colorful mountain ranges. |
| **Particle Galaxy** | Thousands of particles positioned by frequency, colored by pitch. A nebula. |
| **Geometric Mosaic** | Grid of shapes that morph with energy. Stained glass effect. |
| **Waveform Ribbon** | Flowing ribbon from the waveform with width pulsing to the beat. |

## Color Palettes

- **Sunset** - warm oranges, reds, purples
- **Ocean** - teals, deep blues, white foam
- **Neon** - hot pink, electric blue, lime on black
- **Monochrome** - single hue with varying lightness
- **Pastel** - soft muted tones
- **Auto** - derived from the audio (tempo, energy, brightness)

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

## Usage

```bash
# Single file with specific style and palette
python spectra.py song.mp3 --style radial --palette neon --output art.png

# Auto palette derived from the music
python spectra.py song.wav --style terrain --palette auto --output art.png

# Random style, 6000px output
python spectra.py song.flac --style random --palette ocean --output art.png --size 6000

# SVG vector output for print
python spectra.py song.mp3 --style galaxy --palette sunset --output art.svg

# Batch process a folder
python spectra.py album_folder/ --style mosaic --palette neon --output posters/

# Fast 800px preview
python spectra.py song.mp3 --style ribbon --preview

# White background
python spectra.py song.mp3 --style mosaic --palette pastel --bg white --output art.png

# Reproducible output with seed
python spectra.py song.mp3 --style galaxy --seed 42 --output art.png

# YouTube URL - downloads audio automatically
python spectra.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --style radial --palette neon --output art.png

# YouTube Music link works too
python spectra.py "https://music.youtube.com/watch?v=dQw4w9WgXcQ" --style galaxy --output art.png
```

## CLI Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `input_path` | file, directory, or YouTube URL | required | Audio file, folder for batch, or YouTube URL |
| `--style, -s` | radial, terrain, galaxy, mosaic, ribbon, random | radial | Art style |
| `--palette, -p` | sunset, ocean, neon, monochrome, pastel, auto | auto | Color palette |
| `--output, -o` | file path | `<input>_<style>.png` | Output path (.png or .svg) |
| `--size` | integer | 4000 | Canvas size in pixels |
| `--bg` | black, white | black | Background color |
| `--seed` | integer | none | Random seed for reproducibility |
| `--preview` | flag | off | Render at 800px instead of full size |

## YouTube Support

Pass a YouTube URL directly as the input and Spectra will download the audio as MP3 using [yt-dlp](https://github.com/yt-dlp/yt-dlp) before processing. Requires ffmpeg installed on your system.

Supported URL formats:
- `https://www.youtube.com/watch?v=...`
- `https://youtu.be/...`
- `https://music.youtube.com/watch?v=...`

## How It Works

1. **Audio Analysis** - Loads the audio with librosa and extracts: STFT (frequency over time), chromagram (pitch classes), RMS energy, spectral centroid (brightness), onset detection (beats/transients), and tempo.

2. **Feature Normalization** - All features are normalized to [0, 1] so renderers work with consistent ranges.

3. **Art Rendering** - The selected style renderer maps audio features to visual properties (position, size, color, opacity) using matplotlib.

4. **Output** - High-resolution PNG (default 4000x4000) or SVG vector format.

## Project Structure

```
src/main.py            - CLI entry point
src/youtube.py         - YouTube audio downloading via yt-dlp
src/audio.py           - Audio loading and feature extraction
src/palettes.py        - Color palette definitions
src/styles/base.py     - Abstract renderer base class
src/styles/radial.py   - Radial Burst style
src/styles/terrain.py  - Terrain Flow style
src/styles/galaxy.py   - Particle Galaxy style
src/styles/mosaic.py   - Geometric Mosaic style
src/styles/ribbon.py   - Waveform Ribbon style
src/render.py          - Output handler (PNG/SVG)
src/utils.py           - Math utilities
tests/                 - Test suite (runs with synthetic audio)
```

## Dependencies

- librosa - audio analysis
- soundfile - audio I/O
- numpy - array math
- matplotlib - rendering
- click - CLI framework
- yt-dlp - YouTube audio downloading
