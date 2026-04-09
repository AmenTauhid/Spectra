import click
from pathlib import Path

from .audio import extract_features, load_audio_paths
from .palettes import get_palette
from .render import render_artwork, get_style_names
from .youtube import is_youtube_url, download_audio


@click.command()
@click.argument("input_path", type=str)
@click.option("--style", "-s",
              type=click.Choice(["radial", "terrain", "galaxy", "mosaic", "ribbon", "random"],
                                case_sensitive=False),
              default="radial", help="Art style to render.")
@click.option("--palette", "-p",
              type=click.Choice(["sunset", "ocean", "neon", "monochrome", "pastel", "auto"],
                                case_sensitive=False),
              default="auto", help="Color palette.")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output file path (.png or .svg) or directory for batch.")
@click.option("--size", type=int, default=4000,
              help="Output image size in pixels (square).")
@click.option("--bg", type=click.Choice(["black", "white"]), default="black",
              help="Background color.")
@click.option("--seed", type=int, default=None,
              help="Random seed for reproducibility.")
@click.option("--preview", is_flag=True, default=False,
              help="Render a fast low-res 800px preview.")
def main(input_path, style, palette, output, size, bg, seed, preview):
    """Spectra: Generate abstract art from audio files.

    INPUT_PATH can be an audio file (.mp3, .wav, .flac, .ogg), a directory
    for batch processing, or a YouTube URL.
    """
    if is_youtube_url(input_path):
        click.echo(f"Downloading audio from YouTube...")
        mp3_path = download_audio(input_path)
        click.echo(f"  Downloaded: {mp3_path.name}")
        audio_files = [mp3_path]
    else:
        input_p = Path(input_path)
        if not input_p.exists():
            raise click.BadParameter(f"Path does not exist: {input_path}", param_hint="INPUT_PATH")
        audio_files = load_audio_paths(input_p)
    click.echo(f"Found {len(audio_files)} audio file(s).")

    bg_color = "#000000" if bg == "black" else "#FAFAFA"
    render_size = 800 if preview else size

    for i, audio_file in enumerate(audio_files):
        click.echo(f"\n[{i+1}/{len(audio_files)}] Processing: {audio_file.name}")

        click.echo("  Extracting audio features...")
        features = extract_features(audio_file)
        click.echo(f"  Duration: {features.duration:.1f}s, "
                   f"Tempo: {features.tempo:.0f} BPM, "
                   f"Frames: {features.n_frames}")

        pal = get_palette(palette, audio_features=features if palette == "auto" else None)
        click.echo(f"  Palette: {pal.name}")

        out_path = _resolve_output_path(output, audio_file, style, i, len(audio_files))

        click.echo(f"  Rendering style '{style}' at {render_size}x{render_size}...")
        result = render_artwork(features, style, pal, out_path, render_size, bg_color, seed)
        click.echo(f"  Saved: {result}")

    click.echo("\nDone.")


def _resolve_output_path(output, audio_file, style, index, total):
    """Determine output file path."""
    if output is None:
        return audio_file.parent / f"{audio_file.stem}_{style}.png"

    out_p = Path(output)

    # If output is a directory or ends with separator
    if out_p.is_dir() or str(output).endswith(('/', '\\')):
        out_p.mkdir(parents=True, exist_ok=True)
        suffix = ".png"
        stem = audio_file.stem
        if total > 1:
            return out_p / f"{stem}_{style}_{index+1:03d}{suffix}"
        return out_p / f"{stem}_{style}{suffix}"

    # Single file output
    if total > 1:
        stem = out_p.stem
        suffix = out_p.suffix or ".png"
        return out_p.parent / f"{stem}_{index+1:03d}{suffix}"

    return out_p
