import colorsys
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from matplotlib.colors import LinearSegmentedColormap, to_rgba


PaletteName = str  # "sunset" | "ocean" | "neon" | "monochrome" | "pastel" | "auto"


@dataclass(frozen=True)
class Palette:
    name: str
    colors: list[str]
    cmap: LinearSegmentedColormap
    bg_color: str


def _build_palette(name: str, hex_colors: list[str], bg: str = "#000000") -> Palette:
    rgba_list = [to_rgba(c) for c in hex_colors]
    cmap = LinearSegmentedColormap.from_list(name, rgba_list, N=256)
    return Palette(name=name, colors=hex_colors, cmap=cmap, bg_color=bg)


PALETTES: dict[str, Palette] = {
    "sunset": _build_palette("sunset",
        ["#FF6B35", "#F7C548", "#D64045", "#9B2335", "#5B1A5E"], "#0A0A0A"),
    "ocean": _build_palette("ocean",
        ["#006D77", "#83C5BE", "#EDF6F9", "#FFDDD2", "#004E64"], "#001219"),
    "neon": _build_palette("neon",
        ["#FF00FF", "#00FFFF", "#39FF14", "#FF3F00", "#FFFF00"], "#000000"),
    "monochrome": _build_palette("monochrome",
        ["#1A1A2E", "#3A3A5E", "#6A6A9E", "#9A9ACE", "#DADAFF"], "#000000"),
    "pastel": _build_palette("pastel",
        ["#FFB5B5", "#B5D8FF", "#C4FFB5", "#FFE4B5", "#E0B5FF"], "#FAFAFA"),
}


def auto_palette(audio_features) -> Palette:
    """Derive palette from audio features."""
    mean_centroid = float(np.mean(audio_features.spectral_centroid))
    energy_var = float(np.var(audio_features.rms))
    tempo = audio_features.tempo

    # Map centroid (normalized 0-1) to hue (0-300 degrees, red through blue)
    hue_center = mean_centroid * 300.0 / 360.0

    # Energy variance -> saturation
    saturation = 0.3 + min(energy_var * 40.0, 0.7)

    # Tempo -> lightness spread
    if tempo > 140:
        l_low, l_high = 0.2, 0.9
    elif tempo < 80:
        l_low, l_high = 0.35, 0.65
    else:
        t = (tempo - 80) / 60.0
        l_low = 0.35 - t * 0.15
        l_high = 0.65 + t * 0.25

    # Generate 5 hues rotating around center
    offsets = [0, 1/12, -1/12, 1/6, -1/6]  # ~30 and 60 degree offsets
    hex_colors = []
    lightnesses = np.linspace(l_low, l_high, 5)
    for i, offset in enumerate(offsets):
        h = (hue_center + offset) % 1.0
        r, g, b = colorsys.hls_to_rgb(h, lightnesses[i], saturation)
        hex_colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")

    return _build_palette("auto", hex_colors, "#0A0A0A")


def get_palette(name: PaletteName, audio_features=None) -> Palette:
    """Return the named palette. If 'auto', derive from audio_features."""
    if name == "auto":
        if audio_features is None:
            return PALETTES["sunset"]
        return auto_palette(audio_features)
    return PALETTES[name]


def sample_colors(palette: Palette, values: NDArray[np.floating]) -> NDArray[np.floating]:
    """Vectorized: sample RGBA for an array of values in [0,1]. Returns (N, 4)."""
    return palette.cmap(np.clip(values, 0.0, 1.0))
