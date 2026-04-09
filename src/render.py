import matplotlib
matplotlib.use('Agg')

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.figure as mpl_figure
import numpy as np

from .audio import AudioFeatures
from .palettes import Palette
from .styles.base import StyleRenderer
from .styles.radial import RadialBurstRenderer
from .styles.terrain import TerrainFlowRenderer
from .styles.galaxy import ParticleGalaxyRenderer
from .styles.mosaic import GeometricMosaicRenderer
from .styles.ribbon import WaveformRibbonRenderer


STYLE_MAP: dict[str, type[StyleRenderer]] = {
    "radial": RadialBurstRenderer,
    "terrain": TerrainFlowRenderer,
    "galaxy": ParticleGalaxyRenderer,
    "mosaic": GeometricMosaicRenderer,
    "ribbon": WaveformRibbonRenderer,
}


def get_style_names() -> list[str]:
    return list(STYLE_MAP.keys()) + ["random"]


def create_figure(size: int = 4000, dpi: int = 200) -> mpl_figure.Figure:
    inches = size / dpi
    fig = plt.figure(figsize=(inches, inches), dpi=dpi)
    return fig


def render_artwork(features: AudioFeatures, style_name: str, palette: Palette,
                   output_path: Path, size: int = 4000, bg_color: str = "#000000",
                   seed: int | None = None, dpi: int = 200) -> Path:
    """Render artwork and save to output_path. Returns the path."""
    rng = np.random.default_rng(seed)

    if style_name == "random":
        style_name = rng.choice(list(STYLE_MAP.keys()))

    renderer_cls = STYLE_MAP[style_name]
    renderer = renderer_cls(features, palette, size, bg_color, seed)

    fig = create_figure(size, dpi)
    renderer.render(fig)

    output_path = Path(output_path)
    fmt = "svg" if output_path.suffix.lower() == ".svg" else "png"
    fig.savefig(
        str(output_path), format=fmt, dpi=dpi,
        facecolor=fig.get_facecolor(), edgecolor='none',
        bbox_inches='tight', pad_inches=0,
    )
    plt.close(fig)
    return output_path
