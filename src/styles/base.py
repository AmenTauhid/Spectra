from abc import ABC, abstractmethod

import matplotlib.figure as mpl_figure
import matplotlib.axes as mpl_axes
import numpy as np

from ..audio import AudioFeatures
from ..palettes import Palette


class StyleRenderer(ABC):
    name: str

    def __init__(self, features: AudioFeatures, palette: Palette,
                 size: int = 4000, bg_color: str = "#000000",
                 seed: int | None = None):
        self.features = features
        self.palette = palette
        self.size = size
        self.bg_color = bg_color
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def render(self, fig: mpl_figure.Figure) -> None:
        """Draw the artwork onto the given matplotlib Figure."""

    def _setup_axes(self, fig: mpl_figure.Figure,
                    polar: bool = False) -> mpl_axes.Axes:
        """Add axes filling the entire figure with background set."""
        ax = fig.add_axes([0, 0, 1, 1], polar=polar)
        ax.set_facecolor(self.bg_color)
        fig.patch.set_facecolor(self.bg_color)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return ax

    def _downsample_step(self, target: int = 500) -> int:
        """Return step size to downsample n_frames to approximately target frames."""
        return max(1, self.features.n_frames // target)
