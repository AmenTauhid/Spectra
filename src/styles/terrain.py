import numpy as np
from matplotlib.collections import LineCollection

from src.styles.base import StyleRenderer
from src.palettes import sample_colors
from src.utils import resample_to_length, resample_2d_time_axis


class TerrainFlowRenderer(StyleRenderer):
    name = "terrain"

    def render(self, fig):
        ax = self._setup_axes(fig)

        n_rows = 80
        n_cols = 800
        f = self.features

        # Downsample STFT to grid
        stft = resample_2d_time_axis(f.stft_mag, n_cols)
        # Take evenly spaced frequency bands
        freq_indices = np.linspace(0, stft.shape[0] - 1, n_rows, dtype=int)
        stft = stft[freq_indices]

        # Downsample features to n_cols
        centroid = resample_to_length(f.spectral_centroid, n_cols)
        rms = resample_to_length(f.rms, n_cols)

        x = np.linspace(0, 1, n_cols)
        displacement_scale = 0.012

        # Beat markers as faint vertical lines
        beat_x = f.beat_frames / f.n_frames
        ax.vlines(beat_x, 0, 1, colors='white', alpha=0.04, linewidth=0.5, zorder=0)

        # Draw ridges back to front (top = high freq drawn first, bottom = low freq on top)
        for i in range(n_rows - 1, -1, -1):
            y_base = (i + 0.5) / n_rows
            y_displaced = y_base + stft[i] * displacement_scale

            # Occlusion fill: solid background color below the ridge
            ax.fill_between(x, y_base - 0.005, y_displaced,
                            color=self.bg_color, zorder=n_rows - i, linewidth=0)

            # Build colored line segments
            points = np.column_stack([x, y_displaced])
            segments = np.stack([points[:-1], points[1:]], axis=1)

            # Color from spectral centroid
            colors = sample_colors(self.palette, centroid[:-1])
            # Modulate alpha by row amplitude
            row_energy = np.mean(stft[i])
            colors[:, 3] = 0.3 + row_energy * 0.7

            # Linewidth from RMS
            lw = 0.5 + rms[:-1] * 2.0

            lc = LineCollection(segments, colors=colors, linewidths=lw,
                                zorder=n_rows - i + 0.5)
            ax.add_collection(lc)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.05)
