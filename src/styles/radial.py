import numpy as np
from matplotlib.collections import LineCollection

from .base import StyleRenderer
from ..palettes import sample_colors
from ..utils import resample_to_length, resample_2d_time_axis


class RadialBurstRenderer(StyleRenderer):
    name = "radial"

    def render(self, fig):
        ax = self._setup_axes(fig, polar=True)
        f = self.features

        n_angles = 360
        n_bands = 50

        # Downsample STFT
        stft = resample_2d_time_axis(f.stft_mag, n_angles)
        freq_indices = np.linspace(0, stft.shape[0] - 1, n_bands, dtype=int)
        stft = stft[freq_indices]

        centroid = resample_to_length(f.spectral_centroid, n_angles)
        rms = resample_to_length(f.rms, n_angles)

        # Background rings - subtle frequency band ambiance
        theta_ring = np.linspace(0, 2 * np.pi, 200)
        for b in range(8):
            r_low = b / 8
            r_high = (b + 1) / 8
            band_slice = stft[b * (n_bands // 8):(b + 1) * (n_bands // 8)]
            energy = float(np.mean(band_slice))
            color = sample_colors(self.palette, np.array([b / 7]))[0]
            color[3] = energy * 0.15
            ax.fill_between(theta_ring, r_low, r_high, color=color, zorder=0)

        # Main rays via LineCollection
        # Each ray: from inner radius to outer radius at a given angle
        segments = []
        colors_list = []
        widths = []

        for t in range(n_angles):
            theta = (t / n_angles) * 2 * np.pi
            color_base = sample_colors(self.palette, np.array([centroid[t]]))[0]

            for b in range(n_bands):
                amp = stft[b, t]
                if amp < 0.02:
                    continue
                r_start = b / n_bands
                r_end = (b + 1) / n_bands
                segments.append([(theta, r_start), (theta, r_end)])
                c = color_base.copy()
                c[3] = amp ** 0.7
                colors_list.append(c)
                widths.append(0.5 + amp * 4.0)

        if segments:
            lc = LineCollection(segments, colors=colors_list, linewidths=widths, zorder=1)
            ax.add_collection(lc)

        # Beat accent lines
        beat_thetas = (f.beat_frames / f.n_frames) * 2 * np.pi
        if len(beat_thetas) > 0:
            beat_segments = [[(theta, 0.0), (theta, 1.0)] for theta in beat_thetas]
            bright = sample_colors(self.palette, np.array([0.95]))[0]
            bright[3] = 0.6
            beat_lc = LineCollection(beat_segments, colors=[bright] * len(beat_segments),
                                    linewidths=1.0, zorder=2)
            ax.add_collection(beat_lc)

        # Onset scatter
        if len(f.onset_frames) > 0:
            onset_thetas = (f.onset_frames / f.n_frames) * 2 * np.pi
            onset_r = self.rng.uniform(0.3, 0.9, len(onset_thetas))
            onset_sizes = f.onset_env[f.onset_frames] * 30 + 2
            onset_colors = sample_colors(self.palette,
                                         f.spectral_centroid[f.onset_frames])
            onset_colors[:, 3] = 0.5
            ax.scatter(onset_thetas, onset_r, s=onset_sizes, c=onset_colors,
                       edgecolors='none', zorder=3)

        # Center glow
        for r in np.linspace(0.15, 0.01, 10):
            color = sample_colors(self.palette, np.array([0.5]))[0]
            color[3] = 0.03
            circle = np.linspace(0, 2 * np.pi, 100)
            ax.fill_between(circle, 0, r, color=color, zorder=0)

        ax.set_rlim(0, 1)
