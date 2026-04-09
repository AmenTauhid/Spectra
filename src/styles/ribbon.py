import numpy as np
from matplotlib.collections import PolyCollection, LineCollection

from .base import StyleRenderer
from ..palettes import sample_colors
from ..utils import resample_to_length, smooth


class WaveformRibbonRenderer(StyleRenderer):
    name = "ribbon"

    def render(self, fig):
        ax = self._setup_axes(fig)
        f = self.features

        n_pts = 2000

        # Downsample waveform to n_pts via block mean
        block_size = max(1, len(f.waveform) // n_pts)
        trimmed = f.waveform[:block_size * n_pts]
        waveform_ds = trimmed.reshape(n_pts, block_size).mean(axis=1)

        x = np.linspace(0, 1, n_pts)
        y_center = 0.5 + smooth(waveform_ds, window=7) * 0.35

        # Resample features
        rms = resample_to_length(f.rms, n_pts)
        centroid = resample_to_length(f.spectral_centroid, n_pts)
        onset_env = resample_to_length(f.onset_env, n_pts)

        # Apply onset widening before computing ribbon width
        half_width = 0.005 + rms * 0.04
        # Widen at onset peaks
        onset_mask = onset_env > 0.5
        half_width[onset_mask] *= 1.8

        y_upper = y_center + half_width
        y_lower = y_center - half_width

        # Color from spectral centroid
        colors = sample_colors(self.palette, centroid[:-1])
        colors[:, 3] = 0.7 + rms[:-1] * 0.3

        # Build ribbon quads
        verts = np.zeros((n_pts - 1, 4, 2))
        verts[:, 0, 0] = x[:-1]
        verts[:, 0, 1] = y_lower[:-1]
        verts[:, 1, 0] = x[1:]
        verts[:, 1, 1] = y_lower[1:]
        verts[:, 2, 0] = x[1:]
        verts[:, 2, 1] = y_upper[1:]
        verts[:, 3, 0] = x[:-1]
        verts[:, 3, 1] = y_upper[:-1]

        # Drop shadow
        shadow_verts = verts.copy()
        shadow_verts[:, :, 0] += 0.003
        shadow_verts[:, :, 1] -= 0.008
        shadow_colors = np.zeros_like(colors)
        shadow_colors[:, 3] = 0.15
        shadow_pc = PolyCollection(shadow_verts, facecolors=shadow_colors,
                                   edgecolors='none', zorder=1)
        ax.add_collection(shadow_pc)

        # Main ribbon
        pc = PolyCollection(verts, facecolors=colors, edgecolors='none', zorder=2)
        ax.add_collection(pc)

        # Edge lines
        for y_edge in [y_upper, y_lower]:
            points = np.column_stack([x, y_edge])
            segs = np.stack([points[:-1], points[1:]], axis=1)
            edge_colors = sample_colors(self.palette, centroid[:-1])
            edge_colors[:, :3] = np.minimum(edge_colors[:, :3] * 1.3, 1.0)
            edge_colors[:, 3] = 0.5
            lc = LineCollection(segs, colors=edge_colors, linewidths=0.5, zorder=3)
            ax.add_collection(lc)

        # Beat knots
        beat_x = f.beat_frames / f.n_frames
        beat_indices = np.clip((beat_x * n_pts).astype(int), 0, n_pts - 1)
        if len(beat_indices) > 0:
            bx = x[beat_indices]
            by = y_center[beat_indices]
            beat_rms = rms[beat_indices]
            sizes = 20 + beat_rms * 80
            knot_colors = sample_colors(self.palette, np.full(len(beat_indices), 0.9))
            knot_colors[:, 3] = 0.8
            ax.scatter(bx, by, s=sizes, c=knot_colors, marker='D',
                       edgecolors='white', linewidths=0.3, zorder=4)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
