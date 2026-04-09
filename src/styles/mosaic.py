import numpy as np
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.collections import PatchCollection

from src.styles.base import StyleRenderer
from src.palettes import sample_colors
from src.utils import resample_to_length, resample_2d_time_axis


class GeometricMosaicRenderer(StyleRenderer):
    name = "mosaic"

    def render(self, fig):
        ax = self._setup_axes(fig)
        f = self.features

        n_cols = 40
        n_rows = 30

        # Downsample STFT to grid
        stft = resample_2d_time_axis(f.stft_mag, n_cols)
        freq_indices = np.linspace(0, stft.shape[0] - 1, n_rows, dtype=int)
        stft = stft[freq_indices]

        centroid = resample_to_length(f.spectral_centroid, n_cols)
        onset = resample_to_length(f.onset_env, n_cols)

        cell_w = 1.0 / n_cols
        cell_h = 1.0 / n_rows
        cell_size = min(cell_w, cell_h)

        is_white_bg = self.bg_color.upper() in ("#FAFAFA", "#FFFFFF", "WHITE")
        outline_color = 'black' if is_white_bg else 'white'
        grid_color = '#AAAAAA' if is_white_bg else '#333333'

        # Background color wash per column
        col_colors = sample_colors(self.palette, centroid)
        for c in range(n_cols):
            color = col_colors[c].copy()
            color[3] = 0.05
            ax.axvspan(c * cell_w, (c + 1) * cell_w, color=color, zorder=0)

        # Build patches
        patches = []
        face_colors = []
        edge_colors = []
        linewidths = []

        for row in range(n_rows):
            for col in range(n_cols):
                cx = (col + 0.5) * cell_w
                cy = (row + 0.5) * cell_h
                energy = stft[row, col]
                radius = (0.3 + energy * 0.6) * cell_size * 0.45

                # Shape selection by energy level
                if energy < 0.2:
                    patch = Circle((cx, cy), radius)
                elif energy < 0.4:
                    patch = RegularPolygon((cx, cy), numVertices=3, radius=radius,
                                          orientation=self.rng.uniform(0, np.pi))
                elif energy < 0.6:
                    patch = RegularPolygon((cx, cy), numVertices=4, radius=radius,
                                          orientation=np.pi / 4)
                elif energy < 0.8:
                    patch = RegularPolygon((cx, cy), numVertices=5, radius=radius,
                                          orientation=self.rng.uniform(0, np.pi))
                else:
                    patch = RegularPolygon((cx, cy), numVertices=6, radius=radius)

                patches.append(patch)

                # Color from centroid
                color = sample_colors(self.palette, np.array([centroid[col]]))[0]
                color[3] = 0.5 + energy * 0.5
                face_colors.append(color)

                # Onset columns get bold outlines
                if onset[col] > 0.6:
                    edge_colors.append(outline_color)
                    linewidths.append(2.0)
                else:
                    darker = color.copy()
                    darker[:3] *= 0.6
                    darker[3] = 0.4
                    edge_colors.append(darker)
                    linewidths.append(0.5)

        pc = PatchCollection(patches, match_original=False)
        pc.set_facecolors(face_colors)
        pc.set_edgecolors(edge_colors)
        pc.set_linewidths(linewidths)
        pc.set_zorder(2)
        ax.add_collection(pc)

        # Stained glass grid lines
        ax.hlines([r / n_rows for r in range(n_rows + 1)], 0, 1,
                  colors=grid_color, linewidth=0.8, zorder=1)
        ax.vlines([c / n_cols for c in range(n_cols + 1)], 0, 1,
                  colors=grid_color, linewidth=0.8, zorder=1)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
