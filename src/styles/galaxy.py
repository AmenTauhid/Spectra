import numpy as np

from .base import StyleRenderer
from ..palettes import sample_colors
from ..utils import resample_to_length, resample_2d_time_axis, polar_to_cartesian


class ParticleGalaxyRenderer(StyleRenderer):
    name = "galaxy"

    def render(self, fig):
        ax = self._setup_axes(fig)
        f = self.features

        n_time = 600
        stft = resample_2d_time_axis(f.stft_mag, n_time)
        n_freq = stft.shape[0]
        rms = resample_to_length(f.rms, n_time)
        chroma = resample_to_length(f.dominant_chroma.astype(np.float64), n_time)

        # Particles per frame: 10-30 proportional to RMS
        particles_per_frame = (10 + (rms * 20)).astype(int)
        total_particles = int(particles_per_frame.sum())

        # Repeat frame indices for each particle
        frame_indices = np.repeat(np.arange(n_time), particles_per_frame)

        # Weighted frequency sampling: for each particle, pick a frequency bin
        # weighted by STFT magnitude at that frame
        cumsum = np.cumsum(stft, axis=0)
        totals = cumsum[-1, :]
        totals = np.where(totals < 1e-10, 1.0, totals)
        norm_cumsum = cumsum / totals[np.newaxis, :]

        # Random values for frequency selection
        rand_vals = self.rng.uniform(0, 1, total_particles)
        frame_vals = norm_cumsum[:, frame_indices]  # (n_freq, total_particles)
        freq_indices = np.argmax(frame_vals >= rand_vals[np.newaxis, :], axis=0)

        # Position: frequency -> radius, time -> angle (spiral)
        r_base = freq_indices / n_freq
        r_jitter = self.rng.exponential(0.05, total_particles)
        r = np.clip((r_base + r_jitter) * 0.9, 0, 0.95)

        base_angles = (frame_indices / n_time) * 2 * np.pi * 3  # 3 full rotations
        angle_jitter = self.rng.normal(0, 0.15, total_particles)
        theta = base_angles + angle_jitter

        x, y = polar_to_cartesian(r, theta)

        # Size from amplitude
        amp = stft[freq_indices, frame_indices]
        sizes = (amp ** 0.5) * 80 + 2

        # Color from chromagram
        chroma_vals = chroma[frame_indices] / 11.0
        colors = sample_colors(self.palette, chroma_vals)
        colors[:, 3] = np.clip(amp * 0.7 + 0.1, 0, 1)

        # Background glow layer
        glow_colors = colors.copy()
        glow_colors[:, 3] *= 0.04
        ax.scatter(x, y, s=sizes * 3, c=glow_colors, edgecolors='none', zorder=1)

        # Main particles
        ax.scatter(x, y, s=sizes, c=colors, edgecolors='none', zorder=2)

        # Beat bursts
        beat_x_pos = f.beat_frames / f.n_frames
        if len(beat_x_pos) > 0:
            n_burst = 30
            all_bx = []
            all_by = []
            all_bs = []

            beat_centroid = f.spectral_centroid[f.beat_frames]
            for i, bframe in enumerate(f.beat_frames):
                ring_r = 0.2 + beat_centroid[i] * 0.6
                angles = np.linspace(0, 2 * np.pi, n_burst, endpoint=False)
                angles += self.rng.normal(0, 0.1, n_burst)
                bx_pts, by_pts = polar_to_cartesian(
                    ring_r + self.rng.normal(0, 0.03, n_burst), angles
                )
                all_bx.append(bx_pts)
                all_by.append(by_pts)
                all_bs.append(np.full(n_burst, 100 + f.rms[min(bframe, f.n_frames - 1)] * 100))

            all_bx = np.concatenate(all_bx)
            all_by = np.concatenate(all_by)
            all_bs = np.concatenate(all_bs)
            burst_colors = sample_colors(self.palette, np.full(len(all_bx), 0.85))
            burst_colors[:, 3] = 0.6
            ax.scatter(all_bx, all_by, s=all_bs, c=burst_colors,
                       edgecolors='none', zorder=3)

        # Central glow
        n_center = 80
        cx = self.rng.normal(0, 0.06, n_center)
        cy = self.rng.normal(0, 0.06, n_center)
        cs = self.rng.uniform(200, 500, n_center)
        center_colors = sample_colors(self.palette, np.full(n_center, 0.5))
        center_colors[:, 3] = 0.03
        ax.scatter(cx, cy, s=cs, c=center_colors, edgecolors='none', zorder=0)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
