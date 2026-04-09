import numpy as np
import soundfile as sf
from pathlib import Path
import pytest

from src.audio import extract_features, AudioFeatures, load_audio_paths
from src.palettes import get_palette, PALETTES, sample_colors
from src.render import render_artwork, STYLE_MAP
from src.utils import normalize, normalize_percentile, resample_to_length, smooth


def generate_sine_sweep(path: Path, duration: float = 5.0, sr: int = 22050) -> Path:
    """Generate a sine wave sweeping 200-4000 Hz with amplitude modulation."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = np.linspace(200, 4000, len(t))
    phase = np.cumsum(2 * np.pi * freq / sr)
    amplitude = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
    y = (amplitude * np.sin(phase)).astype(np.float32)
    sf.write(str(path), y, sr)
    return path


def generate_beat_pattern(path: Path, duration: float = 5.0, sr: int = 22050,
                          bpm: float = 120.0) -> Path:
    """Generate click track at given BPM."""
    n_samples = int(sr * duration)
    y = np.zeros(n_samples, dtype=np.float32)
    beat_interval = 60.0 / bpm
    click_len = int(0.01 * sr)
    click = np.exp(-np.linspace(0, 10, click_len)).astype(np.float32)
    for beat_time in np.arange(0, duration, beat_interval):
        idx = int(beat_time * sr)
        end = min(idx + click_len, n_samples)
        y[idx:end] += click[:end - idx]
    y = y / np.max(np.abs(y)) * 0.8
    sf.write(str(path), y, sr)
    return path


# --- Utils tests ---

class TestUtils:
    def test_normalize_basic(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize(arr)
        assert abs(result[0]) < 1e-10
        assert abs(result[-1] - 1.0) < 1e-10

    def test_normalize_constant(self):
        arr = np.array([3.0, 3.0, 3.0])
        result = normalize(arr)
        assert np.all(result == 0.0)

    def test_normalize_percentile_clips(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        result = normalize_percentile(arr, percentile=80)
        assert abs(result[-1] - 1.0) < 1e-10

    def test_resample_to_length(self):
        arr = np.array([0.0, 1.0, 2.0, 3.0])
        result = resample_to_length(arr, 7)
        assert len(result) == 7
        assert abs(result[0]) < 1e-10
        assert abs(result[-1] - 3.0) < 1e-10

    def test_smooth(self):
        arr = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        result = smooth(arr, window=3)
        assert len(result) == len(arr)
        assert result[2] < 1.0  # Smoothed down from spike


# --- Audio tests ---

class TestAudioExtraction:
    @pytest.fixture
    def sweep_file(self, tmp_path):
        return generate_sine_sweep(tmp_path / "sweep.wav")

    def test_extract_returns_features(self, sweep_file):
        features = extract_features(sweep_file)
        assert isinstance(features, AudioFeatures)

    def test_feature_shapes(self, sweep_file):
        f = extract_features(sweep_file)
        assert f.rms.shape == (f.n_frames,)
        assert f.spectral_centroid.shape == (f.n_frames,)
        assert f.onset_env.shape == (f.n_frames,)
        assert f.stft_mag.shape[1] == f.n_frames
        assert f.chromagram.shape == (12, f.n_frames)
        assert f.dominant_chroma.shape == (f.n_frames,)

    def test_features_normalized(self, sweep_file):
        f = extract_features(sweep_file)
        assert f.rms.min() >= 0.0
        assert f.rms.max() <= 1.0 + 1e-6
        assert f.spectral_centroid.min() >= 0.0
        assert f.spectral_centroid.max() <= 1.0 + 1e-6

    def test_duration(self, sweep_file):
        f = extract_features(sweep_file)
        assert abs(f.duration - 5.0) < 0.5

    def test_load_audio_paths_file(self, sweep_file):
        paths = load_audio_paths(sweep_file)
        assert len(paths) == 1

    def test_load_audio_paths_dir(self, tmp_path):
        generate_sine_sweep(tmp_path / "a.wav")
        generate_sine_sweep(tmp_path / "b.wav")
        paths = load_audio_paths(tmp_path)
        assert len(paths) == 2


# --- Palette tests ---

class TestPalettes:
    def test_all_palettes_exist(self):
        for name in ["sunset", "ocean", "neon", "monochrome", "pastel"]:
            p = get_palette(name)
            assert p.name == name
            assert len(p.colors) >= 4

    def test_sample_colors_shape(self):
        p = get_palette("sunset")
        values = np.linspace(0, 1, 100)
        colors = sample_colors(p, values)
        assert colors.shape == (100, 4)

    def test_auto_palette(self, tmp_path):
        sweep = generate_sine_sweep(tmp_path / "sweep.wav")
        f = extract_features(sweep)
        p = get_palette("auto", audio_features=f)
        assert p.name == "auto"
        assert len(p.colors) >= 4


# --- Rendering tests ---

class TestRendering:
    @pytest.fixture(scope="class")
    def features(self, tmp_path_factory):
        path = tmp_path_factory.mktemp("audio") / "sweep.wav"
        generate_sine_sweep(path)
        return extract_features(path)

    @pytest.fixture
    def palette(self):
        return get_palette("neon")

    @pytest.mark.parametrize("style_name", ["radial", "terrain", "galaxy", "mosaic", "ribbon"])
    def test_render_png(self, features, palette, style_name, tmp_path):
        out = tmp_path / f"{style_name}.png"
        result = render_artwork(features, style_name, palette, out, size=800, seed=42)
        assert result.exists()
        assert result.stat().st_size > 10000

    @pytest.mark.parametrize("style_name", ["radial", "terrain", "galaxy", "mosaic", "ribbon"])
    def test_render_svg(self, features, palette, style_name, tmp_path):
        out = tmp_path / f"{style_name}.svg"
        result = render_artwork(features, style_name, palette, out, size=800, seed=42)
        assert result.exists()
        assert result.stat().st_size > 1000

    def test_random_style(self, features, palette, tmp_path):
        out = tmp_path / "random.png"
        result = render_artwork(features, "random", palette, out, size=800, seed=42)
        assert result.exists()

    def test_deterministic_with_seed(self, features, palette, tmp_path):
        out1 = tmp_path / "det1.png"
        out2 = tmp_path / "det2.png"
        render_artwork(features, "galaxy", palette, out1, size=400, seed=123)
        render_artwork(features, "galaxy", palette, out2, size=400, seed=123)
        assert out1.read_bytes() == out2.read_bytes()
