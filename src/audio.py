from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from numpy.typing import NDArray

from src.utils import normalize, normalize_percentile

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}


@dataclass
class AudioFeatures:
    sr: int
    duration: float
    n_frames: int
    stft_mag: NDArray[np.floating]       # (n_freq_bins, n_frames)
    chromagram: NDArray[np.floating]      # (12, n_frames)
    rms: NDArray[np.floating]            # (n_frames,)
    spectral_centroid: NDArray[np.floating]  # (n_frames,)
    onset_env: NDArray[np.floating]      # (n_frames,)
    onset_frames: NDArray[np.intp]       # (n_onsets,)
    tempo: float
    beat_frames: NDArray[np.intp]        # (n_beats,)
    dominant_chroma: NDArray[np.intp]    # (n_frames,)
    stft_freq_bins: NDArray[np.floating] # (n_freq_bins,)
    waveform: NDArray[np.floating]       # raw mono waveform


def extract_features(audio_path: Path, sr: int = 22050,
                     n_fft: int = 2048, hop_length: int = 512) -> AudioFeatures:
    """Load audio and extract all features in a single pass."""
    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    duration = len(y) / sr

    # STFT (compute once, reuse magnitude)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    stft_mag = np.abs(S)

    # Chromagram from STFT magnitude
    chromagram = librosa.feature.chroma_stft(S=stft_mag, sr=sr, hop_length=hop_length)

    # RMS energy
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]

    # Onset detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length, onset_envelope=onset_env
    )

    # Tempo and beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)

    # Align all time-series to consistent n_frames
    n_frames = min(stft_mag.shape[1], chromagram.shape[1],
                   len(rms), len(spectral_centroid), len(onset_env))
    stft_mag = stft_mag[:, :n_frames]
    chromagram = chromagram[:, :n_frames]
    rms = rms[:n_frames]
    spectral_centroid = spectral_centroid[:n_frames]
    onset_env = onset_env[:n_frames]

    # Filter beat/onset frames to valid range
    beat_frames = beat_frames[beat_frames < n_frames]
    onset_frames = onset_frames[onset_frames < n_frames]

    # Normalize features to [0, 1]
    stft_mag = normalize_percentile(stft_mag, percentile=98.0)
    chromagram = normalize(chromagram)
    rms = normalize_percentile(rms, percentile=98.0)
    spectral_centroid = normalize_percentile(spectral_centroid, percentile=98.0)
    onset_env = normalize(onset_env)

    # Derived
    dominant_chroma = np.argmax(chromagram, axis=0)
    stft_freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    return AudioFeatures(
        sr=sr, duration=duration, n_frames=n_frames,
        stft_mag=stft_mag, chromagram=chromagram, rms=rms,
        spectral_centroid=spectral_centroid, onset_env=onset_env,
        onset_frames=onset_frames, tempo=tempo, beat_frames=beat_frames,
        dominant_chroma=dominant_chroma, stft_freq_bins=stft_freq_bins,
        waveform=y,
    )


def load_audio_paths(input_path: Path) -> list[Path]:
    """Return list of audio files from a file path or directory."""
    if input_path.is_file():
        if input_path.suffix.lower() in AUDIO_EXTENSIONS:
            return [input_path]
        raise ValueError(f"Unsupported audio format: {input_path.suffix}")
    if input_path.is_dir():
        files = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not files:
            raise ValueError(f"No audio files found in {input_path}")
        return files
    raise ValueError(f"Path does not exist: {input_path}")
