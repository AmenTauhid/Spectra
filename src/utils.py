import numpy as np
from numpy.typing import NDArray


def normalize(arr: NDArray[np.floating], low: float = 0.0, high: float = 1.0) -> NDArray[np.floating]:
    """Min-max normalize array to [low, high]. Returns zeros if constant."""
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max - arr_min < 1e-10:
        return np.full_like(arr, low, dtype=np.float64)
    return (arr - arr_min) / (arr_max - arr_min) * (high - low) + low


def normalize_percentile(
    arr: NDArray[np.floating], percentile: float = 98.0,
    low: float = 0.0, high: float = 1.0
) -> NDArray[np.floating]:
    """Normalize clipping outliers above given percentile before scaling."""
    arr = arr.astype(np.float64)
    p_val = np.percentile(arr, percentile)
    arr_min = arr.min()
    if p_val - arr_min < 1e-10:
        return np.full_like(arr, low, dtype=np.float64)
    clipped = np.clip(arr, arr_min, p_val)
    return (clipped - arr_min) / (p_val - arr_min) * (high - low) + low


def smooth(arr: NDArray[np.floating], window: int = 5) -> NDArray[np.floating]:
    """1D moving average smoothing."""
    if window < 2 or len(arr) < window:
        return arr.copy()
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(arr)]


def resample_to_length(arr: NDArray[np.floating], target_len: int) -> NDArray[np.floating]:
    """Resample 1D array to target length using linear interpolation."""
    if len(arr) == target_len:
        return arr.copy()
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, arr)


def resample_2d_time_axis(arr_2d: NDArray[np.floating], target_cols: int) -> NDArray[np.floating]:
    """Resample 2D array along axis=1 (time) to target_cols columns."""
    n_rows, n_cols = arr_2d.shape
    if n_cols == target_cols:
        return arr_2d.copy()
    result = np.empty((n_rows, target_cols), dtype=np.float64)
    x_old = np.linspace(0, 1, n_cols)
    x_new = np.linspace(0, 1, target_cols)
    for i in range(n_rows):
        result[i] = np.interp(x_new, x_old, arr_2d[i])
    return result


def polar_to_cartesian(
    r: NDArray[np.floating], theta: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Convert polar coordinates to cartesian (x, y)."""
    return r * np.cos(theta), r * np.sin(theta)
