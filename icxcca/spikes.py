import numpy as np
from scipy.signal import find_peaks


def get_spike_info(trace, threshold=None):
    """
    Get spike information from a trace.

    Parameters
    ----------
    trace : nap.Tsd
        The trace to analyze.
    threshold : float, optional
        The threshold for spike detection.
        By default (None) 0.8 of the amplitude.

    Returns
    -------
    tuple
        A tuple containing the spike times, spike amplitudes, and properties.
    """
    time = np.asarray(trace.t, dtype=float)
    signal = np.asarray(trace.d, dtype=float)
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    n_samples = len(signal)
    if n_samples < 3:
        empty_indices = np.array([], dtype=int)
        empty_properties = {
            "peak_heights": np.array([], dtype=float),
            "prominences": np.array([], dtype=float),
            "second_derivative": np.array([], dtype=float),
        }
        return time[empty_indices], signal[empty_indices], empty_properties

    signal_range = signal_max - signal_min
    second_derivative = np.gradient(np.gradient(signal))
    noise_level = np.median(np.abs(signal - np.median(signal)))
    curvature_level = np.median(np.abs(second_derivative))

    if threshold is None:
        threshold = signal_max - 0.2 * signal_range

    min_distance = max(1, n_samples // 100)
    min_prominence = max(signal_range * 0.1, noise_level * 6)
    indices, properties = find_peaks(
        signal,
        height=threshold,
        distance=min_distance,
        prominence=min_prominence,
    )

    if indices.size == 0:
        properties["second_derivative"] = np.array([], dtype=float)
        return time[indices], signal[indices], properties

    peak_curvature = np.abs(second_derivative[indices])
    min_curvature = max(curvature_level * 8, 1e-12)
    keep_mask = peak_curvature >= min_curvature
    indices = indices[keep_mask]
    properties = {
        key: np.asarray(value)[keep_mask] for key, value in properties.items()
    }
    properties["second_derivative"] = peak_curvature[keep_mask]
    spike_times = time[indices]
    spike_amplitudes = signal[indices]
    return spike_times, spike_amplitudes, properties
