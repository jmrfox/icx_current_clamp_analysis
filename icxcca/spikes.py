from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt


def get_spike_info(trace, threshold=None):
    """ """
    time = trace.t
    signal = trace.d
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    n_samples = len(signal)
    if threshold is None:
        threshold = signal_max - 0.2 * (signal_max - signal_min)
    indices, properties = find_peaks(
        signal, height=threshold, distance=n_samples // 100
    )
    spike_times = time[indices]
    spike_amplitudes = signal[indices]
    return spike_times, spike_amplitudes, properties


# calculate spike info and plot
def plot_spike_info(trace, title):
    peak_times, peak_amplitudes, peak_properties = get_spike_info(trace)
    plt.figure()
    plt.plot(trace, marker="")
    plt.scatter(peak_times, peak_amplitudes, color="red", label="Peaks")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
