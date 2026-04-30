import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .spikes import get_spike_info

logger = logging.getLogger(__name__)


def plot_all_features(tsdf, features_per_subplot=3):
    """
    Plot all features from a TsdFrame.

    Parameters
    ----------
    tsdf : nap.TsdFrame
        The data to plot.
    features_per_subplot : int, optional
        The number of features to plot per subplot, by default 3

    Returns
    -------
    tuple
        A tuple containing the figure and axes.
    """
    n_features = tsdf.d.shape[1]
    n_subplots = int(np.ceil(n_features / features_per_subplot))
    y_values = np.asarray(tsdf.d, dtype=float)
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))
    logger.info(
        "Plotting %s features across %s subplots",
        n_features,
        n_subplots,
    )
    logger.info("Using shared y-limits: [%s, %s]", y_min, y_max)
    ncols = int(np.ceil(np.sqrt(n_subplots)))
    nrows = int(np.ceil(n_subplots / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 3.5 * nrows),
        sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()

    columns = (
        list(tsdf.columns)
        if hasattr(tsdf, "columns")
        else [f"Feature {i}" for i in range(n_features)]
    )

    for subplot_idx, ax in enumerate(axes):
        start = subplot_idx * features_per_subplot
        end = min(start + features_per_subplot, n_features)

        if start >= n_features:
            ax.set_visible(False)
            continue

        subplot_data = pd.DataFrame(
            np.asarray(tsdf.d[:, start:end], dtype=float),
            columns=columns[start:end],
        )
        subplot_data.insert(0, "Time (s)", np.asarray(tsdf.t, dtype=float))
        subplot_data = subplot_data.melt(
            id_vars="Time (s)",
            var_name="Feature",
            value_name="Value",
        )

        sns.lineplot(
            data=subplot_data,
            x="Time (s)",
            y="Value",
            hue="Feature",
            ax=ax,
            legend=True,
        )
        ax.set_title(f"Features {start} to {end - 1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Value")
        y_span = y_max - y_min
        ax.set_ylim(y_min - 0.05 * y_span, y_max + 0.05 * y_span)
        legend = ax.get_legend()
        if legend is not None:
            sns.move_legend(ax, "best", frameon=False)

    fig.tight_layout()
    logger.info("Created feature plot figure")
    return (
        fig,
        axes,
    )


def plot_trial_pairs(time_values, trial_pairs, autoscale=True):
    n_trials = len(trial_pairs)
    ncols = int(np.ceil(np.sqrt(n_trials)))
    nrows = int(np.ceil(n_trials / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 3.5 * nrows),
        sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()

    if autoscale:
        pair_values = []
        for trial_pair in trial_pairs:
            pair_values.append(np.asarray(trial_pair["current"], dtype=float))
            pair_values.append(np.asarray(trial_pair["voltage"], dtype=float))
        all_values = np.concatenate(pair_values)
        y_min = float(np.nanmin(all_values))
        y_max = float(np.nanmax(all_values))
        y_span = y_max - y_min
        if y_span == 0:
            y_span = 1.0
    else:
        y_min = None
        y_max = None
        y_span = None

    for ax, trial_pair in zip(axes, trial_pairs):
        subplot_data = pd.DataFrame(
            {
                "Time (s)": np.asarray(time_values, dtype=float),
                trial_pair["current_name"]: np.asarray(
                    trial_pair["current"], dtype=float
                ),
                trial_pair["voltage_name"]: np.asarray(
                    trial_pair["voltage"], dtype=float
                ),
            }
        )
        subplot_data = subplot_data.melt(
            id_vars="Time (s)",
            var_name="Feature",
            value_name="Value",
        )
        sns.lineplot(
            data=subplot_data,
            x="Time (s)",
            y="Value",
            hue="Feature",
            ax=ax,
            legend=True,
        )
        ax.set_title(f"Trial {trial_pair['trial_index'] + 1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Value")
        if autoscale:
            ax.set_ylim(y_min - 0.05 * y_span, y_max + 0.05 * y_span)
        legend = ax.get_legend()
        if legend is not None:
            sns.move_legend(ax, "best", frameon=False)

    for ax in axes[n_trials:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig, axes


def plot_spike_info(trace, title="Spike Information"):
    """
    Plot spike information from a trace.

    Parameters
    ----------
    trace : nap.Tsd
        The trace to plot.
    title : str
        The title of the plot.

    Returns
    -------
    tuple
        A tuple containing the figure and axes.
    """
    peak_times, peak_amplitudes, peak_properties = get_spike_info(trace)
    fig, ax = plt.subplots()
    time = np.asarray(trace.t, dtype=float)
    signal = np.asarray(trace.d, dtype=float)
    ax.plot(time, signal, label="Trace")
    ax.scatter(peak_times, peak_amplitudes, color="red", label="Peaks")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(frameon=False)
    return fig, ax
