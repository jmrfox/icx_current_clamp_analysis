import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import seaborn as sns
from pathlib import Path


logger = logging.getLogger(__name__)


def get_data(filename):
    logger.info("Loading data from %s", filename)
    df = pd.read_csv(filename, sep=",", header=1)
    n_sweep = len(df.columns) - 2
    logger.info("Loaded %s features from %s", n_sweep, filename)
    data = nap.TsdFrame(
        t=np.array(df["Time (s)"].values, dtype="float"),
        d=np.array(df.iloc[:, 1 : (n_sweep + 1)].values, dtype="float"),
        time_units="s",
        columns=df.columns[1 : (n_sweep + 1)],
    )
    return data


def plot_all_features(tsdf, features_per_subplot=3):
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


class DataManager:
    def __init__(self, filepath):
        self.filepath = filepath
        logger.info("Initializing DataManager for %s", filepath)
        self.data = get_data(filepath)

    def figure_filename(self, suffix="png"):
        extension = suffix if suffix.startswith(".") else f".{suffix}"
        filename = Path(self.filepath).with_suffix(extension).name
        logger.info(
            "Derived figure filename %s from %s",
            filename,
            self.filepath,
        )
        return filename

    def plot(self, features_per_subplot=3, autosave=False):
        logger.info(
            "Plot requested for %s with features_per_subplot=%s, autosave=%s",
            self.filepath,
            features_per_subplot,
            autosave,
        )
        fig, axes = plot_all_features(
            self.data,
            features_per_subplot=features_per_subplot,
        )
        if autosave:
            save_filename = self.figure_filename()
            logger.info("Saving figure to %s", save_filename)
            fig.savefig(save_filename)
        return (
            fig,
            axes,
        )

    def get_resting_potentials(self, duration_ms=500):
        logger.info(
            "Calculating resting potentials for %s over first %s ms",
            self.filepath,
            duration_ms,
        )
        duration_s = duration_ms / 1000
        start_time = float(self.data.t[0])
        end_time = start_time + duration_s
        mask = (self.data.t >= start_time) & (self.data.t <= end_time)

        resting_values = np.asarray(self.data.d[mask], dtype=float)
        if resting_values.size == 0:
            logger.warning(
                "No samples found in resting potential window for %s",
                self.filepath,
            )
            raise ValueError("No samples found in the resting potential window.")
        resting_potentials = resting_values.mean(axis=0)
        logger.info(
            "Calculated resting potentials for %s features",
            len(self.data.columns),
        )
        return pd.Series(
            resting_potentials,
            index=self.data.columns,
            name="resting_potential",
        )
