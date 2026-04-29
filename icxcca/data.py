import logging
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import seaborn as sns
from pathlib import Path


logger = logging.getLogger(__name__)


def get_data(filename):
    logger.info("Loading data from %s", filename)
    preview_df = pd.read_csv(filename, sep=",", header=None, nrows=2)
    if preview_df.empty:
        first_cell = ""
    else:
        first_cell = str(preview_df.iloc[0, 0]).strip()
    header_row = 1 if first_cell == "Sweep Start (s)" else 0
    logger.info("Detected CSV header row %s for %s", header_row, filename)
    df = pd.read_csv(filename, sep=",", header=header_row)
    data_columns = [
        column
        for column in df.columns[1:]
        if str(column).strip() and not str(column).startswith("Unnamed:")
    ]
    n_features = len(data_columns)
    logger.info("Loaded %s features from %s", n_features, filename)
    data_values = df.loc[:, data_columns].values
    data = nap.TsdFrame(
        t=np.array(df["Time (s)"].values, dtype="float"),
        d=np.array(data_values, dtype="float"),
        time_units="s",
        columns=data_columns,
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
        self.stimulus = None

    def figure_filename(self, suffix="png"):
        extension = suffix if suffix.startswith(".") else f".{suffix}"
        filename = Path(self.filepath).with_suffix(extension).name
        logger.info(
            "Derived figure filename %s from %s",
            filename,
            self.filepath,
        )
        return filename

    def plot(self, features_per_subplot=2, autosave=False):
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

    def rescale_data(self):
        logger.info("Rescaling voltage data by 1/20 for %s", self.filepath)
        rescaled_values = np.asarray(self.data.d, dtype=float) / 20
        self.data = nap.TsdFrame(
            t=np.asarray(self.data.t, dtype=float),
            d=rescaled_values,
            time_units="s",
            columns=self.data.columns,
        )
        return self.data

    def add_stimulus_data(self, start_time=1.234, end_time=1.734):
        logger.info(
            "Generating stimulus data for %s from %s to %s s",
            self.filepath,
            start_time,
            end_time,
        )
        time_values = np.asarray(self.data.t, dtype=float)
        columns = list(self.data.columns)
        shape = (len(time_values), len(columns))
        stimulus_values = np.zeros(shape, dtype=float)
        active_mask = (time_values >= start_time) & (time_values <= end_time)

        for column_idx, column_name in enumerate(columns):
            sweep_matches = re.findall(r"\d+", str(column_name))
            if not sweep_matches:
                raise ValueError(
                    "Could not determine sweep number from column name: "
                    f"{column_name}"
                )

            sweep_number = int(sweep_matches[-1])
            stimulus_amplitude = -50 + 10 * (sweep_number - 1)
            stimulus_values[active_mask, column_idx] = stimulus_amplitude

        self.stimulus = nap.TsdFrame(
            t=time_values,
            d=stimulus_values,
            time_units="s",
            columns=columns,
        )
        logger.info("Generated stimulus data for %s sweeps", len(columns))
        return self.stimulus

    def write_npz(self, output_filepath):
        output_path = Path(output_filepath).with_suffix(".npz")
        logger.info("Writing pynapple data to %s", output_path)
        self.data.save(output_path)
        logger.info("Wrote pynapple data to %s", output_path)
        return output_path

    def load_npz(self, input_filepath):
        input_path = Path(input_filepath).with_suffix(".npz")
        logger.info("Loading pynapple data from %s", input_path)
        self.data = nap.load_file(input_path)
        logger.info("Loaded pynapple data from %s", input_path)
        return self.data

    def write_csv(self, output_filepath, start_time=1.234, end_time=1.734):
        logger.info("Writing processed data to %s", output_filepath)
        if self.stimulus is None:
            self.add_stimulus_data(start_time=start_time, end_time=end_time)

        time_values = np.asarray(self.data.t, dtype=float)
        sweep_values = np.asarray(self.data.d, dtype=float)
        stimulus_values = np.asarray(self.stimulus.d, dtype=float)
        columns = list(self.data.columns)

        output_data = {"Time (s)": time_values}
        for column_idx, column_name in enumerate(columns, start=1):
            stimulus_column = stimulus_values[:, column_idx - 1]
            sweep_column = sweep_values[:, column_idx - 1]
            output_data[f"Stimulus {column_idx}"] = stimulus_column
            output_data[f"Sweep {column_idx}"] = sweep_column

        output_df = pd.DataFrame(output_data)
        output_df.to_csv(output_filepath, index=False)
        logger.info(
            "Wrote CSV with %s sweeps to %s",
            len(columns),
            output_filepath,
        )
        return output_df

    def get_resting_potentials(self, duration_ms=500):
        logger.info(
            "Calculating resting potentials for %s over first %s ms",
            self.filepath,
            duration_ms,
        )
        duration_s = duration_ms / 1000
        start_time = float(self.data.t[0])
        end_time = start_time + duration_s
        start_mask = self.data.t >= start_time
        end_mask = self.data.t <= end_time
        mask = start_mask & end_mask

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
