import logging
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from pathlib import Path

from .spikes import get_spike_info as compute_spike_info
from .viz import plot_all_features, plot_trial_pairs

logger = logging.getLogger(__name__)


def get_data(filename):
    """
    Load data from a CSV file.

    Parameters
    ----------
    filename : str
        The path to the CSV file.

    Returns
    -------
    nap.TsdFrame
        The loaded data.
    """
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


def get_feature_subset(tsdf, feature_names):
    columns = list(tsdf.columns)
    missing_features = [
        feature_name for feature_name in feature_names if feature_name not in columns
    ]
    if missing_features:
        message = f"Features not found in data columns: {missing_features}"
        raise ValueError(message)
    return tsdf[feature_names].copy()


class DataManager:
    """
    A class for managing voltage recordings, current stimuli, and plotting.

    Parameters
    ----------
    filepath : str
        The path to the data file.
    """

    def __init__(self, filepath):
        """
        Initialize the DataManager.

        Parameters
        ----------
        filepath : str
            The path to the data file.
        """
        self.filepath = Path(filepath)
        logger.info("Initializing DataManager for %s", filepath)
        if self.filepath.suffix == ".npz":
            self.data = nap.load_file(self.filepath)
        else:
            self.data = get_data(self.filepath)
        self.stimulus = None
        self.stimulus_epoch = None

    def figure_filename(self, suffix="png"):
        """
        Generate a figure filename based on the data file path.

        Parameters
        ----------
        suffix : str, optional
            The file extension for the figure, by default "png"

        Returns
        -------
        str
            The derived figure filename.
        """
        extension = suffix if suffix.startswith(".") else f".{suffix}"
        filename = Path(self.filepath).with_suffix(extension).name
        logger.info(
            "Derived figure filename %s from %s",
            filename,
            self.filepath,
        )
        return filename

    def plot(
        self,
        features_per_subplot=2,
        autosave=False,
        feature_names=None,
        trial_indices=None,
    ):
        """
        Plot the data.

        Parameters
        ----------
        features_per_subplot : int, optional
            The number of features to plot per subplot, by default 2
        autosave : bool, optional
            Whether to automatically save the figure, by default False
        feature_names : list[str] | None, optional
            Explicit feature names to plot.
        trial_indices : list[int] | None, optional
            Sweep indices to plot as paired current and voltage traces.

        Returns
        -------
        tuple
            A tuple containing the figure and axes.
        """
        if feature_names is not None and trial_indices is not None:
            message = "Specify either feature_names or trial_indices, not both."
            raise ValueError(message)
        logger.info(
            "Plot requested for %s with features_per_subplot=%s, autosave=%s, "
            "feature_names=%s, trial_indices=%s",
            self.filepath,
            features_per_subplot,
            autosave,
            feature_names,
            trial_indices,
        )
        if feature_names is not None:
            plot_data = get_feature_subset(self.data, feature_names)
            fig, axes = plot_all_features(
                plot_data,
                features_per_subplot=features_per_subplot,
            )
        elif trial_indices is not None:
            fig, axes = self._plot_trials(trial_indices)
        else:
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

    def rescale_data(self, scale_factor=0.05):
        """
        Rescale the voltage data by the specified scale factor.

        Parameters
        ----------
        scale_factor : float, optional
            The scale factor to apply, by default 0.05

        Returns
        -------
        nap.TsdFrame
            The rescaled data.
        """
        logger.info("Rescaling voltage data by %s for %s", scale_factor, self.filepath)
        rescaled_values = np.asarray(self.data.d, dtype=float) * scale_factor
        self.data = nap.TsdFrame(
            t=np.asarray(self.data.t, dtype=float),
            d=rescaled_values,
            time_units="s",
            columns=self.data.columns,
        )
        return self.data

    def add_current_data(self, start_time=1.234, end_time=1.734):
        """
        Generate current stimulus data for each voltage recording.

        Parameters
        ----------
        start_time : float, optional
            The start time of the stimulus epoch, by default 1.234
        end_time : float, optional
            The end time of the stimulus epoch, by default 1.734

        Returns
        -------
        nap.TsdFrame
            The generated current traces.
        """
        logger.info(
            "Generating current data for %s from %s to %s s",
            self.filepath,
            start_time,
            end_time,
        )
        time_values = np.asarray(self.data.t, dtype=float)
        voltage_columns = list(self.data.columns)
        shape = (len(time_values), len(voltage_columns))
        current_values = np.zeros(shape, dtype=float)
        active_mask = (time_values >= start_time) & (time_values <= end_time)

        for column_idx, voltage_name in enumerate(voltage_columns):
            trial_matches = re.findall(r"\d+", str(voltage_name))
            if not trial_matches:
                raise ValueError(
                    "Could not determine trial index from column name: "
                    f"{voltage_name}"
                )

            trial_index = int(trial_matches[-1])
            current_amplitude = -50 + 10 * (trial_index - 1)
            current_values[active_mask, column_idx] = current_amplitude

        self.stimulus = nap.TsdFrame(
            t=time_values,
            d=current_values,
            time_units="s",
            columns=voltage_columns,
        )
        self.stimulus_epoch = nap.IntervalSet(
            start=[start_time],
            end=[end_time],
            time_units="s",
        )
        logger.info(
            "Generated current data for %s voltage recordings",
            len(voltage_columns),
        )
        return self.stimulus

    def add_stimulus_data(self, start_time=1.234, end_time=1.734):
        """
        Backward-compatible wrapper for `add_current_data`.

        Parameters
        ----------
        start_time : float, optional
            The start time of the stimulus epoch, by default 1.234.
        end_time : float, optional
            The end time of the stimulus epoch, by default 1.734.

        Returns
        -------
        nap.TsdFrame
            The generated current traces.
        """
        return self.add_current_data(
            start_time=start_time,
            end_time=end_time,
        )

    def get_stimulus_epoch(self, start_time=1.234, end_time=1.734):
        """
        Get the epoch during which nonzero current was applied.

        Parameters
        ----------
        start_time : float, optional
            The start time of the stimulus epoch, by default 1.234.
        end_time : float, optional
            The end time of the stimulus epoch, by default 1.734.

        Returns
        -------
        nap.IntervalSet
            The current-application epoch.
        """
        if self.stimulus_epoch is None:
            self.add_current_data(start_time=start_time, end_time=end_time)
        return self.stimulus_epoch

    def write_npz(self, output_filepath):
        """
        Write the data to a numpy file.

        Parameters
        ----------
        output_filepath : str
            The path to the output file.

        Returns
        -------
        Path
            The path to the output file.
        """
        output_path = Path(output_filepath).with_suffix(".npz")
        logger.info("Writing pynapple data to %s", output_path)
        self.data.save(output_path)
        logger.info("Wrote pynapple data to %s", output_path)
        return output_path

    def load_npz(self, input_filepath):
        """
        Load the data from a numpy file.

        Parameters
        ----------
        input_filepath : str
            The path to the input file.

        Returns
        -------
        Data
            The loaded data.
        """
        input_path = Path(input_filepath).with_suffix(".npz")
        logger.info("Loading pynapple data from %s", input_path)
        self.data = nap.load_file(input_path)
        logger.info("Loaded pynapple data from %s", input_path)
        return self.data

    def get_current_data(self):
        """
        Get the current stimulus data from the dataset.

        Returns
        -------
        nap.TsdFrame
            The current data.
        """
        columns = [
            column
            for column in self.data.columns
            if str(column).startswith(("Stimulus", "Current"))
        ]
        n_columns = len(columns)
        logger.info("Extracting %s current features", n_columns)
        return self.data[columns].copy()

    def get_voltage_data(self):
        """
        Get the recorded voltage data from the dataset.

        Returns
        -------
        nap.TsdFrame
            The voltage data.
        """
        columns = [
            column
            for column in self.data.columns
            if str(column).startswith(("Sweep", "Voltage"))
        ]
        n_columns = len(columns)
        logger.info("Extracting %s voltage features", n_columns)
        return self.data[columns].copy()

    def _resolve_voltage_features(self, feature_names=None, trial_indices=None):
        if feature_names is not None and trial_indices is not None:
            message = "Specify either feature_names or trial_indices, not both."
            raise ValueError(message)

        voltage_data = self.get_voltage_data()
        voltage_columns = list(voltage_data.columns)

        if feature_names is not None:
            selected_columns = list(
                get_feature_subset(voltage_data, feature_names).columns
            )
        elif trial_indices is not None:
            selected_columns = []
            for trial_index in trial_indices:
                if trial_index < 0:
                    message = "trial_indices must be non-negative."
                    raise ValueError(message)
                if trial_index >= len(voltage_columns):
                    message = (
                        f"Trial index {trial_index} is out of range "
                        "for voltage data."
                    )
                    raise ValueError(message)
                selected_columns.append(voltage_columns[trial_index])
        else:
            selected_columns = voltage_columns

        return voltage_data, selected_columns

    def get_spike_info(
        self,
        threshold=None,
        feature_names=None,
        trial_indices=None,
    ):
        """
        Compute spike information for selected voltage traces.

        Parameters
        ----------
        threshold : float, optional
            Detection threshold passed to the spike detector.
        feature_names : list[str] | None, optional
            Explicit voltage feature names to analyze.
        trial_indices : list[int] | None, optional
            Trial indices of voltage traces to analyze.

        Returns
        -------
        dict
            Mapping of voltage feature name to spike information.
        """
        voltage_data, selected_columns = self._resolve_voltage_features(
            feature_names=feature_names,
            trial_indices=trial_indices,
        )
        spike_info = {}
        for column_name in selected_columns:
            trace = voltage_data[column_name]
            spike_times, spike_amplitudes, properties = compute_spike_info(
                trace,
                threshold=threshold,
            )
            spike_info[column_name] = {
                "trace": trace,
                "spike_times": np.asarray(spike_times, dtype=float),
                "spike_amplitudes": np.asarray(spike_amplitudes, dtype=float),
                "properties": properties,
            }

        logger.info(
            "Computed spike information for %s voltage traces",
            len(spike_info),
        )
        return spike_info

    def get_spike_times(
        self,
        threshold=None,
        feature_names=None,
        trial_indices=None,
    ):
        """
        Get spike times for selected voltage traces as a TsGroup.

        Parameters
        ----------
        threshold : float, optional
            Detection threshold passed to the spike detector.
        feature_names : list[str] | None, optional
            Explicit voltage feature names to analyze.
        trial_indices : list[int] | None, optional
            Trial indices of voltage traces to analyze.

        Returns
        -------
        nap.TsGroup
            Spike times grouped by voltage feature.
        """
        spike_info = self.get_spike_info(
            threshold=threshold,
            feature_names=feature_names,
            trial_indices=trial_indices,
        )
        spike_trains = {
            column_name: nap.Ts(t=info["spike_times"], time_units="s")
            for column_name, info in spike_info.items()
        }
        return nap.TsGroup(spike_trains)

    def plot_spike_info(
        self,
        threshold=None,
        feature_names=None,
        trial_indices=None,
        autosave=False,
    ):
        """
        Plot selected voltage traces with detected spikes labelled.

        Parameters
        ----------
        threshold : float, optional
            Detection threshold passed to the spike detector.
        feature_names : list[str] | None, optional
            Explicit voltage feature names to plot.
        trial_indices : list[int] | None, optional
            Trial indices of voltage traces to plot.
        autosave : bool, optional
            Whether to automatically save the figure, by default False.

        Returns
        -------
        tuple
            A tuple containing the figure and axes.
        """
        spike_info = self.get_spike_info(
            threshold=threshold,
            feature_names=feature_names,
            trial_indices=trial_indices,
        )
        n_features = len(spike_info)
        if n_features == 0:
            raise ValueError("No voltage traces available for spike plotting.")

        ncols = int(np.ceil(np.sqrt(n_features)))
        nrows = int(np.ceil(n_features / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 3.5 * nrows),
            sharex=True,
        )
        axes = np.atleast_1d(axes).ravel()

        for ax, (column_name, info) in zip(axes, spike_info.items()):
            time_values = np.asarray(info["trace"].t, dtype=float)
            voltage_values = np.asarray(info["trace"].d, dtype=float)
            ax.plot(time_values, voltage_values, label=column_name)
            ax.scatter(
                info["spike_times"],
                info["spike_amplitudes"],
                color="red",
                label="Spikes",
                zorder=3,
            )
            ax.set_title(column_name)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Voltage")
            legend = ax.get_legend()
            if legend is None:
                ax.legend(frameon=False)

        for ax in axes[n_features:]:
            ax.set_visible(False)

        fig.tight_layout()
        if autosave:
            save_filename = self.figure_filename(suffix="spikes.png")
            logger.info("Saving spike figure to %s", save_filename)
            fig.savefig(save_filename)
        return fig, axes

    def _plot_trials(self, trial_indices):
        if len(trial_indices) == 0:
            message = "trial_indices must contain at least one trial index."
            raise ValueError(message)

        voltage_data = self.get_voltage_data()
        voltage_columns = list(voltage_data.columns)
        current_columns = [
            column
            for column in self.data.columns
            if str(column).startswith(("Stimulus", "Current"))
        ]
        if current_columns:
            current_data = self.get_current_data()
        else:
            if self.stimulus is None:
                self.add_current_data()
            current_data = self.stimulus.copy()
            current_columns = list(current_data.columns)

        trial_pairs = []
        for trial_index in trial_indices:
            if trial_index < 0:
                message = "trial_indices must be non-negative."
                raise ValueError(message)
            if trial_index >= len(voltage_columns):
                message = (
                    f"Trial index {trial_index} is out of range " "for voltage data."
                )
                raise ValueError(message)
            if trial_index >= len(current_columns):
                message = (
                    f"Trial index {trial_index} is out of range " "for current data."
                )
                raise ValueError(message)
            voltage_name = voltage_columns[trial_index]
            current_name = current_columns[trial_index]
            trial_pairs.append(
                {
                    "trial_index": trial_index,
                    "current_name": current_name,
                    "voltage_name": voltage_name,
                    "current": np.asarray(
                        current_data[current_name].d,
                        dtype=float,
                    ),
                    "voltage": np.asarray(
                        voltage_data[voltage_name].d,
                        dtype=float,
                    ),
                }
            )

        time_values = np.asarray(self.data.t, dtype=float)
        return plot_trial_pairs(time_values, trial_pairs)

    def write_csv(self, output_filepath, start_time=1.234, end_time=1.734):
        """
        Write the processed data to a CSV file.

        Parameters
        ----------
        output_filepath : str
            The path to the output file.
        start_time : float, optional
            The start time of the stimulus epoch, by default 1.234
        end_time : float, optional
            The end time of the stimulus epoch, by default 1.734

        Returns
        -------
        None
        """
        logger.info("Writing processed data to %s", output_filepath)
        if self.stimulus is None:
            self.add_current_data(start_time=start_time, end_time=end_time)

        time_values = np.asarray(self.data.t, dtype=float)
        voltage_values = np.asarray(self.data.d, dtype=float)
        current_values = np.asarray(self.stimulus.d, dtype=float)
        columns = list(self.data.columns)

        output_data = {"Time (s)": time_values}
        for column_idx, column_name in enumerate(columns, start=1):
            current_column = current_values[:, column_idx - 1]
            voltage_column = voltage_values[:, column_idx - 1]
            current_name = f"Current {column_idx}"
            voltage_name = f"Voltage {column_idx}"
            output_data[current_name] = current_column
            output_data[voltage_name] = voltage_column

        output_df = pd.DataFrame(output_data)
        output_df.to_csv(output_filepath, index=False)
        logger.info(
            "Wrote CSV with %s voltage/current trial pairs to %s",
            len(columns),
            output_filepath,
        )
        return output_df

    def get_resting_potentials(self, duration_ms=500):
        """
        Calculate the resting potentials for the data.

        Parameters
        ----------
        duration_ms : int, optional
            The duration of the resting potential window in milliseconds, by default 500

        Returns
        -------
        np.ndarray
            The resting potentials.
        """
        logger.info(
            "Calculating resting potentials for %s over first %s ms",
            self.filepath,
            duration_ms,
        )
        duration_s = duration_ms / 1000
        start_time = float(self.data.t[0])
        end_time = start_time + duration_s
        resting_data = self.data.get(start_time, end_time)
        resting_values = np.asarray(
            resting_data.d,
            dtype=float,
        )
        if resting_values.size == 0:
            warning_message = (
                "No samples found in resting potential window for %s. "
                "Please check the duration_ms parameter."
            )
            warning_path = self.filepath
            logger.warning(warning_message, warning_path)
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
