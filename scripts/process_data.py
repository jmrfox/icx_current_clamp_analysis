from icxcca import *
import pynapple as nap
from pathlib import Path
import numpy as np

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

raw_filenames = [
    "2022_04_12_0002.csv",
    "2022_04_12_0004.csv",
    "2022_04_13_0005.csv",
    "2022_04_14_0001.csv",
    "2022_04_14_0002.csv",
    "2022_04_14_0003.csv",
    "2022_04_28_0012.csv",
    "2022_04_28_0015.csv",
    "2022_04_28_0016.csv",
]

need_rescale = [
    "2022_04_12_0002.csv",
    "2022_04_12_0004.csv",
    "2022_04_14_0001.csv",
    "2022_04_14_0002.csv",
    "2022_04_14_0003.csv",
]

raw_data_dir = "data/raw"
proc_data_dir = "data/processed"
raw_filepaths = [Path(raw_data_dir) / Path(f) for f in raw_filenames]

for fp, fn in zip(raw_filepaths, raw_filenames):
    dm = DataManager(fp)
    if fn in need_rescale:
        dm.rescale_data()
    dm.add_stimulus_data()
    proc_fp = Path(proc_data_dir) / Path(fn)
    dm.write_csv(proc_fp)
    logger.info(f"Processed {fn} and saved to {proc_fp}")
