"""Downloading, loading and labelling the NASA C-MAPSS data."""

import io
import ssl
import urllib.request
import zipfile

import pandas as pd

from .config import DATA_RAW, RAW_COLS

DATA_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)


def download_cmapss(dest=DATA_RAW):
    """Download the C-MAPSS archive from the NASA PCoE mirror and extract the txt files."""
    dest.mkdir(parents=True, exist_ok=True)
    print("Downloading C-MAPSS (about 12 MB)...")
    # Use certifi's certificates when available (python.org builds on macOS ship without any)
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = None
    with urllib.request.urlopen(DATA_URL, context=context) as resp:
        outer = zipfile.ZipFile(io.BytesIO(resp.read()))

    # The archive contains a second zip (CMAPSSData.zip) with the actual files.
    inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    for name in inner.namelist():
        if name.endswith(".txt") and name != "readme.txt":
            (dest / name).write_bytes(inner.read(name))
    print(f"Saved files to {dest.relative_to(dest.parents[1])}")


def load_cmapss(subset="FD001", data_dir=DATA_RAW):
    """Return (train, test, rul) for one C-MAPSS subset.

    train, test : DataFrames with columns unit, cycle, setting_1..3, s_1..s_21
    rul         : Series with the true RUL at the last test cycle, indexed by unit
    """
    if not (data_dir / f"train_{subset}.txt").exists():
        download_cmapss(data_dir)

    def read(name):
        return pd.read_csv(data_dir / name, sep=r"\s+", header=None, names=RAW_COLS)

    train = read(f"train_{subset}.txt")
    test = read(f"test_{subset}.txt")
    rul = pd.read_csv(data_dir / f"RUL_{subset}.txt", header=None).iloc[:, 0]
    rul.index = pd.RangeIndex(1, len(rul) + 1, name="unit")
    rul.name = "RUL"
    return train, test, rul


def add_train_rul(train):
    """Training engines run to failure, so RUL = last cycle of the engine - current cycle."""
    out = train.copy()
    out["RUL"] = out.groupby("unit")["cycle"].transform("max") - out["cycle"]
    return out


def add_test_rul(test, rul):
    """Label every test cycle: true RUL at the last cycle plus the cycles still to come."""
    out = test.copy()
    last_cycle = out.groupby("unit")["cycle"].transform("max")
    out["RUL"] = out["unit"].map(rul) + (last_cycle - out["cycle"])
    return out


def last_cycle(df):
    """Keep only the final observed cycle of every engine."""
    return df.loc[df.groupby("unit")["cycle"].idxmax()].reset_index(drop=True)
