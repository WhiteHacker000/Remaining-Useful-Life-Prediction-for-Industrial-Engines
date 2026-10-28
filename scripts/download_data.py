"""Download the NASA C-MAPSS data into data/raw/.

Usage (from the project root):  python scripts/download_data.py
The notebooks also call this automatically if the files are missing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import download_cmapss  # noqa: E402

if __name__ == "__main__":
    download_cmapss()
