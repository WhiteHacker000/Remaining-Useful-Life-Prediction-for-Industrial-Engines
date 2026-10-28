# Data: NASA C-MAPSS Turbofan Engine Degradation Simulation

This folder holds the raw C-MAPSS files used in the project. The FD001 subset is the main data set; FD002 is only used in notebooks 02 and 03 to demonstrate operating-regime normalisation.

## Source

| | |
|---|---|
| Data set | Turbofan Engine Degradation Simulation Data Set (C-MAPSS) |
| Creators | A. Saxena and K. Goebel, NASA Ames Research Center (2008) |
| Original host | NASA Prognostics Center of Excellence (PCoE) Data Repository |
| Mirror used | https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip |
| Archive size | about 12 MB (zip inside a zip; the inner `CMAPSSData.zip` contains the text files) |
| Reference paper | Saxena, Goebel, Simon and Eklund, "Damage propagation modeling for aircraft engine run-to-failure simulation", PHM 2008 |

If the files are missing, `src/data.py` downloads and extracts them automatically (or run `python scripts/download_data.py`).

## Why this data set

The project brief recommends C-MAPSS FD001. It meets the project requirements:

- **Size:** FD001 has 20,631 training rows and 13,096 test rows, well above the 3,000-row minimum.
- **Run-to-failure histories:** every training engine is observed until failure, so the RUL label can be computed exactly from the last cycle.
- **Engine ids:** every row carries an engine id, so train/validation splits can be made by engine rather than by row, as the brief requires.
- **Official test labels:** `RUL_FD001.txt` gives the true RUL of each test engine, so results are comparable with published work.

## Files

| File | Rows | Engines | Content |
|---|---|---|---|
| `train_FD001.txt` | 20,631 | 100 | complete run-to-failure trajectories |
| `test_FD001.txt` | 13,096 | 100 | trajectories cut off at a random point before failure |
| `RUL_FD001.txt` | 100 | 100 | true RUL at the last cycle of each test engine |
| `train_FD002.txt` | 53,759 | 260 | as above, six operating conditions |
| `test_FD002.txt` | 33,991 | 259 | as above |
| `RUL_FD002.txt` | 259 | 259 | as above |

| Subset | Operating conditions | Fault modes |
|---|---|---|
| FD001 | 1 (sea level) | 1 (HPC degradation) |
| FD002 | 6 | 1 (HPC degradation) |

## Format

Plain text, space-separated, **no header row**, 26 columns per row (one row = one engine in one operating cycle):

| Column(s) | Name in code | Meaning |
|---|---|---|
| 1 | `unit` | engine id |
| 2 | `cycle` | operating cycle (time index, starts at 1) |
| 3-5 | `setting_1` .. `setting_3` | operational settings (altitude, Mach number, throttle resolver angle) |
| 6-26 | `s_1` .. `s_21` | sensor measurements (full dictionary in `src/config.py` and notebook 01) |

Some lines end with trailing spaces, so the files are read with `sep=r"\s+"`.

## Cleaning and validation

The raw data needs no imputation or row removal. Notebook 01 and `tests/test_data.py` verify that:

- there are no missing, infinite or duplicated values
- every engine starts at cycle 1 and its cycles are consecutive (no gaps)
- the number of RUL labels equals the number of test engines

Seven columns carry no information in FD001 and are dropped before modelling: `s_1`, `s_5`, `s_10`, `s_16`, `s_18`, `s_19` and `setting_3` are constant, and `s_6` only takes two values. The remaining **14 sensors** are listed in `USEFUL_SENSORS` in `src/config.py`.

## Integrity check (SHA-256)

```
963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8  train_FD001.txt
3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851  test_FD001.txt
a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca  RUL_FD001.txt
dac6c4dbc4e7c1bdeb5747da3d313d05c395bb99801b44a002b26a2ba13d788f  train_FD002.txt
de7b5bf7e998a985c378488480528b7c02cff1406a46740def362dda8d9b4e02  test_FD002.txt
c851dd96a6ea6998d3c4a8f834d3c8013aa90e93a6ed950dc826ad0655b2906b  RUL_FD002.txt
```

Check with `shasum -a 256 data/raw/*.txt` (macOS/Linux) or `certutil -hashfile <file> SHA256` (Windows).
