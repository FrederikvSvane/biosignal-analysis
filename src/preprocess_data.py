"""
Preprocess raw biosignals into one time-series CSV per person.

Reads through all the files in the raw data like so:  

    data/raw/dataset/<cohort>/<ID>/<round>/<phase>/{BVP,EDA,HR,TEMP}.csv

Writes to data/preprocessed like so: 

    data/preprocessed/Person{N}_{cohort}_{ID}.csv

With these columns (round and phase are included to make downstream analysis easier, even though its slightly inefficient):

    columns: [time, BVP, EDA, HR, TEMP, round, phase]

The signals are sampled at three different frequencies. We account for this by downsampling and interpolating to a common frequency of 16hz (62.5 ms):

    BVP (64 Hz) -> downsampled with mean
    EDA, TEMP (4 Hz), HR (1 Hz) -> linearly interpolated

This does result in some data loss (from BVP), but we suspect it wont matter. Time will tell.
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DATASET_DIR = REPO_ROOT / "data" / "raw" / "dataset"
PREPROCESSED_DIR = REPO_ROOT / "data" / "preprocessed"

RESAMPLE_PERIOD = "62.5ms"
SIGNALS = ("BVP", "EDA", "HR", "TEMP")


def prepare_signal(phase_dir: Path, signal: str) -> pd.DataFrame:
    df = pd.read_csv(phase_dir / f"{signal}.csv")
    df["time"] = pd.to_datetime(df["time"], format="mixed")
    df = df.set_index("time").sort_index()[[signal]]

    if signal == "BVP":
        return df.resample(RESAMPLE_PERIOD).mean()

    upsampled = df.resample(RESAMPLE_PERIOD).asfreq()
    upsampled[signal] = upsampled[signal].interpolate(method="linear")
    return upsampled


def build_phase_dataframe(phase_dir: Path) -> pd.DataFrame:
    parts = [prepare_signal(phase_dir, sig) for sig in SIGNALS]
    combined = pd.concat(parts, axis=1, join="inner").reset_index()
    combined["round"] = phase_dir.parent.name
    combined["phase"] = phase_dir.name
    return combined


def preprocess_all(raw_dir: Path = RAW_DATASET_DIR, out_dir: Path = PREPROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    person_number = 0
    for cohort_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for person_dir in sorted(p for p in cohort_dir.iterdir() if p.is_dir()):
            person_number += 1
            out_path = out_dir / f"Person{person_number}_{cohort_dir.name}_{person_dir.name}.csv"
            if out_path.exists():
                out_path.unlink()

            header_written = False
            for round_dir in sorted(p for p in person_dir.iterdir() if p.is_dir()):
                for phase_dir in sorted(p for p in round_dir.iterdir() if p.is_dir()):
                    phase_df = build_phase_dataframe(phase_dir)
                    phase_df.to_csv(out_path, mode="a", index=False, header=not header_written)
                    header_written = True

            print(f"Finished processing {out_path.name}")

    print(f"Done! Wrote {person_number} files to {out_dir}")


if __name__ == "__main__":
    preprocess_all()
