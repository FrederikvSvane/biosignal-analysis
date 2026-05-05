"""
Run this to preprocess raw biosignals and self-reported responses into per-person CSVs.

Reads through all the files in the raw data like so:

    data/raw/dataset/<cohort>/<ID>/<round>/<phase>/{BVP,EDA,HR,TEMP,response}.csv

Writes two parallel files per person under data/preprocessed:

    data/preprocessed/biosignals/Person{N}_{cohort}_{ID}.csv          (time-series)
    data/preprocessed/responses/Person{N}_{cohort}_{ID}_responses.csv (one row per phase)

With these columns (round and phase are included in both files to make downstream analysis easier, even though its slightly inefficient):

    biosignals: [time, BVP, EDA, HR, TEMP, round, phase]
    responses:  [round, phase, participant_ID, puzzler, team_ID, E4_nr,
                 upset, hostile, alert, ashamed, inspired, nervous,
                 determined, attentive, afraid, active, frustrated, difficulty]

The signals are sampled at three different frequencies. We account for this by downsampling and interpolating (respectively) to a common frequency of 16hz (62.5 ms):

    BVP (64 Hz) -> downsampled with mean
    EDA, TEMP (4 Hz), HR (1 Hz) -> linearly interpolated

This does result in some data loss (from BVP), but we suspect it wont matter. Time will tell.
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DATASET_DIR = REPO_ROOT / "data" / "raw" / "dataset"
PREPROCESSED_DIR = REPO_ROOT / "data" / "preprocessed"
BIOSIGNAL_DIR = PREPROCESSED_DIR / "biosignals"
RESPONSES_DIR = PREPROCESSED_DIR / "responses"

RESAMPLE_PERIOD = "62.5ms"
SIGNALS = ("BVP", "EDA", "HR", "TEMP")


def prepare_signal(phase_dir: Path, signal: str) -> pd.DataFrame:
    df = pd.read_csv(phase_dir / f"{signal}.csv")
    if df.isnull().any().any()==True:
        print(f'{df.isnull().sum()} missing values in {phase_dir} {signal} ')
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


# Fixing column-name inconsistencies for responses:
#   i) "particpant_ID" (typo in D1_1..D1_5) -> "participant_ID"
#   ii) "parent" (D1_6) -> "puzzler" (same thing, don't know why they call it parent all of a sudden)
RESPONSE_COLUMN_RENAMES = {
    "particpant_ID": "participant_ID",
    "parent": "puzzler",
}


def build_response_dataframe(phase_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(phase_dir / "response.csv")
    if df.isnull().any().any()==True:
        missing_cols = df.columns[df.isnull().any()]
        print(f'{df.isnull().sum().sum()} missing values in {phase_dir} responses cols: {(", ").join(missing_cols)}')
    df = df.drop(columns=["index", "Unnamed: 0"], errors="ignore")
    df = df.rename(columns=RESPONSE_COLUMN_RENAMES)
    df.insert(0, "round", phase_dir.parent.name)
    df.insert(1, "phase", phase_dir.name)
    return df


def _append(df: pd.DataFrame, path: Path, header_written: bool) -> None:
    df.to_csv(path, mode="a", index=False, header=not header_written)


def preprocess_all(
    raw_dir: Path = RAW_DATASET_DIR,
    biosignal_dir: Path = BIOSIGNAL_DIR,
    responses_dir: Path = RESPONSES_DIR,
) -> None:
    # Ensures both output dirs exist, and does nothing if they do (exist_ok=True).
    # Also deletes any left over CSVs from a previous run, so re-running the script doesn't produce duplicates.
    for dir in (biosignal_dir, responses_dir):
        dir.mkdir(parents=True, exist_ok=True)
        for existing in dir.glob("*.csv"):
            existing.unlink()

    person_number = 0
    for cohort_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for person_dir in sorted(p for p in cohort_dir.iterdir() if p.is_dir()):
            person_number += 1
            person_key = f"Person{person_number}_{cohort_dir.name}_{person_dir.name}"
            biosignal_path = biosignal_dir / f"{person_key}.csv"
            responses_path = responses_dir / f"{person_key}_responses.csv"

            # We treat responses differently than biosignals, because schemas vary across
            # cohorts and phases (e.g. phase2 has an extra `difficulty` column,
            # D1_6 renames `particpant_ID`->`participant_ID` and `puzzler`->`parent`). wtf.
            # so we let pandas align columns by name and fill missing ones with NaN.
            biosignal_header = False
            response_frames = []
            for round_dir in sorted(p for p in person_dir.iterdir() if p.is_dir()):
                for phase_dir in sorted(p for p in round_dir.iterdir() if p.is_dir()):
                    _append(build_phase_dataframe(phase_dir), biosignal_path, biosignal_header)
                    biosignal_header = True
                    response_frames.append(build_response_dataframe(phase_dir))

            pd.concat(response_frames, ignore_index=True).to_csv(responses_path, index=False)

            print(f"Finished processing {person_key}")

    print(f"Done! Wrote {person_number} biosignal + {person_number} response files")


if __name__ == "__main__":
    preprocess_all()
