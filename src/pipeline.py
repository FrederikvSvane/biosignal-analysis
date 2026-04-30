"""
The full modelling pipeline. Configure and run to get a "working" anomaly detection model.

Pipeline:
    StandardScaler -> [PCA(n_components=0.95), optional] -> {OCSVM | GMM | IsolationForest}

A note on realness:
    In a strict zero-shot setting we would not have access to the test subject's resting baseline! 
    This is consistent with the case description ("Train on Resting, test on Puzzle"). 
    But maybe we should revisit to make it not so. Would be more realistic!
"""

from enum import Enum
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# PCA mode for the pipeline. NONE skips PCA; VAR95 keeps enough components to
# cover 95% of variance (attempt to reduce noise); TWO_D fixes 2 components (for pretty plotting).
class PCAMode(Enum):
    NONE = None
    VAR95 = 0.95
    TWO_D = 2

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = REPO_ROOT / "data" / "features"
FEATURES_CSV = FEATURE_DIR / "biosignal_features_30s.csv"

META_COLS = ["subject_id", "round", "phase", "time"]
RESTING_PHASES = ("phase1", "phase3")
PUZZLE_PHASE = "phase2"


def load_features() -> pd.DataFrame:
    return pd.read_csv(FEATURES_CSV)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def make_pipeline(pca_mode: PCAMode = PCAMode.VAR95) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if pca_mode is not PCAMode.NONE:
        # Python AND scikit learn are funny. They automatically parse n_components=0.95 as "cover 95% variance"
        # and n_components=2 as "reduce to 2d". Sure it feels natural, but lol 
        steps.append(("pca", PCA(n_components=pca_mode.value, random_state=0)))
    return Pipeline(steps)


if __name__ == "__main__":
    df = load_features()
    feat_cols = feature_columns(df)

    resting = df[df["phase"].isin(RESTING_PHASES)]
    puzzle = df[df["phase"] == PUZZLE_PHASE]

    X_rest = resting[feat_cols].values
    X_puzzle = puzzle[feat_cols].values

    print(f"Features ({len(feat_cols)}): {feat_cols}")
    print(f"Resting windows (phase1+phase3): {len(X_rest)}")
    print(f"Puzzle windows (phase2):         {len(X_puzzle)}")

    pipe = make_pipeline(PCAMode.VAR95)
    Z_rest = pipe.fit_transform(X_rest)
    Z_puzzle = pipe.transform(X_puzzle)

    print(f"\nTransformed resting shape: {Z_rest.shape}")
    print(f"Transformed puzzle shape:  {Z_puzzle.shape}")

    evr = pipe.named_steps["pca"].explained_variance_ratio_
    print(f"\nPCA components retained: {len(evr)}")
    print(f"Explained variance ratio: {evr}")
    print(f"Cumulative explained variance: {evr.cumsum()}")
