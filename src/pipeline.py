"""
The full modelling evaluation pipeline. Configure and run to get a evalution metrics of how the chosen model performs.

Pipeline:
    StandardScaler -> [PCA(n_components=0.95), optional] -> {OCSVM | GMM | IsolationForest} using LOSO -> mean AUROC score

    In english:
    The pipeline assumes the raw data has been preprocessed by the `preprocess_data.py` script.
    And that the preprocessed data has been distilled into aggregated, normalized features by the `aggregate_normalize_data.py` script.
    It then takes the aggregated, normalized data `biosignal_features_30s.csv` as input, and does:
        1) Scale the data numerically, to make variance dependend methods such as PCA agnostic to individual subjects.
        2) Optionally run PCA to reduce the amount of features
        3) Train model on 25 subjects, test on the 26th, report AUROC
        4) Return the mean of all the reported AUROCs

A note on realness:
    In a strict zero-shot setting we would not have access to the test subject's resting baseline! 
    This is consistent with the case description ("Train on Resting, test on Puzzle"). 
    But maybe we should revisit to make it not so. Would be more realistic!

    Edit: On later thought, this could be solved by having the subject perform a "resting phase calibration" protocol.
"""

from collections.abc import Iterator
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
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


def make_pipeline(model: BaseEstimator, pca_mode: PCAMode = PCAMode.VAR95) -> Pipeline:
    steps: list = [("scaler", StandardScaler())]
    if pca_mode is not PCAMode.NONE:
        # Python AND scikit learn are funny. They automatically parse n_components=0.95 as "cover 95% variance"
        # and n_components=2 as "reduce to 2d". Sure it feels natural, but lol
        steps.append(("pca", PCA(n_components=pca_mode.value, random_state=0)))
    steps.append(("model", model))
    return Pipeline(steps)


def loso_splits(df: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    for subject in df["subject_id"].unique():
        train_df = df[(df["subject_id"] != subject) & df["phase"].isin(RESTING_PHASES)]
        test_df = df[(df["subject_id"] == subject) & (df["phase"] == PUZZLE_PHASE)]
        yield subject, train_df, test_df


def score_anomalies(pipe: Pipeline, X_test: np.ndarray) -> np.ndarray:
    # GMM/IsolationForest output score_samples, OCSVM outputs decision_function, hence the switch. 
    # Both return "normality", so we negate.
    # higher score = more anomalous. 
    if hasattr(pipe, "score_samples"):
        return -pipe.score_samples(X_test)
    return -pipe.decision_function(X_test)


if __name__ == "__main__":
    df = load_features()
    feat_cols = feature_columns(df)

    # Find these params by GridSearchCV somewhere else, like in a notebook
    GMM_PARAMS = dict(n_components=3, covariance_type="full", random_state=0)

    for subject, train_df, test_df in loso_splits(df):
        X_train = train_df[feat_cols].values
        X_test = test_df[feat_cols].values

        pipe = make_pipeline(GaussianMixture(**GMM_PARAMS), PCAMode.VAR95)
        pipe.fit(X_train)
        scores = score_anomalies(pipe, X_test)

        print(f"{subject}: test_windows={len(scores)}  mean_anom={scores.mean():.3f}")
