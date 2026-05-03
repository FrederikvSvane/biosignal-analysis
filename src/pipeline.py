"""
The full modelling evaluation pipeline. Configure and run to get a evalution metrics of how the chosen model performs.

Pipeline:
    StandardScaler -> [PCA(n_components=0.95), optional] -> {OCSVM | GMM} using LOSO -> mean AUROC score

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
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


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
    test_phases = RESTING_PHASES + (PUZZLE_PHASE,)
    for subject in df["subject_id"].unique():
        train_df = df[(df["subject_id"] != subject) & df["phase"].isin(RESTING_PHASES)]
        test_df = df[(df["subject_id"] == subject) & df["phase"].isin(test_phases)]
        yield subject, train_df, test_df


def score_anomalies(pipe: Pipeline, X_test: np.ndarray) -> np.ndarray:
    # GMM exposes score_samples, OCSVM exposes decision_function, hence the switch.
    # Both return "normality", so we negate.
    # higher score = more anomalous.
    if hasattr(pipe, "score_samples"):
        return -pipe.score_samples(X_test)
    if hasattr(pipe, "decision_function"):
        return -pipe.decision_function(X_test)
    else:
        raise RuntimeError("Not able to return score anomalies from this model")


def plot_loso_2d(folds: list, model_name: str) -> None:
    # Each entry in `folds`: (subject, Z_train, Z_test, y_test, fitted_pipe, auroc).
    # Z_* are already in 2D PCA space, so we can draw a real decision-surface contour
    # by sweeping the model step alone over a meshgrid.
    import matplotlib.pyplot as plt

    n = len(folds)
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6), squeeze=False)

    for ax, (subject, Z_train, Z_test, y_test, pipe, auroc) in zip(axes.ravel(), folds):
        all_pts = np.vstack([Z_train, Z_test])
        pad = 0.5
        x_min, x_max = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
        y_min, y_max = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 80), np.linspace(y_min, y_max, 80))
        grid = np.c_[xx.ravel(), yy.ravel()]

        model = pipe.named_steps["model"]
        if hasattr(model, "score_samples"):
            zz = -model.score_samples(grid)
        else:
            zz = -model.decision_function(grid)
        zz = zz.reshape(xx.shape)

        ax.contourf(xx, yy, zz, levels=15, cmap="RdBu_r", alpha=0.6)
        ax.scatter(Z_train[:, 0], Z_train[:, 1], s=2, c="grey", alpha=0.3)
        ax.scatter(Z_test[y_test == 0, 0], Z_test[y_test == 0, 1], s=8, c="blue", label="rest")
        ax.scatter(Z_test[y_test == 1, 0], Z_test[y_test == 1, 1], s=8, c="red", label="puzzle")
        ax.set_title(f"{subject.split('_')[0]} — {auroc:.2f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    fig.suptitle(f"{model_name.upper()} anomaly score in 2D PCA space (red = anomalous)", fontsize=11)
    fig.tight_layout()
    plt.show()


def build_model(name: str) -> BaseEstimator:
    # Hardcoded params per model. Validate / re-tune in a separate exploration script.
    # GMM: n_components=3, full covariance came from a small BIC sweep.
    # OCSVM: nu=0.05, gamma='scale' was the consistent winner across subjects in Max's nested-LOSO calibration.
    if name == "gmm":
        return GaussianMixture(n_components=3, covariance_type="full", random_state=0)
    if name == "ocsvm":
        return OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    raise ValueError(f"Unknown model: {name}")


if __name__ == "__main__":
    MODEL = "ocsvm"  # "gmm" | "ocsvm"
    PCA_MODE = PCAMode.TWO_D  # set to TWO_D to auto-render the per-subject 2D plot

    df = load_features()
    feat_cols = feature_columns(df)

    per_subject_aurocs = []
    anomaly_scores_all_folds = []
    puzzle_labels_all_folds = []
    plot_folds: list = []

    for subject, train_df, test_df in loso_splits(df):
        X_train = train_df[feat_cols].values
        X_test = test_df[feat_cols].values
        y_test = (test_df["phase"] == PUZZLE_PHASE).astype(int).values

        pipe = make_pipeline(build_model(MODEL), PCA_MODE)
        pipe.fit(X_train)
        scores = score_anomalies(pipe, X_test)

        auroc = roc_auc_score(y_test, scores)
        per_subject_aurocs.append(auroc)
        anomaly_scores_all_folds.append(scores)
        puzzle_labels_all_folds.append(y_test)

        if PCA_MODE is PCAMode.TWO_D:
            # pipe[:-1] = scaler + PCA; gives us the 2D coordinates the model actually saw.
            Z_train = pipe[:-1].transform(X_train)
            Z_test = pipe[:-1].transform(X_test)
            plot_folds.append((subject, Z_train, Z_test, y_test, pipe, auroc))

        print(f"{subject}: n={len(scores)}  auroc={auroc:.3f}")

    mean_auroc = float(np.mean(per_subject_aurocs))
    pooled_auroc = roc_auc_score(
        np.concatenate(puzzle_labels_all_folds),
        np.concatenate(anomaly_scores_all_folds),
    )
    print(f"\n[{MODEL}] mean per-subject AUROC: {mean_auroc:.3f}")
    print(f"[{MODEL}] pooled AUROC:           {pooled_auroc:.3f}")

    if PCA_MODE is PCAMode.TWO_D:
        plot_loso_2d(plot_folds, MODEL)
