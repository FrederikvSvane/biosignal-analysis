"""
Pipeline

Executes the zero-shot anomaly detection pipeline (GMM & OC-SVM) using nested LOSO cross-validation. 
It dynamically tunes hyperparameters on resting baselines and exports anomaly scores for downstream evaluation.
"""

from collections.abc import Iterator
from enum import Enum
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class PCAMode(Enum):
    NONE = None
    VAR95 = 0.95
    TWO_D = 2

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = REPO_ROOT / "data" / "features"
FEATURES_CSV = FEATURE_DIR / "biosignal_features_30s.csv"

META_COLS = ["subject_id", "round", "phase", "time"]

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

# Yields Train (Phase 1 only) and Test splits for all three phases independently
def loso_splits(df: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    for subject in df["subject_id"].unique():
        # Train ONLY on Phase 1 (Pure baseline, no Phase 3 residual stress pollution)
        train_df = df[(df["subject_id"] != subject) & (df["phase"] == "phase1")]
        
        # Test on all three phases to prove the model can differentiate them
        test_p1 = df[(df["subject_id"] == subject) & (df["phase"] == "phase1")]
        test_p2 = df[(df["subject_id"] == subject) & (df["phase"] == "phase2")]
        test_p3 = df[(df["subject_id"] == subject) & (df["phase"] == "phase3")]
        
        yield subject, train_df, test_p1, test_p2, test_p3

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

    # Define both models
    GMM_PARAMS = dict(n_components=3, covariance_type="full", random_state=0)
    SVM_PARAMS = dict(kernel='rbf', nu=0.10, gamma='scale')
    
    # We will store the results to calculate correlations later
    gmm_results = []
    svm_results = []

    # DEFINE HYPERPARAMETER GRIDS
    svm_param_grid = {
        'nu': [0.05, 0.10, 0.15, 0.20, 0.25], # "Allowable outlier rate in phase 1"
        'gamma': ['scale', 'auto', 0.1, 0.01] # Wiggliness of the decision boundary
    }
    
    gmm_param_grid = {
        'n_components': [1, 2, 3, 5] # How many distributions to fit
    }

    gmm_results = []
    svm_results = []

    print("Starting Nested Zero-Shot Evaluation\n")

    for subject, train_df, test_p1, test_p2, test_p3 in loso_splits(df):
        if len(test_p2) == 0: continue
        
        X_train = train_df[feat_cols].values
        X_p1 = test_p1[feat_cols].values
        X_p2 = test_p2[feat_cols].values
        X_p3 = test_p3[feat_cols].values
        
        inner_subjects = train_df['subject_id'].unique()

        # 1. NESTED TUNING: OC-SVM (Calibration)
        best_svm_params = {'nu': 0.10, 'gamma': 'scale'} # Fallback
        best_cal_error = float('inf')
        
        for nu in svm_param_grid['nu']:
            for gamma in svm_param_grid['gamma']:
                inner_rates = []
                for val_sub in inner_subjects:
                    inner_train = train_df[train_df['subject_id'] != val_sub][feat_cols].values
                    inner_val = train_df[train_df['subject_id'] == val_sub][feat_cols].values
                    if len(inner_train) < 5 or len(inner_val) < 1: continue
                    
                    inner_pipe = make_pipeline(OneClassSVM(kernel='rbf', nu=nu, gamma=gamma), PCAMode.VAR95)
                    inner_pipe.fit(inner_train)
                    
                    # predict() returns -1 for anomaly
                    preds = inner_pipe.named_steps['model'].predict(inner_pipe[:-1].transform(inner_val))
                    inner_rates.append(np.mean(preds == -1))
                
                if inner_rates:
                    cal_error = abs(np.mean(inner_rates) - nu)
                    if cal_error < best_cal_error:
                        best_cal_error = cal_error
                        best_svm_params = {'nu': nu, 'gamma': gamma}

        # 2. NESTED TUNING: GMM (Minimize BIC)
        best_gmm_params = {'n_components': 3} # Fallback
        best_bic = float('inf')
        
        for n_comp in gmm_param_grid['n_components']:
            inner_bics = []
            for val_sub in inner_subjects:
                inner_train = train_df[train_df['subject_id'] != val_sub][feat_cols].values
                inner_val = train_df[train_df['subject_id'] == val_sub][feat_cols].values
                if len(inner_train) < 5 or len(inner_val) < 1: continue
                
                inner_pipe = make_pipeline(GaussianMixture(n_components=n_comp, covariance_type="full", random_state=0), PCAMode.VAR95)
                inner_pipe.fit(inner_train)
                
                # Transform validation data through scaler/PCA to calculate BIC
                X_val_transformed = inner_pipe[:-1].transform(inner_val)
                val_bic = inner_pipe.named_steps['model'].bic(X_val_transformed)
                inner_bics.append(val_bic)
                
            if inner_bics:
                avg_bic = np.mean(inner_bics)
                if avg_bic < best_bic:
                    best_bic = avg_bic
                    best_gmm_params = {'n_components': n_comp}

        # We tie the expected noise rate together for fairness
        target_noise_pct = best_svm_params['nu'] * 100
        percentile_threshold = 100 - target_noise_pct 

        print(f"[{subject}] Tuned -> SVM: (nu={best_svm_params['nu']}, gamma={best_svm_params['gamma']}) | GMM: (n_comp={best_gmm_params['n_components']})")

        # FINAL EVALUATION: GMM
        gmm_pipe = make_pipeline(GaussianMixture(n_components=best_gmm_params['n_components'], covariance_type="full", random_state=0), PCAMode.VAR95)
        gmm_pipe.fit(X_train)
        
        gmm_scores_train = score_anomalies(gmm_pipe, X_train)
        gmm_scores_p1 = score_anomalies(gmm_pipe, X_p1)
        gmm_scores_p2 = score_anomalies(gmm_pipe, X_p2)
        gmm_scores_p3 = score_anomalies(gmm_pipe, X_p3)
        
        gmm_threshold = np.percentile(gmm_scores_train, percentile_threshold)
        
        gmm_rate_p1 = np.mean(gmm_scores_p1 > gmm_threshold) * 100 if len(X_p1) > 0 else 0
        gmm_rate_p2 = np.mean(gmm_scores_p2 > gmm_threshold) * 100 if len(X_p2) > 0 else 0
        gmm_rate_p3 = np.mean(gmm_scores_p3 > gmm_threshold) * 100 if len(X_p3) > 0 else 0
        
        gmm_results.append({
            'subject_id': subject, 
            'GMM_P1': gmm_scores_p1.mean() if len(X_p1) > 0 else 0, 
            'GMM_P2_Stress': gmm_scores_p2.mean(), 
            'GMM_P3': gmm_scores_p3.mean() if len(X_p3) > 0 else 0,
            'GMM_Rate_P1': gmm_rate_p1,
            'GMM_Rate_P2': gmm_rate_p2,
            'GMM_Rate_P3': gmm_rate_p3
        })

        # FINAL EVALUATION: OC-SVM
        svm_pipe = make_pipeline(OneClassSVM(kernel='rbf', **best_svm_params), PCAMode.VAR95)
        svm_pipe.fit(X_train)
        
        svm_scores_train = score_anomalies(svm_pipe, X_train)
        svm_scores_p1 = score_anomalies(svm_pipe, X_p1)
        svm_scores_p2 = score_anomalies(svm_pipe, X_p2)
        svm_scores_p3 = score_anomalies(svm_pipe, X_p3)
        
        svm_threshold = np.percentile(svm_scores_train, percentile_threshold)
        
        svm_rate_p1 = np.mean(svm_scores_p1 > svm_threshold) * 100 if len(X_p1) > 0 else 0
        svm_rate_p2 = np.mean(svm_scores_p2 > svm_threshold) * 100 if len(X_p2) > 0 else 0
        svm_rate_p3 = np.mean(svm_scores_p3 > svm_threshold) * 100 if len(X_p3) > 0 else 0
        
        svm_results.append({
            'subject_id': subject, 
            'SVM_P1': svm_scores_p1.mean() if len(X_p1) > 0 else 0, 
            'SVM_P2_Stress': svm_scores_p2.mean(), 
            'SVM_P3': svm_scores_p3.mean() if len(X_p3) > 0 else 0,
            'SVM_Rate_P1': svm_rate_p1,
            'SVM_Rate_P2': svm_rate_p2,
            'SVM_Rate_P3': svm_rate_p3
        })

        print(f"  -> Flag Rate P1 -> P2 | GMM: {gmm_rate_p1:04.1f}% -> {gmm_rate_p2:04.1f}% | SVM: {svm_rate_p1:04.1f}% -> {svm_rate_p2:04.1f}%\n")

    print("\nAggregate results and evaluation")
    
    gmm_df = pd.DataFrame(gmm_results)
    svm_df = pd.DataFrame(svm_results)
    
    # Merge GMM and SVM into one validation dataframe
    final_eval_df = gmm_df.merge(svm_df, on='subject_id')

    # Define output path
    RESULTS_CSV = FEATURE_DIR / "zero_shot_results.csv"
    
    # Save the merged dataset
    final_eval_df.to_csv(RESULTS_CSV, index=False)