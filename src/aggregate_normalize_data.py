import pandas as pd
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

# Aggregation of biosignals into 30s windows:

REPO_ROOT = Path(__file__).resolve().parent.parent
BIOSIGNAL_DIR = REPO_ROOT / "data" / "preprocessed" / "biosignals"
FEATURE_DIR = REPO_ROOT / "data" / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = '30s'

# Helper function: Calculates the linear slope of a 30-second window.
def temp_slope(series):
    series = series.dropna()
    if len(series) < 2:
        return np.nan
    x = np.arange(len(series))
    # np.polyfit returns [slope, intercept], we just want slope.
    return np.polyfit(x, series.values, 1)[0]

# Helper function: Counts the number of peaks in the EDA signal for a 30-second window.
def eda_peaks(series):
    series = series.dropna()
    peaks, _ = find_peaks(series.values)
    return len(peaks)

# Reads a person's preprocessed biosignal file, normalizes the raw signal and extracts features.
def extract_features(file_path: Path) -> pd.DataFrame:
    
    # Load data and set the datetime index
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    
    person_id = file_path.stem

    # Baseline Normalization
    signals = ['BVP', 'HR', 'EDA', 'TEMP']
    
    # Isolate Phase 1 (Baseline) across all rounds for this specific person
    baseline_data = df[df['phase'] == 'phase1'][signals]
    
    if not baseline_data.empty:
        # Calculate baseline mean and std directly from the raw 16Hz signal
        b_mean = baseline_data.mean()
        b_std = baseline_data.std().replace(0, 1e-8) # Defense against sensor flatlines
        
        # Apply Z-score normalization to the entire signal (Phase 1, 2, and 3)
        df[signals] = (df[signals] - b_mean) / b_std
    else:
        print(f"  Warning: No phase1 data for {person_id}. Extracting features from raw signal.")
    
    # Group by round and phase to prevent "bleeding" across experimental boundaries
    # Then resample the time index into 30-second tumbling windows
    windowed = df.groupby(['round', 'phase']).resample(WINDOW_SIZE)
    
    # 3. Calculate statistics for each window (These are now features of the NORMALIZED signal)
    features = windowed.agg({
        'BVP': ['mean', 'std', 'max', 'min'],
        'HR': ['mean', 'std', 'max', 'min'], 
        'EDA': ['mean', 'std', 'max', 'min', eda_peaks],
        'TEMP': [temp_slope]
    })
    
    # 4. Clean up the multi-level columns created by .agg()
    features.columns = [f"{col[0].lower()}_{col[1]}" for col in features.columns]
    
    # 5. Clean up the index
    features = features.dropna()
    features = features.reset_index()
    
    # Add our subject identifier back in
    features.insert(0, 'subject_id', person_id)
    
    return features

def build_feature_dataset():
    print(f"Scanning for preprocessed biosignals in {BIOSIGNAL_DIR}...")
    
    all_features = []
    processed_count = 0
    
    for file_path in BIOSIGNAL_DIR.glob("*.csv"):
        person_features = extract_features(file_path)
        all_features.append(person_features)
        processed_count += 1
        # print(f"Processed features for: {file_path.stem} ({len(person_features)} windows)")
        
    if not all_features:
        print("No CSV files found. Check your BIOSIGNAL_DIR path.")
        return
        
    # Combine all subjects into one master dataset
    final_dataset = pd.concat(all_features, ignore_index=True)
    
    # Save
    output_path = FEATURE_DIR / "biosignal_features_30s.csv"
    final_dataset.to_csv(output_path, index=False)
    
    print("\n--- Feature Extraction Complete ---")
    print(f"Total subjects processed: {processed_count}")
    print(f"Total 30-second windows generated: {len(final_dataset)}")
    print(f"Dataset saved to: {output_path}\n")

# Normalization of the raw responses data

RESPONSES_DIR = REPO_ROOT / "data" / "preprocessed" / "responses"

def normalize_responses_by_baseline(output_csv: str = "responses_features.csv",
                                    baseline_phase: str = "phase1"):
    
    print(f"Scanning for response files in {RESPONSES_DIR}...")
    
    # 1. Load and combine all individual response files
    all_responses = []
    for file_path in RESPONSES_DIR.glob("*.csv"):
        df = pd.read_csv(file_path)
        
        # The filename looks like "Person1_D1_1_1234_responses.csv"
        # We strip "_responses" to make it match the biosignal 'subject_id' exactly
        global_subject_id = file_path.stem.replace("_responses", "")
        df.insert(0, 'subject_id', global_subject_id)
        
        all_responses.append(df)
        
    if not all_responses:
        print("No CSV files found in the responses directory.")
        return
        
    combined_df = pd.concat(all_responses, ignore_index=True)
    
    # Define metadata and features
    # Add our new 'subject_id' to the metadata so it doesn't get normalized
    metadata_cols = ['subject_id', 'round', 'phase', 'participant_ID', 'puzzler', 'team_ID', 'E4_nr', 'difficulty']
    
    # Select only numeric columns for normalization, dropping metadata
    numeric_df = combined_df.drop(columns=metadata_cols, errors='ignore').select_dtypes(include=[np.number])
    feature_cols = numeric_df.columns.tolist()
    
    df_normalized = combined_df.copy()
    
    # Cast target columns to float to hold the mean-centered decimal values
    df_normalized[feature_cols] = df_normalized[feature_cols].astype(float)
    
    print(f"Normalizing {len(feature_cols)} emotion features using '{baseline_phase}' as the baseline...")
    
    # Group by our new globally unique 'subject_id' instead of 'participant_ID'
    for subject, group in combined_df.groupby('subject_id'):
        
        # Isolate this specific subject's baseline data
        baseline_data = group[group['phase'] == baseline_phase][feature_cols]
        
        if baseline_data.empty:
            print(f"  Warning: No baseline data for participant {subject}. Skipping.")
            continue
            
        # Calculate baseline mean
        b_mean = baseline_data.mean()

        # Apply the transformation to ALL phases for this subject
        normalized_features = group[feature_cols] - b_mean
        
        # Insert the normalized values back into our main dataframe
        df_normalized.loc[group.index, feature_cols] = normalized_features
        
    # Save the final dataset
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURE_DIR / output_csv
    df_normalized.to_csv(out_path, index=False)
    
    print(f"Success! Normalized responses dataset saved to {out_path}")

if __name__ == "__main__":
    # Load raw biosignals, then normalize signal, then extract features
    build_feature_dataset() 
    
    # Load questionnaire responses then normalize features via mean subtraction
    normalize_responses_by_baseline()
