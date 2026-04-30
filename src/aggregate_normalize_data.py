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

# Reads a person's preprocessed biosignal file and extracts 30s window features.
def extract_features(file_path: Path) -> pd.DataFrame:
    
    # Load data and set the datetime index
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    
    # Extract the person ID from the filename (e.g., "Person1_D1_1_1234")
    person_id = file_path.stem
    
    # Group by round and phase to prevent "bleeding" across experimental boundaries
    # Then resample the time index into 30-second tumbling windows
    windowed = df.groupby(['round', 'phase']).resample(WINDOW_SIZE)
    
    # Calculate statistics for each window
    # We start with just EDA mean and standard deviation
    features = windowed.agg({
        'BVP': ['mean', 'std', 'max', 'min'],
        'HR': ['mean', 'std', 'max', 'min'], # Could add a helper function to get the frequency domain HR signal here
        'EDA': ['mean', 'std', 'max', 'min', eda_peaks],
        'TEMP': [temp_slope]
    })
    
    # Clean up the multi-level columns created by .agg()
    # This turns ('EDA', 'mean') into 'eda_mean'
    features.columns = [f"{col[0].lower()}_{col[1]}" for col in features.columns]
    
    # Clean up the index
    # Resampling creates windows where there might be no data
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

# Normalization of extracted features from biosignal data

FEATURE_DIR = REPO_ROOT / "data" / "features"

def normalize_by_baseline(input_csv: str = "biosignal_features_30s.csv", 
                          output_csv: str = "biosignal_features_30s_baseline_norm.csv",
                          baseline_phase: str = "phase1"):
    
    file_path = FEATURE_DIR / input_csv
    print(f"Loading unnormalized features from {file_path}...")
    df = pd.read_csv(file_path)
    
    metadata_cols = ['subject_id', 'time', 'round', 'phase']
    # Drop metadata and ensure we only grab numeric columns
    numeric_df = df.drop(columns=metadata_cols, errors='ignore').select_dtypes(include=[np.number])
    feature_cols = numeric_df.columns.tolist()
    
    df_normalized = df.copy()

    # Cast target columns to float so Pandas allows us to insert decimals
    df_normalized[feature_cols] = df_normalized[feature_cols].astype(float)
    
    print(f"Normalizing {len(feature_cols)} features using '{baseline_phase}' as the baseline...")
    
    # Process each subject individually
    for subject, group in df.groupby('subject_id'):
        
        # Isolate this specific subject's baseline data
        baseline_data = group[group['phase'] == baseline_phase][feature_cols]
        
        # If a subject somehow doesn't have phase1 data, skip or warn
        if baseline_data.empty:
            print(f"  Warning: No baseline ({baseline_phase}) data for {subject}. Skipping normalization.")
            continue
            
        # Calculate baseline mean and standard deviation
        b_mean = baseline_data.mean()
        b_std = baseline_data.std()
        
        # Edge Case Defense: If a sensor flatlines, std becomes 0. 
        # Division by zero creates NaNs. We replace 0s with a tiny number.
        b_std = b_std.replace(0, 1e-8)
        
        # Apply the transformation to ALL phases for this subject
        normalized_features = (group[feature_cols] - b_mean) / b_std
        
        # Insert the normalized values back into our main dataframe
        df_normalized.loc[group.index, feature_cols] = normalized_features

    # Save the final dataset
    out_path = FEATURE_DIR / output_csv
    df_normalized.to_csv(out_path, index=False)
    print(f"Success! Normalized dataset saved to {out_path}\n")

# Normalization of the raw responses data

RESPONSES_DIR = REPO_ROOT / "data" / "preprocessed" / "responses"
FEATURE_DIR = REPO_ROOT / "data" / "features"

def normalize_responses_by_baseline(output_csv: str = "responses_features_baseline_norm.csv",
                                    baseline_phase: str = "phase1"):
    
    print(f"Scanning for response files in {RESPONSES_DIR}...")
    
    # Load and combine all individual response files
    all_responses = []
    for file_path in RESPONSES_DIR.glob("*.csv"):
        df = pd.read_csv(file_path)
        all_responses.append(df)
        
    if not all_responses:
        print("No CSV files found in the responses directory.")
        return
        
    combined_df = pd.concat(all_responses, ignore_index=True)
    
    # Define metadata and features
    # 'difficulty' is explicitly excluded because it has no phase1 baseline
    metadata_cols = ['round', 'phase', 'participant_ID', 'puzzler', 'team_ID', 'E4_nr', 'difficulty']
    
    # Select only numeric columns for normalization, dropping metadata
    numeric_df = combined_df.drop(columns=metadata_cols, errors='ignore').select_dtypes(include=[np.number])
    feature_cols = numeric_df.columns.tolist()
    
    df_normalized = combined_df.copy()
    
    print(f"Normalizing {len(feature_cols)} emotion features using '{baseline_phase}' as the baseline...")
    
    # Process each participant individually
    # Grouping by participant_ID to handle their specific psychological baseline
    for subject, group in combined_df.groupby('participant_ID'):
        
        # Isolate this specific subject's baseline data
        baseline_data = group[group['phase'] == baseline_phase][feature_cols]
        
        if baseline_data.empty:
            print(f"  Warning: No baseline data for participant {subject}. Skipping.")
            continue
            
        # Calculate baseline mean
        b_mean = baseline_data.mean()

        # NOTE: We ignore the Standard Deviation to avoid the divide by zero error and since they should
        # all already be on the same scale
        
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
    build_feature_dataset()
    normalize_by_baseline()
    normalize_responses_by_baseline()
