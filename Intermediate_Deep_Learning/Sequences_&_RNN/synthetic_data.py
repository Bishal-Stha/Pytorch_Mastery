import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Config ---
FILENAME = "synthetic_time_series.csv"
SEQ_LENGTH = 10

# --- Step 1: Check if file exists ---
if not os.path.exists(FILENAME):
    print(f"File not found: {FILENAME}. Creating synthetic data...")
    # Create synthetic time series data (e.g., sine wave + noise)
    np.random.seed(42)
    time = np.arange(0, 500)  # 500 time steps
    values = np.sin(0.05 * time) + np.random.normal(scale=0.5, size=len(time))
    
    # Save to CSV
    df = pd.DataFrame({'time': time, 'value': values})
    df.to_csv(FILENAME, index=False)
    print(f"Synthetic data saved to: {FILENAME}")
else:
    print(f"Loading existing data from {FILENAME}")

# --- Step 2: Load data ---
df = pd.read_csv(FILENAME)
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# --- Step 3: Train-test split ---
N = len(df)
SPLIT = round(0.7 * N)
train_data, test_data = df[:SPLIT], df[SPLIT:]

# --- Step 4: Sequence creation function ---
def create_sequences(df, seq_length):
    xs, ys = [], []
    for i in range(len(df) - seq_length):
        x = df.iloc[i:(i + seq_length), 1]  # column 'value'
        y = df.iloc[i + seq_length, 1]
        xs.append(x.values)  # convert Series to np.array
        ys.append(y)
    return np.array(xs), np.array(ys)

# --- Step 5: Create sequences ---
X_train, y_train = create_sequences(train_data, SEQ_LENGTH)
X_test, y_test = create_sequences(test_data, SEQ_LENGTH)

# --- Step 6: Output shapes ---
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
