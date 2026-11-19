import os
import numpy as np
import pandas as pd
import joblib

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# Create model directory
os.makedirs("models", exist_ok=True)

# Load normal sensor data
print("Loading normal sensor data...")

data = pd.read_csv("data/normal_sensor_data.csv")

features = ["temperature", "vibration", "pressure"]
X = data[features].values

# Standardize sensor readings
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Data loaded successfully!")
print("Training samples:", X_scaled.shape[0])

# Compare different numbers of hidden states
states_to_test = [2, 3, 4]

best_model = None
best_bic = float("inf")
best_states = None

print("\nComparing HMM models...")

for states in states_to_test:

    print(f"\nTraining HMM with {states} hidden states...")

    model = GaussianHMM(
        n_components=states,
        covariance_type="diag",
        n_iter=200,
        random_state=42,
        min_covar=0.01
    )

    model.fit(X_scaled)

    bic_score = model.bic(X_scaled)

    print(f"BIC Score: {bic_score:.2f}")

    if bic_score < best_bic:
        best_bic = bic_score
        best_model = model
        best_states = states

# Select best model
print("\nBest model selected!")
print("Number of hidden states:", best_states)
print("Best BIC:", round(best_bic, 2))

# Calculate training window likelihoods
window_size = 100

training_scores = []

for i in range(len(X_scaled) - window_size + 1):

    window = X_scaled[i:i + window_size]

    score = best_model.score(window) / window_size

    training_scores.append(score)

# Set anomaly threshold using training distribution
threshold = np.percentile(training_scores, 1)

print("\nAnomaly threshold:", round(threshold, 4))

# Save model and preprocessing details
model_data = {
    "model": best_model,
    "scaler": scaler,
    "features": features,
    "threshold": threshold,
    "window_size": window_size,
    "hidden_states": best_states,
    "bic": best_bic
}

joblib.dump(model_data, "models/hmm_model.pkl")

print("\nModel saved successfully!")
print("Location: models/hmm_model.pkl")
print("Training completed successfully!")
