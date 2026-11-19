import numpy as np
import pandas as pd
import joblib
import os

# Load trained HMM model
print("Loading trained HMM model...")
saved_data = joblib.load("models/hmm_model.pkl")
model = saved_data["model"]
scaler = saved_data["scaler"]
features = saved_data["features"]

threshold = saved_data["threshold"]
threshold = threshold - 1.0
print("Adjusted threshold:",threshold)
window_size = saved_data["window_size"]

# Load test sensor data
test_data = pd.read_csv("data/test_sensor_data.csv")
X_test = test_data[features].values
# Standardize test data using training scaler
X_test_scaled = scaler.transform(X_test)
print("Test data loaded successfully!")
print("Total test readings:", len(X_test))
# Sliding window anomaly detection
print("\nStarting real-time anomaly detection...")
window_results = []
anomaly_mask = np.zeros(len(X_test), dtype=bool)
likelihood_sum = np.zeros(len(X_test))
likelihood_count = np.zeros(len(X_test))
for i in range(len(X_test) - window_size + 1):
    window = X_test_scaled[i:i + window_size]
    # Calculate average log-likelihood
    log_likelihood = model.score(window) / window_size
    # Detect anomaly
    is_anomaly = log_likelihood < threshold
    window_results.append({
        "window_start": i,
        "window_end": i + window_size - 1,
        "log_likelihood": log_likelihood,
        "anomaly": int(is_anomaly)
    })

    # Mark anomalous regions
    if is_anomaly:
        pass
    likelihood_sum[i:i + window_size] += log_likelihood
    likelihood_count[i:i + window_size] += 1
# Calculate average likelihood for each reading
point_likelihood = np.divide(
    likelihood_sum,
    likelihood_count,
    out=np.zeros_like(likelihood_sum),
    where=likelihood_count > 0 )

# Add results to test dataset
test_data["log_likelihood"] = point_likelihood
test_data["anomaly"] = (
    point_likelihood < threshold).astype(int)

# Save window results
os.makedirs("outputs", exist_ok=True)
window_df = pd.DataFrame(window_results)
window_df.to_csv(
    "outputs/window_detection_results.csv",
    index=False )
test_data.to_csv(
    "outputs/test_anomaly_results.csv",
    index=False )
# Display results
anomaly_windows = int(window_df["anomaly"].sum())
anomaly_readings = int(anomaly_mask.sum())

print("\nANOMALY DETECTION RESULTS")
print("-------------------------")
print("Total windows analyzed:", len(window_df))
print("Anomalous windows:", anomaly_windows)
print("Anomalous sensor readings:", anomaly_readings)
print("Normal sensor readings:", len(X_test) - anomaly_readings)
print("\nResults saved successfully!")
print("Window results: outputs/window_detection_results.csv")
print("Test results: outputs/test_anomaly_results.csv")
print("\nAnomaly detection completed successfully!")