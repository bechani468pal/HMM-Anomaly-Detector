import numpy as np
import pandas as pd
import os

# Create reproducible sensor data
np.random.seed(42)

os.makedirs("data", exist_ok=True)

# Generate normal machine sensor readings
def generate_normal_data(n_samples=3000):
    temperature = np.random.normal(70, 2, n_samples)
    vibration = np.random.normal(5, 0.4, n_samples)
    pressure = np.random.normal(100, 3, n_samples)

    return pd.DataFrame({
        "temperature": temperature,
        "vibration": vibration,
        "pressure": pressure
    })


# Generate training data (normal operation only)
train_data = generate_normal_data(3000)
train_data.to_csv("data/normal_sensor_data.csv", index=False)

# Generate test data
test_data = generate_normal_data(1500)

# Inject abnormal machine behaviour
test_data.loc[400:470, "temperature"] += 18
test_data.loc[400:470, "vibration"] += 3

test_data.loc[900:960, "pressure"] += 25
test_data.loc[900:960, "vibration"] += 2

# Save test dataset
test_data.to_csv("data/test_sensor_data.csv", index=False)

print("Sensor datasets generated successfully!")
print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)
print("Normal and anomalous test regions created.")
