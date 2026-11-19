import pandas as pd
import matplotlib.pyplot as plt
import os

# Load anomaly detection results
data = pd.read_csv("outputs/test_anomaly_results.csv")

# Create output directory
os.makedirs("outputs", exist_ok=True)

# Sensor features
features = ["temperature", "vibration", "pressure"]

# Create visualization
fig, axes = plt.subplots(
    4, 1,
    figsize=(15, 13),
    sharex=True
)

fig.suptitle(
    "HMM-Based Machine Sensor Anomaly Detection",
    fontsize=16,
    fontweight="bold"
)

# Plot each sensor
for ax, feature in zip(axes[:3], features):

    ax.plot(
        data.index,
        data[feature],
        label=feature.capitalize(),
        color="blue",
        linewidth=1
    )

    # Highlight detected anomaly regions
    anomaly = data["anomaly"].values

    ax.fill_between(
        data.index,
        data[feature].min(),
        data[feature].max(),
        where=anomaly == 1,
        color="red",
        alpha=0.25,
        label="Detected Anomaly"
    )

    ax.set_ylabel(feature.capitalize())
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

# Plot log likelihood
axes[3].plot(
    data.index,
    data["log_likelihood"],
    color="purple",
    label="Log Likelihood"
)

axes[3].set_ylabel("Likelihood")
axes[3].set_xlabel("Time / Sensor Reading")
axes[3].legend(loc="upper right")
axes[3].grid(alpha=0.3)

# Save visualization
plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig(
    "outputs/anomaly_detection.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Visualization generated successfully!")
print("Saved at: outputs/anomaly_detection.png")
