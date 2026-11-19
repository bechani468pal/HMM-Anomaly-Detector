import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report)
# Load detection results
data = pd.read_csv("outputs/test_anomaly_results.csv")

# Create actual labels based on injected anomalies
data["actual_anomaly"] = 0
# First injected anomaly region
data.loc[400:470, "actual_anomaly"] = 1
# Second injected anomaly region
data.loc[900:960, "actual_anomaly"] = 1
# Actual and predicted labels
y_true = data["actual_anomaly"]
y_pred = data["anomaly"]

# Calculate performance metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
print("\nMODEL PERFORMANCE EVALUATION")
print("=" * 40)
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Normal", "Anomaly"],
    zero_division=0
))
# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
# Visualize confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Anomaly"],
    yticklabels=["Normal", "Anomaly"]
)
plt.title("HMM Anomaly Detection - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
os.makedirs("outputs", exist_ok=True)
plt.tight_layout()
plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300)
plt.show()
# Save evaluation results
with open("outputs/evaluation_report.txt", "w") as file:
    file.write("HMM ANOMALY DETECTION EVALUATION\n")
    file.write("=" * 40 + "\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1 Score: {f1:.4f}\n\n")
    file.write(classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Anomaly"],
        zero_division=0 ))
print("\nEvaluation completed successfully!")
print("Confusion matrix saved in outputs/")
print("Evaluation report saved in outputs/")
