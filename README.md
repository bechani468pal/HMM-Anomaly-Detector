HMM-Based Machine Sensor Anomaly Detector

1. Project Overview

The HMM-Based Machine Sensor Anomaly Detector is a machine learning project that identifies unusual patterns in industrial machine sensor readings using Hidden Markov Models (HMMs).

The system learns normal machine operating behaviour and detects abnormal patterns using sequence log-likelihood and a sliding window approach.

2. Project Objectives

- Train a Gaussian Hidden Markov Model using normal sensor data.
- Calculate sequence log-likelihood.
- Establish an anomaly detection threshold using training data.
- Detect unusual sensor patterns using sliding windows.
- Visualize detected anomaly regions.
- Evaluate model performance using classification metrics.

3. Technologies Used

- Python 3.12
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- hmmlearn
- Joblib

4. Dataset Description

The project uses simulated machine sensor data containing three features:

- Temperature
- Vibration
- Pressure

Dataset details:

- Training dataset: 3,000 readings.
- Testing dataset: 1,500 readings.
- Training data represents normal machine behaviour.
- Testing data contains normal and artificially injected abnormal regions.

5. Model Development

A Gaussian Hidden Markov Model was trained using the hmmlearn library.

Different hidden-state configurations were evaluated using the Bayesian Information Criterion (BIC).

Selected model:

- Number of hidden states: 2
- Best BIC score: 25656.56

The two-state model was selected based on its BIC score.

6. Anomaly Detection

The trained model calculates the log-likelihood of sensor sequences.

A sliding window approach processes the testing data to identify unusual patterns.

Windows with likelihood scores below the selected anomaly threshold are classified as anomalous.

Selected anomaly threshold: -4.4663

7. Detection Results

The detection process produced the following results:

- Total windows analyzed: 1,401
- Anomalous windows: 408
- Anomalous sensor readings: 830
- Normal sensor readings: 670

8. Visualization

The project generates visualizations showing:

- Temperature readings and detected anomalies.
- Vibration readings and detected anomalies.
- Pressure readings and detected anomalies.
- Sequence log-likelihood and anomaly threshold behaviour.

The visualization is saved as:

outputs/anomaly_detection.png

9. Model Evaluation

The evaluation module calculates:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Evaluation results are saved in:

- outputs/evaluation_report.txt
- outputs/confusion_matrix.png

The evaluation uses the artificially injected anomaly regions as reference labels.

10. Project Structure

HMM - Anomaly - Detector/

- data/
  - normal_sensor_data.csv
  - test_sensor_data.csv
- models/
  - hmm_model.pkl
- outputs/
  - anomaly_detection.png
  - test_anomaly_results.csv
  - window_detection_results.csv
  - confusion_matrix.png
  - evaluation_report.txt
- generate_data.py
- train_model.py
- detect_anomalies.py
- visualize.py
- evaluate_model.py
- requirements.txt
- .gitignore
- README.md

11. How to Run the Project

Install dependencies:

pip install -r requirements.txt

Generate sensor datasets:

python generate_data.py

Train the HMM:

python train_model.py

Detect anomalies:

python detect_anomalies.py

Generate visualization:

python visualize.py

Evaluate model performance:

python evaluate_model.py

12. Real-World Applications

This project demonstrates how HMM-based anomaly detection can support predictive maintenance in manufacturing industries.

Potential applications include:

- Industrial machine monitoring.
- Equipment failure prevention.
- Abnormal vibration detection.
- Temperature monitoring.
- Predictive maintenance systems.

13. Conclusion

The project successfully demonstrates a sequential anomaly detection approach using Gaussian Hidden Markov Models.

It combines probabilistic modelling, likelihood-based anomaly detection, sliding window processing, visualization and model evaluation.

The system provides a foundation for developing intelligent machine mon