import pandas as pd
import numpy as np
import time
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from pyod.models.iforest import IForest

# Start measuring runtime
start_time = time.time()

# Load ARFF data
#data, meta = arff.loadarff('Data Sets/WPBC/WPBC_withoutdupl_norm.arff')
data, meta = arff.loadarff('Data Sets/Pima/Pima_withoutdupl_02_v01.arff')
df = pd.DataFrame(data)

# Convert byte strings to regular strings for 'outlier' column
df['outlier'] = df['outlier'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Ensure both levels of the target variable are present
print("Unique values in the 'outlier' column:")
print(df['outlier'].value_counts())

# Drop non-feature columns (if any other than 'outlier') and keep features
X = df.drop(columns=['outlier'])
y = df['outlier'].astype('category').cat.codes  # Convert 'yes'/'no' to 1/0

# Convert categorical columns to numeric (if applicable)
X = pd.get_dummies(X)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and fit Isolation Forest
model = IForest()
model.fit(X_scaled)

# Predict anomalies
y_test_pred = model.predict(X_scaled)  # 1 for anomaly, 0 for normal

# End measuring runtime
end_time = time.time()
runtime = end_time - start_time

# Calculate accuracy and AUC
accuracy = accuracy_score(y, y_test_pred)
prob_predictions = model.decision_scores_  # Raw anomaly scores
auc = roc_auc_score(y, prob_predictions)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Runtime: {runtime:.4f} seconds")

# Print anomaly scores
print("Anomaly scores:", model.decision_scores_)
print("Predicted anomalies (1 indicates anomaly, 0 indicates normal):", y_test_pred)
