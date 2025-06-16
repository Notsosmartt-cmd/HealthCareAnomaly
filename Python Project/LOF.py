import pandas as pd
from scipy.io import arff
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, confusion_matrix
import time

# Load ARFF file
data, meta = arff.loadarff('Data Sets/WPBC/WPBC_withoutdupl_norm.arff')

# Convert to DataFrame
df = pd.DataFrame(data)

# Assuming the last column is the 'outlier' column
# ARFF files typically load byte strings, so we need to decode them
df['outlier'] = df['outlier'].apply(lambda x: x.decode('utf-8'))

# Convert 'yes'/'no' to 1/0 in the outlier column if necessary
df['outlier'] = df['outlier'].map({'yes': 1, 'no': 0})

# Separate features and labels
X = df.drop(columns=['outlier'])  # Drop outlier column
y = df['outlier']

# Start measuring time
start_time = time.time()

# Apply Local Outlier Factor (LOF) algorithm
lof = LocalOutlierFactor(n_neighbors=10)  # Adjust parameters as needed
lof.fit(X)

# Obtain anomaly scores
anomaly_scores = -lof.negative_outlier_factor_

# End measuring time
end_time = time.time()
runtime = end_time - start_time

# Calculate AUC-ROC score
auc_roc = roc_auc_score(y, anomaly_scores)

# Set threshold for identifying anomalies
threshold = anomaly_scores.mean() + 2 * anomaly_scores.std()

# Convert anomaly scores to binary predictions (1 for anomaly, 0 for normal)
predictions = (anomaly_scores > threshold).astype(int)

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()

# Calculate accuracy
accuracy = (tp + tn) / (tp + tn + fp + fn)

# Print AUC score and accuracy
print("AUC-ROC Score:", auc_roc)
print("Accuracy:", accuracy)
print(f"Runtime for anomaly detection: {runtime:.2f} seconds")
