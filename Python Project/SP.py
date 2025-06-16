import pandas as pd
from scipy.io import arff
from pyod.models.copod import COPOD
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load ARFF file
data, meta = arff.loadarff('Data Sets/WPBC/WPBC_withoutdupl_norm.arff')
df = pd.DataFrame(data)

# Decode byte strings and convert 'yes'/'no' to 1/0 in the outlier column
df['outlier'] = df['outlier'].apply(lambda x: 1 if x.decode('utf-8') == 'yes' else 0)

# Separate features and labels
X = df.drop(columns=['outlier'])  # Drop outlier column
y = df['outlier']

# Start measuring time
start_time = time.time()

# Apply COPOD for anomaly detection
copod = COPOD()
copod.fit(X)

# Predict anomalies
predictions = copod.labels_  # Binary labels: 0 for normal, 1 for anomalies

# End measuring time
end_time = time.time()
runtime = end_time - start_time

# Add predictions to the dataframe for visualization
df['anomaly'] = predictions

# Evaluate the model
print(classification_report(y, predictions, target_names=['Normal', 'Anomaly']))
print(confusion_matrix(y, predictions))

# Print runtime
print(f"Runtime for anomaly detection: {runtime:.2f} seconds")

# Visualize the anomalies using scatter plot (example using the first two features)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1], hue='anomaly', palette={0: 'blue', 1: 'red'})
plt.title("Anomaly Detection")
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.legend(title='Anomaly', loc='upper left', labels=['Normal', 'Anomaly'])
plt.show()

# Obtain anomaly scores
anomaly_scores = copod.decision_scores_

# Calculate AUC score
auc_score = roc_auc_score(y, -anomaly_scores)  # The '-' sign is used because higher scores indicate anomalies

# Print AUC score
print("AUC Score:", auc_score)
