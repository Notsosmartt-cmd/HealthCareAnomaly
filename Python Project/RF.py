import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load ARFF file
data, meta = arff.loadarff('/Users/carlostorres/Documents/Python Project/Data Sets/WPBC/WPBC_withoutdupl_norm.arff')
df = pd.DataFrame(data)

# Convert byte strings to regular strings
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.decode('utf-8')

# Convert 'yes'/'no' to 1/0 in the outlier column
df['outlier'] = df['outlier'].map({'yes': 1, 'no': 0})

# Separate features and labels
X = df.drop(columns=['id', 'outlier'])  # Drop ID and outlier columns, assuming 'id' is a column
y = df['outlier']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start measuring time
start_time = time.time()

# Apply Random Forest Classifier for anomaly detection
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict anomalies on the test set
predictions = rf.predict(X_test)

# End measuring time
end_time = time.time()
runtime = end_time - start_time

# Add predictions to the test dataframe for visualization
test_df = X_test.copy()
test_df['anomaly'] = predictions

# Evaluate the model
print(classification_report(y_test, predictions, target_names=['Normal', 'Anomaly']))
print(confusion_matrix(y_test, predictions))

# Print runtime
print(f"Runtime for anomaly detection: {runtime:.2f} seconds")

# Visualize the anomalies using scatter plot (example using the first two features)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=test_df, x=test_df.columns[0], y=test_df.columns[1], hue='anomaly', palette={0: 'blue', 1: 'red'})
plt.title("Anomaly Detection")
plt.xlabel(test_df.columns[0])
plt.ylabel(test_df.columns[1])
plt.legend(title='Anomaly', loc='upper left', labels=['Normal', 'Anomaly'])
plt.show()

# Obtain anomaly scores (in this case, probabilities of being an anomaly)
anomaly_scores = rf.predict_proba(X_test)[:, 1]

# Calculate AUC score
auc_score = roc_auc_score(y_test, anomaly_scores)

# Print AUC score
print("AUC Score:", auc_score)
