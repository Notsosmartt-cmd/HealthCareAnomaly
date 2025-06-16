import pandas as pd
import numpy as np
import time
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Start measuring runtime
start_time = time.time()

# Load the dataset from the ARFF file
data = arff.loadarff("Data Sets/WPBC/WPBC_withoutdupl_norm.arff")

# Convert to pandas DataFrame
df = pd.DataFrame(data[0])

# Convert byte strings to regular strings for 'outlier' column
df['outlier'] = df['outlier'].apply(lambda x: x.decode('utf-8'))

# Check the unique values in the target column 'outlier'
print("Unique values in the dataset:")
print(df['outlier'].value_counts())

# Ensure both levels of the target variable are present
if len(df['outlier'].unique()) < 2:
    raise ValueError("The dataset must have at least two unique values for the target variable")

# Split the data into features and target
X = df.drop(columns=['outlier'])
y = df['outlier'].astype('category').cat.codes

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Building a logistic regression model with increased max_iter
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Making predictions on the test set
predictions = model.predict(X_test_scaled)

# Calculating the accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Making probability predictions on the test set for AUC calculation
prob_predictions = model.predict_proba(X_test_scaled)[:, 1]

# Calculating the AUC
auc = roc_auc_score(y_test, prob_predictions)
print(f"AUC: {auc}")

# End measuring runtime
end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime} seconds")
