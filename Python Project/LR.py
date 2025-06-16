import pandas as pd
import numpy as np
import time
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Start measuring runtime
start_time = time.time()

# Load the training and testing data from the ARFF files
training_data = arff.loadarff("/Users/carlostorres/Documents/Python Project/Data Sets/Pima/Pima_withoutdupl_norm_02_v10.arff")
testing_data = arff.loadarff("/Users/carlostorres/Documents/Python Project/Data Sets/Pima/Pima_withoutdupl_norm_35.arff")

# Convert to pandas DataFrame
training_df = pd.DataFrame(training_data[0])
testing_df = pd.DataFrame(testing_data[0])

# Convert byte strings to regular strings for 'outlier' column
training_df['outlier'] = training_df['outlier'].apply(lambda x: x.decode('utf-8'))
testing_df['outlier'] = testing_df['outlier'].apply(lambda x: x.decode('utf-8'))

# Check the unique values in the target column 'outlier' in the training data
print("Unique values in training data:")
print(training_df['outlier'].value_counts())

# Check the unique values in the target column 'outlier' in the testing data
print("Unique values in testing data:")
print(testing_df['outlier'].value_counts())

# Ensure both levels of the target variable are present in the training data
if len(training_df['outlier'].unique()) < 2:
    raise ValueError("The training data must have at least two unique values for the target variable")

# Splitting data into features and target
X_train = training_df.drop(columns=['outlier'])
y_train = training_df['outlier'].astype('category').cat.codes
X_test = testing_df.drop(columns=['outlier'])
y_test = testing_df['outlier'].astype('category').cat.codes

# Ensure the columns in X_test match those in X_train
X_test = X_test[X_train.columns]

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
