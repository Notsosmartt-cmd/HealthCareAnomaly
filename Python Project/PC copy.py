import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt
import time

def load_and_process_arff(file_path):
    # Load the ARFF file
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    
    # Identify the target column and exclude 'id' and 'outlier'
    target_column = 'outlier'
    excluded_columns = ['id', target_column]
    
    # Drop non-feature columns, keep only feature columns
    feature_columns = [col for col in df.columns if col not in excluded_columns]
    
    return df[feature_columns], df[target_column]

def tune_model(model, param_grid, X_train, y_train):
    # Perform GridSearchCV for the given model
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=StratifiedKFold(n_splits=3), n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def train_and_evaluate(X_train, y_train, X_test, y_test):
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define base learners
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    gb = GradientBoostingClassifier(random_state=42)
    extra_trees = ExtraTreesClassifier(random_state=42)
    svc = SVC(probability=True, random_state=42)
    
    # Define parameter grids for GridSearchCV
    param_grids = {
        'rf': {'n_estimators': [100, 200]},
        'gb': {'n_estimators': [100, 200]},
        'extra_trees': {'n_estimators': [100, 200]},
        'svc': {'C': [0.1, 1, 10]}
    }
    
    # Tune each model
    best_rf = tune_model(rf, param_grids['rf'], X_train_scaled, y_train)
    best_gb = tune_model(gb, param_grids['gb'], X_train_scaled, y_train)
    best_extra_trees = tune_model(extra_trees, param_grids['extra_trees'], X_train_scaled, y_train)
    best_svc = tune_model(svc, param_grids['svc'], X_train_scaled, y_train)
    
    # Create a Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', best_rf),
            ('gb', best_gb),
            ('extra_trees', best_extra_trees),
            ('svc', best_svc)
        ],
        voting='soft'
    )
    
    # Measure runtime
    start_time = time.time()
    
    # Train the Voting Classifier
    voting_clf.fit(X_train_scaled, y_train)
    
    # Predict and evaluate
    y_pred = voting_clf.predict(X_test_scaled)
    y_proba = voting_clf.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    runtime = time.time() - start_time
    
    # Compute feature importances from ensemble models
    importances = np.zeros(X_train.shape[1])
    for clf in [best_rf, best_extra_trees]:
        importances += clf.feature_importances_
    importances /= 2  # Average importances
    
    # Create a DataFrame for feature importances
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    })
    
    # Sort the DataFrame by importance
    feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
    
    # Plot feature importances
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Pima')
    plt.gca().invert_yaxis()
    plt.show()
    
    return accuracy, auc, runtime, feature_importances

def main(train_file_path, test_file_path):
    # Load and process training data
    X_train, y_train = load_and_process_arff(train_file_path)
    
    # Convert target variable to numeric if necessary
    y_train = pd.get_dummies(y_train, drop_first=True).values.ravel()
    
    # Load and process testing data
    X_test, y_test = load_and_process_arff(test_file_path)
    
    # Convert target variable to numeric if necessary
    y_test = pd.get_dummies(y_test, drop_first=True).values.ravel()
    
    # Align test data columns to training data columns (if necessary)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    # Train and evaluate
    accuracy, auc, runtime, feature_importances = train_and_evaluate(X_train, y_train, X_test, y_test)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Runtime: {runtime:.4f} seconds")
    print("Feature Importances:")
    print(feature_importances)

if __name__ == "__main__":
    # Replace with paths to your training and testing ARFF files
    train_file_path = '/Users/carlostorres/Documents/Python Project/Data Sets/Pima/Pima_withoutdupl_norm_02_v10.arff'
    test_file_path = '/Users/carlostorres/Documents/Python Project/Data Sets/Pima/Pima_withoutdupl_norm_35.arff'
    main(train_file_path, test_file_path)
