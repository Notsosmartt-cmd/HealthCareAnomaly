import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed

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

def parallel_tune_models(models, param_grids, X_train, y_train):
    def tune_single_model(model, param_grid):
        return tune_model(model, param_grid, X_train, y_train)
    
    # Run GridSearchCV in parallel for all models
    results = Parallel(n_jobs=-1)(delayed(tune_single_model)(model, param_grid) 
                                  for model, param_grid in zip(models, param_grids))
    return results

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
        rf: {'n_estimators': [100, 200]},
        gb: {'n_estimators': [100, 200]},
        extra_trees: {'n_estimators': [100, 200]},
        svc: {'C': [0.1, 1, 10]}
    }
    
    models = list(param_grids.keys())
    grid_search_params = list(param_grids.values())
    
    # Parallelize model tuning
    tuned_models = parallel_tune_models(models, grid_search_params, X_train_scaled, y_train)
    
    # Create a Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', tuned_models[0]),
            ('gb', tuned_models[1]),
            ('extra_trees', tuned_models[2]),
            ('svc', tuned_models[3])
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
    for clf in [tuned_models[0], tuned_models[2]]:
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
    plt.title('Wisconsin Prognostic Breast Cancer')
    plt.gca().invert_yaxis()
    plt.show()
    
    return accuracy, auc, runtime, feature_importances

def main(file_path):
    # Load and process data
    X, y = load_and_process_arff(file_path)
    
    # Convert target variable to numeric if necessary
    y = pd.get_dummies(y, drop_first=True).values.ravel()
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train and evaluate
    accuracy, auc, runtime, feature_importances = train_and_evaluate(X_train, y_train, X_test, y_test)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Runtime: {runtime:.4f} seconds")
    print("Feature Importances:")
    print(feature_importances)

if __name__ == "__main__":
    # Replace with the path to your ARFF file
    file_path = 'Data Sets/WPBC/WPBC_withoutdupl_norm.arff'
    main(file_path)
