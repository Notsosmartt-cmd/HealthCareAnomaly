# Machine Learning for Anomaly Detection in Healthcare Datasets

This repository contains various machine learning scripts designed to detect anomalies in healthcare datasets. The focus is on leveraging different algorithms and their combinations to achieve optimal performance.

## Requirements

- Python 3.x
- Pandas
- Scikit-learn
- SciPy
- NumPy
- Matplotlib

You can install the necessary packages using pip:

```bash
pip install pandas scikit-learn scipy numpy matplotlib
```

## How to Use

### 1. Loading and Processing Data

Each script typically contains a function to load and preprocess data. For ARFF files, a sample function might look like this:

```python
from scipy.io import arff
import pandas as pd

def load_and_process_arff(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    target_column = 'outlier'
    excluded_columns = ['id', target_column]
    feature_columns = [col for col in df.columns if col not in excluded_columns]
    return df[feature_columns], df[target_column]
```

### 2. Model Training and Tuning

Scripts usually include functions for training and tuning machine learning models. For example:

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def tune_model(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=StratifiedKFold(n_splits=3), n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
```

### 3. Evaluation

Evaluation functions assess the model's performance using metrics such as accuracy and AUC:

```python
from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    return accuracy, auc
```

### 4. Running the Script

Each script has a main function that orchestrates the loading, processing, training, and evaluation steps. Replace the file paths with your specific dataset paths:

```python
def main(train_file_path, test_file_path):
    X_train, y_train = load_and_process_arff(train_file_path)
    y_train = pd.get_dummies(y_train, drop_first=True).values.ravel()
    X_test, y_test = load_and_process_arff(test_file_path)
    y_test = pd.get_dummies(y_test, drop_first=True).values.ravel()
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    # Train and evaluate your model here
```

### Example

Here is an example of how to run a script:

```python
if __name__ == "__main__":
    train_file_path = 'path/to/your/training.arff'
    test_file_path = 'path/to/your/testing.arff'
    main(train_file_path, test_file_path)
```

Run the script:

```bash
python your_script_name.py
```

## Contributing

If you have a new method or improvement, feel free to fork this repository, implement your changes, and submit a pull request. Make sure to include a description of your changes and update the README accordingly.