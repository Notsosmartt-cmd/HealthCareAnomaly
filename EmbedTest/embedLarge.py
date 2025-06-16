import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from scipy.io import arff

# Configuration
ARFF_DIRECTORY = "../Data_Sets/Pima"  # Update with your directory path
EMBEDDING_DIMENSIONS = 3  # Number of dimensions for vector embeddings
OUTPUT_DIR = 'unified_embeddings_output/'  # Directory to save results


def process_arff_file(file_path):
    """Process a single ARFF file and generate embeddings"""
    # Load ARFF file
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    # Decode byte strings to regular strings
    str_df = df.select_dtypes([object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        df[col] = str_df[col]

    print(f"\nProcessing: {os.path.basename(file_path)}")
    print(f"Original data shape: {df.shape}")
    print(f"Features: {list(df.columns)}")

    # Try to find outcome column automatically
    outcome_col = None
    for col in ['Outcome', 'Class', 'Target', 'Result']:
        if col in df.columns:
            outcome_col = col
            break

    # Prepare features and outcome
    if outcome_col:
        features = df.drop(outcome_col, axis=1)
        outcomes = df[outcome_col]
        print(f"Using '{outcome_col}' as outcome variable")
    else:
        features = df
        outcomes = None
        print("No outcome variable found - using unsupervised mode")

    # Convert object columns to numeric where possible
    for col in features.columns:
        if features[col].dtype == 'object':
            try:
                features[col] = pd.to_numeric(features[col])
            except ValueError:
                # Handle non-numeric columns (simple approach)
                features = pd.get_dummies(features, columns=[col])

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(features)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA for vector embeddings
    pca = PCA(n_components=EMBEDDING_DIMENSIONS)
    embeddings = pca.fit_transform(X_scaled)

    # Create results directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.basename(file_path).replace('.arff', '')

    # Save embeddings to CSV
    embedding_df = pd.DataFrame(embeddings, columns=[f'PC{i + 1}' for i in range(EMBEDDING_DIMENSIONS)])
    if outcomes is not None:
        embedding_df['Outcome'] = outcomes.values
    embedding_df.to_csv(f"{OUTPUT_DIR}{base_name}_embeddings.csv", index=False)

    # Print explained variance
    print(f"Explained variance ratio: {np.round(pca.explained_variance_ratio_, 4)}")
    print(f"Total explained variance: {round(np.sum(pca.explained_variance_ratio_), 4)}")

    # Generate visualizations
    visualize_embeddings(embeddings, outcomes, base_name)


def visualize_embeddings(embeddings, outcomes, base_name):
    """Create 2D and 3D visualizations of embeddings"""
    # 3D Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create color mapping if outcomes exist
    if outcomes is not None:
        # Convert outcome to numeric codes if needed
        if outcomes.dtype == 'object':
            unique_outcomes = outcomes.unique()
            color_map = {outcome: plt.cm.tab10(i) for i, outcome in enumerate(unique_outcomes)}
            colors = [color_map[o] for o in outcomes]

            # Create legend
            patches = [mpatches.Patch(color=color_map[o], label=o) for o in unique_outcomes]
            plt.legend(handles=patches)
        else:
            # Numeric outcomes
            colors = outcomes
            plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), label='Outcome Value')
    else:
        colors = 'blue'

    # Create 3D scatter plot
    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                         c=colors, alpha=0.7, s=50)

    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    plt.title(f'{base_name} - 3D Embeddings')
    plt.savefig(f"{OUTPUT_DIR}{base_name}_3d.png")

    # 2D Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, alpha=0.7, s=50)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'{base_name} - 2D Embeddings')
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}{base_name}_2d.png")
    plt.close('all')


def main():
    # Process all ARFF files in directory
    for file_name in os.listdir(ARFF_DIRECTORY):
        if file_name.endswith('.arff'):
            file_path = os.path.join(ARFF_DIRECTORY, file_name)
            try:
                process_arff_file(file_path)
                print(f"Successfully processed {file_name}\n{'=' * 80}")
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}\n{'=' * 80}")


if __name__ == "__main__":
    main()