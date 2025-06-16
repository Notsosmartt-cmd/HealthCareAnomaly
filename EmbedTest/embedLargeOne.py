import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from scipy.io import arff
import re

# Configuration
ARFF_DIRECTORY = "../Data_Sets/Pima"  # Update with your directory path
EMBEDDING_DIMENSIONS = 3  # Number of dimensions for vector embeddings
OUTPUT_DIR = 'unified_embeddings_output/'  # Directory to save results
EXCLUDE_COLUMNS = ['id', 'outlier', 'patient_id', 'record_id']  # Columns to exclude from embedding


def load_and_preprocess_arff(file_path):
    """Load and preprocess a single ARFF file, excluding specified columns"""
    # Load ARFF file
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    # Decode byte strings to regular strings
    str_df = df.select_dtypes([object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        df[col] = str_df[col]

    # Extract dataset name
    dataset_name = os.path.basename(file_path).replace('.arff', '')

    # Identify columns to exclude (case-insensitive)
    actual_exclude = []
    for col in df.columns:
        if any(excl.lower() == col.lower() for excl in EXCLUDE_COLUMNS):
            actual_exclude.append(col)
            print(f"Excluding column: {col}")

    # Prepare features and metadata
    features = df.drop(columns=actual_exclude, errors='ignore')
    metadata = df[actual_exclude] if actual_exclude else pd.DataFrame()

    # Try to find outcome column automatically
    outcome_col = None
    for col in ['Outcome', 'Class', 'Target', 'Result']:
        if col in features.columns:
            outcome_col = col
            break

    # Prepare features and outcome
    if outcome_col:
        outcomes = features[outcome_col]
        features = features.drop(outcome_col, axis=1)
    else:
        outcomes = None

    # Convert object columns to numeric where possible
    for col in features.columns:
        if features[col].dtype == 'object':
            try:
                features[col] = pd.to_numeric(features[col])
            except ValueError:
                # Handle non-numeric columns by attempting to extract numbers
                features[col] = features[col].apply(
                    lambda x: float(re.search(r'[\d\.]+', str(x)).group())
                    if re.search(r'[\d\.]+', str(x)) else np.nan
                )

    return features, outcomes, metadata, dataset_name


def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect all data
    all_features = []
    all_outcomes = []
    all_metadata = []
    all_origins = []  # Track dataset origins separately

    print("Loading and preprocessing ARFF files...")
    for file_name in os.listdir(ARFF_DIRECTORY):
        if file_name.endswith('.arff'):
            file_path = os.path.join(ARFF_DIRECTORY, file_name)
            try:
                features, outcomes, metadata, dataset_name = load_and_preprocess_arff(file_path)
                all_features.append(features)
                all_metadata.append(metadata)

                # Track dataset origins
                origins = pd.Series([dataset_name] * len(features))
                all_origins.append(origins)

                if outcomes is not None:
                    # Add dataset name to outcome for uniqueness
                    all_outcomes.append(outcomes.astype(str) + f" ({dataset_name})")
                else:
                    all_outcomes.append(pd.Series([dataset_name] * len(features)))

                print(f"Processed: {file_name} ({len(features)} records)")
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")

    if not all_features:
        print("No ARFF files found. Exiting.")
        return

    # Combine all datasets
    combined_features = pd.concat(all_features, ignore_index=True)
    combined_outcomes = pd.concat(all_outcomes, ignore_index=True)
    combined_metadata = pd.concat(all_metadata, ignore_index=True)
    combined_origins = pd.concat(all_origins, ignore_index=True)

    print(f"\nCombined dataset shape: {combined_features.shape}")
    print(f"Total patients: {len(combined_features)}")
    print(f"Unique datasets: {combined_origins.nunique()}")

    # Handle missing values using global median (only on numerical features)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(combined_features)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Perform PCA for unified vector embeddings
    pca = PCA(n_components=EMBEDDING_DIMENSIONS)
    embeddings = pca.fit_transform(X_scaled)

    # Save unified embeddings
    embedding_df = pd.DataFrame(embeddings, columns=[f'PC{i + 1}' for i in range(EMBEDDING_DIMENSIONS)])
    embedding_df['Outcome'] = combined_outcomes
    embedding_df['Dataset'] = combined_origins

    # Add metadata back to embeddings
    for col in combined_metadata.columns:
        embedding_df[col] = combined_metadata[col]

    embedding_df.to_csv(f"{OUTPUT_DIR}unified_embeddings.csv", index=False)

    # Print explained variance
    print(f"\nExplained variance ratio: {np.round(pca.explained_variance_ratio_, 4)}")
    print(f"Total explained variance: {round(np.sum(pca.explained_variance_ratio_), 4)}")

    # Visualize unified embeddings
    visualize_unified_embeddings(embedding_df, pca)


def visualize_unified_embeddings(embedding_df, pca):
    """Create visualizations of unified embeddings"""
    # Prepare colors by dataset
    datasets = embedding_df['Dataset'].unique()
    color_map = {ds: plt.cm.tab20(i % 20) for i, ds in enumerate(datasets)}
    colors = [color_map[ds] for ds in embedding_df['Dataset']]

    # 3D Plot by dataset
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each point with dataset color
    ax.scatter(
        embedding_df['PC1'],
        embedding_df['PC2'],
        embedding_df['PC3'],
        c=colors, alpha=0.7, s=30
    )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
    plt.title("Unified Patient Embeddings by Dataset")

    # Create legend
    patches = [mpatches.Patch(color=color_map[ds], label=ds) for ds in datasets]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}unified_embeddings_by_dataset.png")

    # 2D Plot by outcome if available
    if 'outlier' in embedding_df.columns:
        plt.figure(figsize=(14, 10))
        outlier_colors = {'yes': 'red', 'no': 'blue'}
        colors = [outlier_colors.get(str(x).lower(), 'gray') for x in embedding_df['outlier']]

        plt.scatter(
            embedding_df['PC1'],
            embedding_df['PC2'],
            c=colors, alpha=0.6
        )

        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        plt.title("Unified Patient Embeddings by Outlier Status")

        # Create legend
        patches = [
            mpatches.Patch(color='red', label='Outlier (yes)'),
            mpatches.Patch(color='blue', label='Non-outlier (no)')
        ]
        plt.legend(handles=patches)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}unified_embeddings_by_outlier.png")

    # 3D Plot with outlier highlighting
    if 'outlier' in embedding_df.columns:
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Plot non-outliers
        non_outliers = embedding_df[embedding_df['outlier'].str.lower() == 'no']
        ax.scatter(
            non_outliers['PC1'],
            non_outliers['PC2'],
            non_outliers['PC3'],
            c='blue', alpha=0.3, s=20, label='Non-outlier'
        )

        # Plot outliers
        outliers = embedding_df[embedding_df['outlier'].str.lower() == 'yes']
        ax.scatter(
            outliers['PC1'],
            outliers['PC2'],
            outliers['PC3'],
            c='red', alpha=1.0, s=50, label='Outlier'
        )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
        plt.title("Outlier Detection in Unified Embeddings")
        plt.legend()
        plt.savefig(f"{OUTPUT_DIR}unified_outlier_detection.png")

    plt.show()

    # Save PCA loadings for interpretation
#    if hasattr(pca, 'components_'):
#        # Get feature names
#        feature_names = combined_features.columns
#        loadings_df = pd.DataFrame(
#            pca.components_.T,
#            columns=[f'PC{i + 1}' for i in range(EMBEDDING_DIMENSIONS)],
#            index=feature_names
#        )
#        loadings_df.to_csv(f"{OUTPUT_DIR}pca_loadings.csv")
#        print(f"\nPCA loadings saved to {OUTPUT_DIR}pca_loadings.csv")


if __name__ == "__main__":
    main()