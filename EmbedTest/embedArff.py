import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
from collections import defaultdict
import os

# Configuration
ARFF_FILE = '../Data_Sets/Pima/Pima_withoutdupl_02_v02.arff'  # Path to your ARFF file
EMBEDDING_DIMENSIONS = 3  # Number of dimensions for vector embeddings
MODEL_FILE = 'patient_embeddings_model.pkl'  # Output model filename


class PatientEmbeddingModel:
    """Stores patient embeddings and related information"""

    def __init__(self):
        self.embeddings = {}
        self.pca_model = None
        self.feature_names = []
        self.explained_variance = None
        self.pca_components = None

    def add_patient(self, patient_id, embedding, features):
        """Add a patient to the model"""
        self.embeddings[patient_id] = {
            'embedding': embedding,
            'features': features
        }

    def save(self, filename):
        """Save the model to a file"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load model from file"""
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def get_embedding_matrix(self):
        """Return embeddings as a matrix (patients x dimensions)"""
        return np.array([data['embedding'] for data in self.embeddings.values()])

    def get_patient_ids(self):
        """Return list of patient IDs"""
        return list(self.embeddings.keys())


def main():
    print(f"Loading ARFF file: {ARFF_FILE}")

    # Load ARFF data
    data, meta = arff.loadarff(ARFF_FILE)
    df = pd.DataFrame(data)

    # Convert byte strings to regular strings
    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].str.decode('utf-8')

    # Identify feature columns (exclude ID and target)
    feature_cols = [col for col in df.columns
                    if col not in ['id', 'outlier']
                    and df[col].dtype in ['float64', 'int64']]

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(df[feature_cols])

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA for vector embeddings
    pca = PCA(n_components=EMBEDDING_DIMENSIONS)
    embeddings = pca.fit_transform(X_scaled)

    # Create and populate the embedding model
    model = PatientEmbeddingModel()
    model.feature_names = feature_cols
    model.pca_model = pca
    model.explained_variance = pca.explained_variance_ratio_
    model.pca_components = pca.components_

    # Add each patient to the model
    for i, row in df.iterrows():
        patient_id = row['id'] if 'id' in df.columns else i
        model.add_patient(
            patient_id=patient_id,
            embedding=embeddings[i],
            features=row[feature_cols].to_dict()
        )

    # Save the model
    model.save(MODEL_FILE)
    print(f"\nSaved embedding model to: {MODEL_FILE}")
    print(f"Number of patients embedded: {len(model.embeddings)}")

    # Print model summary
    print("\nModel Summary:")
    print(f"Embedding dimensions: {EMBEDDING_DIMENSIONS}")
    print(f"Features used: {', '.join(feature_cols)}")
    print(f"Total explained variance: {np.sum(model.explained_variance):.4f}")

    # Print explained variance per component
    print("\nExplained variance per component:")
    for i, var in enumerate(model.explained_variance):
        print(f"PC{i + 1}: {var:.4f} ({var * 100:.1f}%)")

    # Prepare target for coloring
    if 'outlier' in df.columns:
        target = df['outlier']
        color_map = {'no': 'blue', 'yes': 'red'}
        colors = target.map(color_map).values
        legend_labels = [
            mpatches.Patch(color='blue', label='Normal'),
            mpatches.Patch(color='red', label='Outlier')
        ]
    else:
        colors = 'blue'
        legend_labels = None

    # Get embedding matrix
    embedding_matrix = model.get_embedding_matrix()

    # 3D Plot with coloring
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each patient
    scatter = ax.scatter(
        embedding_matrix[:, 0],
        embedding_matrix[:, 1],
        embedding_matrix[:, 2],
        c=colors,
        alpha=0.7,
        s=50,
        depthshade=True
    )

    ax.set_xlabel(f'PC1 ({model.explained_variance[0] * 100:.1f}%)')
    ax.set_ylabel(f'PC2 ({model.explained_variance[1] * 100:.1f}%)')
    ax.set_zlabel(f'PC3 ({model.explained_variance[2] * 100:.1f}%)')
    plt.title('Patient Embeddings Space')

    # Add legend if we have target colors
    if legend_labels:
        plt.legend(handles=legend_labels)

    # Add grid and adjust layout
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save and show plot
    plot_filename = os.path.splitext(ARFF_FILE)[0] + '_embeddings.png'
    plt.savefig(plot_filename, dpi=150)
    print(f"\nSaved visualization: {plot_filename}")
    plt.show()

    # Print sample embeddings
    print("\nSample patient embeddings:")
    patient_ids = model.get_patient_ids()[:5]  # First 5 patients
    for pid in patient_ids:
        emb = model.embeddings[pid]['embedding']
        print(f"Patient {pid}: {np.round(emb, 4)}")

    # Print PCA component loadings
    print("\nTop feature loadings per principal component:")
    components_df = pd.DataFrame(
        model.pca_components,
        columns=model.feature_names,
        index=[f'PC{i + 1}' for i in range(EMBEDDING_DIMENSIONS)]
    )

    for i in range(EMBEDDING_DIMENSIONS):
        print(f"\nPC{i + 1} ({model.explained_variance[i] * 100:.1f}%):")
        sorted_loadings = components_df.iloc[i].sort_values(key=abs, ascending=False)
        print(sorted_loadings.head(5))  # Top 5 features per component


if __name__ == "__main__":
    main()