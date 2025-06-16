import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configuration
FILE_PATH = 'diabetes.csv'  # Update with your file path
EMBEDDING_DIMENSIONS = 3  # Number of dimensions for vector embeddings


def main():
    # 1. Load and preprocess data
    df = pd.read_csv(FILE_PATH)
    print(f"Original data shape: {df.shape}")

    # Handle missing values (common in healthcare datasets)
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(df.iloc[:, :-1])  # Exclude outcome column

    # 2. Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Perform PCA for vector embeddings
    pca = PCA(n_components=EMBEDDING_DIMENSIONS)
    embeddings = pca.fit_transform(X_scaled)

    # 4. Print patient embeddings with indices
    print("\nPatient Vector Embeddings:")
    print(f"{'Index':<8} {'Embedding':<30} {'PCA Components':<50}")
    print("-" * 80)
    for i, embedding in enumerate(embeddings):
        print(f"{i:<8} {str(np.round(embedding, 4)):<30} {str(np.round(pca.components_, 4))}")

    # 5. Print explained variance
    print("\nExplained variance ratio:", np.round(pca.explained_variance_ratio_, 4))
    print("Total explained variance:", round(np.sum(pca.explained_variance_ratio_), 4))

    # Add this after creating the embeddings
    outcomes = df['Outcome'].values

    # 3D Plot with outcome coloring
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Color by outcome: 0=blue (no diabetes), 1=red (diabetes)
    colors = ['blue' if o == 0 else 'red' for o in outcomes]
    ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=colors, alpha=0.6)

    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    plt.title('Patient Embeddings (Blue: Healthy, Red: Diabetes)')

    # Add legend

    healthy_patch = mpatches.Patch(color='blue', label='No Diabetes')
    diabetes_patch = mpatches.Patch(color='red', label='Diabetes')
    plt.legend(handles=[healthy_patch, diabetes_patch])

    # 3D Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embeddings[:,0], embeddings[:,1], embeddings[:,2])
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    plt.title('Patient Vector Embeddings')
    plt.show()


if __name__ == "__main__":
    main()