import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

# Function to read all CSV files from a given folder
def read_csv_files_from_folder(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = {}

    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(folder_path, csv_file))
        # Select specific columns
        selected_columns = ["GO1", "GO2", "PANI_F1", "Mg"]
        df = df[selected_columns]
        dataframes[csv_file] = df
    return dataframes

# Function to compute the difference between two dataframes
def compute_difference(dataframes):
    if len(dataframes) != 2:
        raise ValueError("There should be exactly two CSV files in the folder.")
    (csv_name1, df1), (csv_name2, df2) = dataframes.items()

    # Adjust dataframes to have the same shape
    min_rows = min(df1.shape[0], df2.shape[0])
    min_cols = min(df1.shape[1], df2.shape[1])
    
    df1 = df1.iloc[:min_rows, :min_cols]
    df2 = df2.iloc[:min_rows, :min_cols]

    difference_df = df1 - df2
    return difference_df

# Path to the parent folder containing subfolders
parent_folder_path = "Tuber-Data-TBH"

# Get a list of all subfolders
subfolders = [f.path for f in os.scandir(parent_folder_path) if f.is_dir()]

# Dictionary to store the differences for each folder
folder_differences = {}

# Read CSV files and compute differences for each folder
for folder in subfolders:
    dataframes = read_csv_files_from_folder(folder)
    folder_name = os.path.basename(folder)
    folder_differences[folder_name] = compute_difference(dataframes)

# Get column names (assuming all CSVs have the same columns)
columns = next(iter(folder_differences.values())).columns

selected_columns = ["GO1", "GO2", "PANI_F1", "Mg"]

# Iterate over the folders and compute the combined feature
combined_feature_data = []
folder_names = []

for folder_name, diff_df in folder_differences.items():
    # Extract each selected column without scaling
    columns_data = []
    for col in selected_columns:
        column_data = diff_df[col].values.flatten()
        columns_data.append(column_data)
    
    # Combine the columns into a single feature with 600 values
    combined_feature = np.concatenate(columns_data)
    
    combined_feature_data.append(combined_feature)
    folder_names.append(folder_name)

# Convert to a numpy array for clustering
combined_feature_data = np.array(combined_feature_data)
print(combined_feature_data,combined_feature_data.shape)

# Convert to a numpy array for clustering
# combined_feature_data = np.array(combined_fea|ture_data)

# Apply Agglomerative Clustering
agglo_cluster = AgglomerativeClustering(n_clusters=2)
agglo_labels = agglo_cluster.fit_predict(combined_feature_data)

# Perform PCA to reduce the dimensionality to 2D for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(combined_feature_data)

# Visualization
plt.figure(figsize=(12, 10))
for i, folder_name in enumerate(folder_names):
    plt.scatter(reduced_data[i, 0], reduced_data[i, 1], color='blue' if agglo_labels[i] == 0 else 'red')
    plt.text(reduced_data[i, 0] + 0.02, reduced_data[i, 1] + 0.02, folder_name, fontsize=9)

plt.title('Agglomerative Clustering Visualization with Folder Names')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.show()

# Create a dataframe with results
result_df = pd.DataFrame({
    'Folder': folder_names,
    'Agglomerative Cluster': agglo_labels
})

# Display the resulting dataframe
print(result_df)


import joblib

# Save the model after fitting
model_path = 'agglo_cluster_model.pkl'
joblib.dump(agglo_cluster, model_path)
print(f"Model saved to {model_path}")


import joblib
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

# Load the previously saved model
model_path = 'agglo_cluster_model.pkl'
loaded_model = joblib.load(model_path)

# Example new test sample (replace with actual test sample)
new_test_sample = np.random.rand(1, combined_feature_data.shape[1])  # Replace with actual test data

# Get the labels for the existing dataset
agglo_labels = loaded_model.fit_predict(combined_feature_data)

# Compute the centroids for each cluster
centroids = []
for label in np.unique(agglo_labels):
    centroid = combined_feature_data[agglo_labels == label].mean(axis=0)
    centroids.append(centroid)

centroids = np.array(centroids)

# Find the nearest centroid to the new test sample
distances = cdist(new_test_sample, centroids)
closest_centroid_index = np.argmin(distances)
predicted_label = closest_centroid_index

print(f"Predicted Label for the new sample: {predicted_label}")

# (Optional) Retrain the model after adding the new test sample
updated_feature_data = np.vstack([combined_feature_data, new_test_sample])

# Re-run Agglomerative Clustering on the updated dataset
agglo_cluster_updated = AgglomerativeClustering(n_clusters=2)
agglo_labels_updated = agglo_cluster_updated.fit_predict(updated_feature_data)

# Save the updated model
model_path_updated = 'agglo_cluster_model_updated.pkl'
joblib.dump(agglo_cluster_updated, model_path_updated)
print(f"Updated model saved to {model_path_updated}")