# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.decomposition import PCA
# from sklearn.cluster import AgglomerativeClustering
# from scipy.spatial.distance import cdist
# import joblib

# # Function to read all CSV files from a given folder
# def read_csv_files_from_folder(folder_path):
#     csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
#     dataframes = {}

#     for csv_file in csv_files:
#         df = pd.read_csv(os.path.join(folder_path, csv_file))
#         # Select specific columns
#         selected_columns = ["GO1", "GO2", "PANI_F1", "Mg"]
#         df = df[selected_columns]
#         dataframes[csv_file] = df
#     return dataframes

# # Function to compute the difference between two dataframes
# def compute_difference(dataframes):
#     if len(dataframes) != 2:
#         raise ValueError("There should be exactly two CSV files in the folder.")
#     (csv_name1, df1), (csv_name2, df2) = dataframes.items()

#     # Adjust dataframes to have the same shape
#     min_rows = min(df1.shape[0], df2.shape[0])
#     min_cols = min(df1.shape[1], df2.shape[1])
    
#     df1 = df1.iloc[:min_rows, :min_cols]
#     df2 = df2.iloc[:min_rows, :min_cols]

#     difference_df = df1 - df2
#     return difference_df

# # Path to the parent folder containing subfolders
# parent_folder_path = "Tuber-Data-TBH"

# # Get a list of all subfolders
# subfolders = [f.path for f in os.scandir(parent_folder_path) if f.is_dir()]

# # Dictionary to store the differences for each folder
# folder_differences = {}

# # Read CSV files and compute differences for each folder
# for folder in subfolders:
#     dataframes = read_csv_files_from_folder(folder)
#     folder_name = os.path.basename(folder)
#     folder_differences[folder_name] = compute_difference(dataframes)

# # Get column names (assuming all CSVs have the same columns)
# columns = next(iter(folder_differences.values())).columns

# selected_columns = ["GO1", "GO2", "PANI_F1", "Mg"]

# # Iterate over the folders and compute the combined feature
# combined_feature_data = []
# folder_names = []

# for folder_name, diff_df in folder_differences.items():
#     # Extract each selected column without scaling
#     columns_data = []
#     for col in selected_columns:
#         column_data = diff_df[col].values.flatten()
#         columns_data.append(column_data)
    
#     # Combine the columns into a single feature with 600 values
#     combined_feature = np.concatenate(columns_data)
    
#     combined_feature_data.append(combined_feature)
#     folder_names.append(folder_name)

# # Convert to a numpy array for clustering
# combined_feature_data = np.array(combined_feature_data)
# print(combined_feature_data,combined_feature_data.shape)

# # Convert to a numpy array for clustering
# # combined_feature_data = np.array(combined_fea|ture_data)

# # Apply Agglomerative Clustering
# agglo_cluster = AgglomerativeClustering(n_clusters=2)
# agglo_labels = agglo_cluster.fit_predict(combined_feature_data)

# # Perform PCA to reduce the dimensionality to 2D for visualization
# pca = PCA(n_components=2)
# reduced_data = pca.fit_transform(combined_feature_data)

# # Visualization
# plt.figure(figsize=(12, 10))
# for i, folder_name in enumerate(folder_names):
#     plt.scatter(reduced_data[i, 0], reduced_data[i, 1], color='blue' if agglo_labels[i] == 0 else 'red')
#     plt.text(reduced_data[i, 0] + 0.02, reduced_data[i, 1] + 0.02, folder_name, fontsize=9)

# plt.title('Agglomerative Clustering Visualization with Folder Names')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.grid(True)
# plt.show()

# # Create a dataframe with results
# result_df = pd.DataFrame({
#     'Folder': folder_names,
#     'Agglomerative Cluster': agglo_labels
# })

# # Display the resulting dataframe
# print(result_df)

# #---------------------------------------------------------------------------------------------------------------------------------

# # Save the model after fitting
# MODEL_PATH = 'agglo_cluster_model.pkl'
# joblib.dump(agglo_cluster, MODEL_PATH)
# print(f"Model saved to {MODEL_PATH}")

# #-------------------------------------------------------------------------------------------------------------------------------


# # Load the previously saved model
# MODEL_PATH = 'agglo_cluster_model.pkl'
# loaded_model = joblib.load(MODEL_PATH)

# # Example new test sample (replace with actual test sample)
# new_test_sample = np.random.rand(1, combined_feature_data.shape[1])  # Replace with actual test data

# # Get the labels for the existing dataset
# agglo_labels = loaded_model.fit_predict(combined_feature_data)

# # Compute the centroids for each cluster
# centroids = []
# for lable in np.unique(agglo_labels):
#     centroid = combined_feature_data[agglo_labels == lable].mean(axis=0)
#     centroids.append(centroid)

# centroids = np.array(centroids)

# # Find the nearest centroid to the new test sample
# distances = cdist(new_test_sample, centroids)
# closest_centroid_index = np.argmin(distances)
# predicted_label = closest_centroid_index

# print(f"Predicted Label for the new sample: {predicted_label}")

# # (Optional) Retrain the model after adding the new test sample
# updated_feature_data = np.vstack([combined_feature_data, new_test_sample])

# # Re-run Agglomerative Clustering on the updated dataset
# agglo_cluster_updated = AgglomerativeClustering(n_clusters=2)
# agglo_labels_updated = agglo_cluster_updated.fit_predict(updated_feature_data)

# # Save the updated model
# model_path_updated = 'agglo_cluster_model_updated.pkl'
# joblib.dump(agglo_cluster_updated, model_path_updated)
# print(f"Updated model saved to {model_path_updated}")

# #---------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from .schema import ResultInfoModel
import os
import json

# # Step 2: Retrain with New Data
# # Assuming X_train_new and y_train_new are the new training data
# loaded_model.fit(X_train_new, y_train_new)
# # Save the updated model
# joblib.dump(loaded_model, 'xgboost_model_updated.pkl')
# print("Model retrained and saved as 'xgboost_model_updated.pkl'")
# # Step 3: Inference with Updated Model
# # Load the updated model
# updated_model = joblib.load('xgboost_model_updated.pkl')
# # Predict class for new input data
# new_data = X_test[0].reshape(1, -1)  # Replace X_test[0] with your actual input data
# predicted_label = updated_model.predict(new_data)
# # Map numerical lable to class lable
# class_mapping = {0: 'TB-Negative', 1: 'TB-Positive'}
# predicted_class = class_mapping[predicted_label[0]]
# # Output the predicted class
# print(f'Predicted class: {predicted_class}')




MODEL_PATH:str = 'app/ml_model/xgboost_model.pkl'
NP_ARRAYS_PATH: str = 'data/numpy_arrays/'
INFERENCE_RESULTS_PATH: str = 'data/inference_results.json'

os.makedirs(NP_ARRAYS_PATH, exist_ok=True)

def load_model():
    try:
        loaded_model:XGBClassifier = joblib.load(MODEL_PATH)
        print("Existing model loaded.")
    except FileNotFoundError:
        print("No existing model found. Please train the model first.")
        return None
    return loaded_model

def save_numpy_array(array: np.ndarray, save_name_npy:str):
    npy_file = os.path.join(NP_ARRAYS_PATH, f"{save_name_npy}")
    np.save(npy_file, array)
    print(f"Numpy array saved to {npy_file}")



def save_inference_result(save_name_npy:str, result: bool):
    if os.path.exists(INFERENCE_RESULTS_PATH):
        with open(INFERENCE_RESULTS_PATH, 'r') as f:
            results_data = json.load(f)
    else:
        results_data = {}

    
    results_data[f"{save_name_npy}"] = result

    with open(INFERENCE_RESULTS_PATH, 'w') as f:
        json.dump(results_data, f, indent=4)
    print(f"Inference result saved to {INFERENCE_RESULTS_PATH}")




def convertToDiff(sample: pd.DataFrame) -> np.ndarray:
    if sample.shape[0] < 600:
        raise ValueError("Insufficient data: Expected at least 600 rows")

    baseline = sample[:300]
    breath = sample[300:600]

    diff_df = baseline.values - breath.values
    # diff_df = diff_df.reshape(1, -1)
    # diff_df=np.vstack([diff_df.flatten()])

    # if diff_df.shape[1] == 300:
    #     return diff_df.reshape(1, -1) 


    return diff_df

def inference_get_result(df: pd.DataFrame,sendInfo:ResultInfoModel,save_name:str):

    file_name_npy=f"{save_name}.npy"

    inference_data = convertToDiff(df)
    
    xgmodel:XGBClassifier = load_model()
    assert xgmodel is not None
    predicted_label = xgmodel.predict(inference_data)

    save_numpy_array(inference_data,file_name_npy)



    TB_prediction:bool= predicted_label[0] == 1

    save_inference_result(file_name_npy,TB_prediction)

    sendInfo.TB_InferenceResult=TB_prediction


    return
    # return predicted_class



def retrain_model():
    get_arrays=[]
    get_lables=[]


    with open(INFERENCE_RESULTS_PATH,'r')as f:
        result_data_json=json.load(f)
   
    for npy_file in os.listdir(NP_ARRAYS_PATH):
        path_npy=os.path.join(NP_ARRAYS_PATH,npy_file)
        data_array=np.load(path_npy)
        get_arrays.append(data_array)

        
        lable=result_data_json.get(path_npy,)

        get_lables.append(1 if lable else 0)
    
    x_train=np.vstack(get_arrays)
    y_train=np.array(get_lables)

    xgb=XGBClassifier()
    xgb.fit(x_train,y_train)

    joblib.dump(xgb,MODEL_PATH)