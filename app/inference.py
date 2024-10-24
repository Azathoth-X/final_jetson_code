
# #---------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from .schema import ResultInfoModel
import os
import json



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



def save_inference_result(save_name_npy:str, result: int):
    if os.path.exists(INFERENCE_RESULTS_PATH):
        with open(INFERENCE_RESULTS_PATH, 'r') as f:
            results_data = json.load(f)
    else:
        results_data = {}

    
    results_data[f"{save_name_npy}"] = int(result)

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
    diff_df=diff_df.T


    return diff_df

def inference_get_result(df: pd.DataFrame,sendInfo:ResultInfoModel,save_name:str):

    file_name_npy=f"{save_name}.npy"

    inference_data = convertToDiff(df)
    
    xgmodel:XGBClassifier = load_model()
    assert xgmodel is not None
    predicted_label = xgmodel.predict(inference_data)

    save_numpy_array(inference_data,file_name_npy)



    TB_prediction:int= predicted_label[0]

    save_inference_result(file_name_npy,TB_prediction)

    sendInfo.TB_InferenceResult=TB_prediction==1


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

        
        lable=int(result_data_json.get(npy_file,0))

        get_lables.append(lable)
    
    x_train=np.vstack(get_arrays)
    y_train=np.array(get_lables)

    xgb=XGBClassifier()
    xgb.fit(x_train,y_train)

    joblib.dump(xgb,MODEL_PATH)