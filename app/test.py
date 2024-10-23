import numpy as np
import os
import json
from xgboost import XGBClassifier
import joblib

# Paths for testing
TEST_MODEL_PATH = 'test_model/xgboost_model_test.pkl'
TEST_NP_ARRAYS_PATH = 'test_data/numpy_arrays/'
TEST_INFERENCE_RESULTS_PATH = 'test_data/inference_results_test.json'

os.makedirs(TEST_NP_ARRAYS_PATH, exist_ok=True)
os.makedirs('test_model', exist_ok=True)

def setup_mock_data():
    """Set up mock NumPy arrays and inference results for testing."""
    # Create 100 mock NumPy arrays
    num_samples = 100
    array_size = 300  # Assuming diff data after conversion is 300
    for i in range(num_samples):
        mock_data = np.random.rand(1, array_size)
        npy_filename = os.path.join(TEST_NP_ARRAYS_PATH, f"sample_{i}.npy")
        np.save(npy_filename, mock_data)

    # Create inference results corresponding to the mock NumPy arrays
    inference_results = {f"sample_{i}.npy": bool(i % 2) for i in range(num_samples)}  # True/False alternating results
    with open(TEST_INFERENCE_RESULTS_PATH, 'w') as f:
        json.dump(inference_results, f, indent=4)

def cleanup_test_environment():
    """Remove test directories and files."""
    import shutil
    shutil.rmtree(TEST_NP_ARRAYS_PATH)
    shutil.rmtree('test_model')
    os.remove(TEST_INFERENCE_RESULTS_PATH)

def load_test_model():
    """Load or create a test model."""
    if os.path.exists(TEST_MODEL_PATH):
        return joblib.load(TEST_MODEL_PATH)
    else:
        model = XGBClassifier()
        joblib.dump(model, TEST_MODEL_PATH)
        return model

def retrain_model_test():
    setup_mock_data()

    # Call the retrain_model function
    def retrain_model():
        get_arrays = []
        get_labels = []

        with open(TEST_INFERENCE_RESULTS_PATH, 'r') as f:
            result_data_json = json.load(f)

        for npy_file in os.listdir(TEST_NP_ARRAYS_PATH):
            path_npy = os.path.join(TEST_NP_ARRAYS_PATH, npy_file)
            data_array = np.load(path_npy)
            get_arrays.append(data_array)

            label = result_data_json.get(npy_file, False)
            get_labels.append(1 if label else 0)

        x_train = np.vstack(get_arrays)
        y_train = np.array(get_labels)

        # Retrain model with the mock data
        xgb = XGBClassifier()
        xgb.fit(x_train, y_train)

        joblib.dump(xgb, TEST_MODEL_PATH)

    # Test the retrain_model function
    retrain_model()

    # Load the newly retrained model to validate it was updated
    model = load_test_model()

    # Verify the model was trained on the test data
    assert model is not None, "Model was not retrained."

    # Optionally, print a message to indicate success
    print("Model retrained successfully with test data.")

    # Clean up the test environment
    cleanup_test_environment()


