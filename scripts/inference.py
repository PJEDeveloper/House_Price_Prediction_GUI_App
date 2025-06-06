import os
import numpy as np
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model
import logging
from utils.config_utils import load_config

# Load configuration
config = load_config()

# Extract paths from the configuration
DATA_DIR = config["data"]["processed"]["excel"]
MODELS_DIR = config["models_dir"]
SCALERS_DIR = config["scalers_dir"]
LOGS_DIR = config["results"]["logs"]
INFERENCE_LOG = os.path.join(LOGS_DIR, config["inference_log"])

# Setup logging
os.makedirs(LOGS_DIR, exist_ok=True)  # Ensure the logs directory exists
logging.basicConfig(filename=INFERENCE_LOG, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load Processed Dataset
data = pd.read_excel(DATA_DIR).select_dtypes(include=[np.number]).dropna()

# Prepare Input Vector
def prepare_input_vector(predictors, user_inputs, data, feature_scaler, target_feature):
    all_features = data.drop(columns=[target_feature]).iloc[-1].to_dict()
    for predictor, value in user_inputs.items():
        all_features[predictor] = value
    input_vector = np.array([all_features[feature] for feature in data.columns if feature != target_feature]).reshape(1, -1)
    scaled_input = feature_scaler.transform(input_vector)
    return scaled_input

# Predict House Price
def predict_house_price(target_feature):
    try:
        model_path = os.path.join(MODELS_DIR, f"{target_feature}_tensorflow_model.keras")
        feature_scaler_path = os.path.join(SCALERS_DIR, f"{target_feature}_feature_scaler.joblib")
        target_scaler_path = os.path.join(SCALERS_DIR, f"{target_feature}_target_scaler.joblib")

        model = load_model(model_path)
        feature_scaler = load(feature_scaler_path)
        target_scaler = load(target_scaler_path)

        correlation = data.corrwith(data[target_feature]).abs().sort_values(ascending=False)
        top_predictors = correlation.index[1:11]
        logging.info(f"Top predictors for {target_feature}: {list(top_predictors)}")

        user_inputs = {}
        for predictor in top_predictors:
            default_value = data[predictor].iloc[-1]
            user_input = input(f"{predictor} (default: {default_value:.2f}): ").strip()
            user_inputs[predictor] = float(user_input) if user_input else default_value

        scaled_input = prepare_input_vector(top_predictors, user_inputs, data, feature_scaler, target_feature)
        logging.info(f"Scaled Input Vector: {scaled_input}")

        scaled_prediction = model.predict(scaled_input)
        prediction = target_scaler.inverse_transform(scaled_prediction)[0][0]
        logging.info(f"Prediction for {target_feature}: {prediction}")
        print(f"Prediction for {target_feature}: {prediction:.2f}")
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        print(f"Error during inference: {e}")

# Main Execution
if __name__ == "__main__":
    target_feature = input("Enter the target feature (e.g., 'Yuma County, AZ'): ").strip()
    predict_house_price(target_feature)
