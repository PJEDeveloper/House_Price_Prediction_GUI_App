import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from joblib import dump
import tensorflow as tf
import logging
from utils.config_utils import load_config

# Load configuration
config = load_config()

# Paths from config
DATA_DIR = config["data"]["processed"]["excel"]
MODELS_DIR = config["models_dir"]
SCALERS_DIR = config["scalers_dir"]
LOGS_DIR = config["results"]["logs"]
METRICS_DIR = config["results"]["evaluation_metrics"]
TRAINING_LOG = os.path.join(LOGS_DIR, config["training_log"])

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCALERS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(filename=TRAINING_LOG, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# TensorFlow GPU Setup
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        logging.info(f"TensorFlow GPU setup: {len(physical_devices)} GPU(s) available.")
    else:
        logging.info("No GPU detected. Using CPU.")
except Exception as e:
    logging.error(f"TensorFlow GPU initialization error: {e}")

# Load Processed Dataset
data = pd.read_excel(DATA_DIR).select_dtypes(include=[np.number]).dropna()

# Model Creation
def create_model(input_dim, learning_rate=0.001, regularization=1e-4, dropout_rate=0.2):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim, kernel_regularizer=l2(regularization)),
        Dropout(dropout_rate),
        Dense(32, activation='relu', kernel_regularizer=l2(regularization)),
        Dropout(dropout_rate),
        Dense(1, activation='linear')
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mae',
        metrics=['mae', 'mse']
    )
    return model

# Training and Evaluation
def train_and_evaluate(column, data):
    logging.info(f"Processing target: {column}")
    model_path = os.path.join(MODELS_DIR, f"{column}_tensorflow_model.keras")
    feature_scaler_path = os.path.join(SCALERS_DIR, f"{column}_feature_scaler.joblib")
    target_scaler_path = os.path.join(SCALERS_DIR, f"{column}_target_scaler.joblib")
    metrics_path = os.path.join(METRICS_DIR, f"{column}_metrics.json")

    # Skip if already processed
    if all(os.path.exists(path) for path in [model_path, feature_scaler_path, target_scaler_path, metrics_path]):
        logging.info(f"Skipping {column}: Already trained and saved.")
        return

    # Data Scaling
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    X = data.drop(columns=[column]).values
    y = data[column].values.reshape(-1, 1)
    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Model Creation and Training
    model = create_model(input_dim=X_train.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=150,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Evaluation
    y_pred_train = target_scaler.inverse_transform(model.predict(X_train))
    y_pred_test = target_scaler.inverse_transform(model.predict(X_test))
    y_train_orig = target_scaler.inverse_transform(y_train)
    y_test_orig = target_scaler.inverse_transform(y_test)

    metrics = {
        "Train RMSE": np.sqrt(mean_squared_error(y_train_orig, y_pred_train)),
        "Test RMSE": np.sqrt(mean_squared_error(y_test_orig, y_pred_test)),
        "Train R2": r2_score(y_train_orig, y_pred_train),
        "Test R2": r2_score(y_test_orig, y_pred_test),
        "Feature Scaler Min": feature_scaler.min_.tolist(),
        "Feature Scaler Scale": feature_scaler.scale_.tolist(),
        "Target Scaler Min": target_scaler.min_.tolist(),
        "Target Scaler Scale": target_scaler.scale_.tolist(),
    }
    logging.info(f"Metrics for {column}: {metrics}")

    # Save Model, Scalers, and Metrics
    model.save(model_path)
    dump(feature_scaler, feature_scaler_path)
    dump(target_scaler, target_scaler_path)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

# Main Execution
if __name__ == "__main__":
    for column in data.columns:
        train_and_evaluate(column, data)
