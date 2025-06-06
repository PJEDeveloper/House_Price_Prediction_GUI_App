import os
import tkinter as tk
from tkinter import ttk, messagebox
from joblib import load
from tensorflow.keras.models import load_model
import numpy as np
from utils.config_utils import load_config

# Load configuration
config = load_config()
DATA_DIR = config["data"]["processed"]["excel"]
MODELS_DIR = config["models_dir"]
SCALERS_DIR = config["scalers_dir"]

# Load processed data
import pandas as pd
data = pd.read_excel(DATA_DIR).select_dtypes(include=[np.number]).dropna()

# GUI Application
class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("House Price Prediction App")
        self.root.geometry("800x600")
        self.root.resizable(False, False)

        # Title Label
        title_label = tk.Label(root, text="House Price Prediction", font=("Helvetica", 20, "bold"))
        title_label.pack(pady=20)

        # Input Section
        self.target_feature = tk.StringVar()
        target_label = tk.Label(root, text="Select County:")
        target_label.pack(pady=5)
        self.target_dropdown = ttk.Combobox(root, textvariable=self.target_feature, state="readonly")
        self.target_dropdown['values'] = data.columns.tolist()
        self.target_dropdown.pack(pady=5)
        self.target_dropdown.bind("<<ComboboxSelected>>", self.load_predictors)

        self.predictors_frame = tk.Frame(root)
        self.predictors_frame.pack(pady=10)

        # Prediction Button
        self.predict_button = tk.Button(root, text="Predict", command=self.make_prediction, state="disabled")
        self.predict_button.pack(pady=20)

        # Output Section
        self.result_label = tk.Label(root, text="", font=("Helvetica", 16))
        self.result_label.pack(pady=10)

        self.info_label = tk.Label(root, text="", wraplength=700, justify="left", font=("Helvetica", 12))
        self.info_label.pack(pady=5)

    def load_predictors(self, event):
        """Load top predictors dynamically based on the selected target feature."""
        for widget in self.predictors_frame.winfo_children():
            widget.destroy()

        target_feature = self.target_feature.get()
        correlation = data.corrwith(data[target_feature]).abs().sort_values(ascending=False)
        self.top_predictors = correlation.index[1:11]

        self.input_fields = {}
        for predictor in self.top_predictors:
            frame = tk.Frame(self.predictors_frame)
            frame.pack(pady=5)

            label = tk.Label(frame, text=f"{predictor} (Default: {data[predictor].iloc[-1]:.2f}):")
            label.pack(side=tk.LEFT, padx=5)

            entry = tk.Entry(frame, width=20)
            entry.insert(0, f"{data[predictor].iloc[-1]:.2f}")
            entry.pack(side=tk.LEFT, padx=5)

            self.input_fields[predictor] = entry

        self.predict_button.config(state="normal")

    def make_prediction(self):
        """Make prediction based on user inputs."""
        target_feature = self.target_feature.get()
        if not target_feature:
            messagebox.showerror("Error", "Please select a county.")
            return

        try:
            model_path = os.path.join(MODELS_DIR, f"{target_feature}_tensorflow_model.keras")
            feature_scaler_path = os.path.join(SCALERS_DIR, f"{target_feature}_feature_scaler.joblib")
            target_scaler_path = os.path.join(SCALERS_DIR, f"{target_feature}_target_scaler.joblib")

            model = load_model(model_path)
            feature_scaler = load(feature_scaler_path)
            target_scaler = load(target_scaler_path)

            user_inputs = {}
            for predictor, entry in self.input_fields.items():
                user_inputs[predictor] = float(entry.get())

            input_vector = self.prepare_input_vector(user_inputs, target_feature, feature_scaler)
            scaled_prediction = model.predict(input_vector)
            prediction = target_scaler.inverse_transform(scaled_prediction)[0][0]

            self.result_label.config(text=f"Predicted Price: ${prediction:.2f}")
            self.info_label.config(text="The predicted price is based on the most relevant features for this county.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during prediction: {e}")

    def prepare_input_vector(self, user_inputs, target_feature, feature_scaler):
        """Prepare scaled input vector."""
        all_features = data.drop(columns=[target_feature]).iloc[-1].to_dict()
        for predictor, value in user_inputs.items():
            all_features[predictor] = value
        input_vector = np.array([all_features[feature] for feature in data.columns if feature != target_feature]).reshape(1, -1)
        return feature_scaler.transform(input_vector)

# Main Execution
if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()
