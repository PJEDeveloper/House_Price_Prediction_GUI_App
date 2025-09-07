📘 Description  
This project is an end-to-end House Price Prediction System using machine learning. It includes data preprocessing, model training, inference, and a graphical user interface (GUI) for user interaction. It is built in Python using TensorFlow, Scikit-learn, and Tkinter, and structured to work with time-series housing data at the county level.

📊 Overview
1. Data Preprocessing (preprocessing.py)
Transforms raw county-level housing CSV data into a time-series format suitable for machine learning. It also generates metadata and logs all operations.

2. Model Training (house_price_predicting_mlm_v6_FT.py)
Trains deep learning regression models using TensorFlow/Keras for each county (target feature) using the rest as predictors. Includes:

Feature and target scaling

Model architecture definition

Early stopping and learning rate reduction

Model and metrics saving

3. Inference (inference.py)
A command-line interface to perform predictions for a selected county using the trained model. It automatically identifies top predictors and scales inputs accordingly.

4. Graphical User Interface (hpp_gui.py)
Provides a user-friendly Tkinter-based GUI for house price prediction, allowing selection of a county and input of relevant predictors. It dynamically updates input fields based on feature correlation.

## Data Source

This project uses housing market data provided by [Zillow Research](https://www.zillow.com/research/data/).  
Specifically, it relies on metrics such as the **Zillow Home Value Index (ZHVI)**, **Zillow Home Value Forecast (ZHVF)**, and the **Zillow Observed Rent Index (ZORI)**.  

Please note:
- Zillow updates their datasets monthly (typically on the 16th).  
- More information about ZHVI and proper use can be found in the [ZHVI User Guide](https://www.zillow.com/research/zhvi-user-guide/).  
- Zillow occasionally changes CSV download paths.  

🌟 Key Features
✅ General
Modular architecture with centralized configuration via config_utils.

Dynamic metadata and logging for traceability.

Compatible with GPU acceleration (TensorFlow GPU support).

📂 preprocessing.py
Validates and sanitizes column names.

Handles missing values using median imputation.

Converts wide-form time series data into long-form format.

Exports processed data to Excel and metadata to JSON.

Logging of each preprocessing step for debugging and reproducibility.

🧠 house_price_predicting_mlm_v6_FT.py
Automatically detects and trains a model for each county.

Uses a dense neural network with L2 regularization and dropout for generalization.

Stores models, scalers, and evaluation metrics (RMSE, R²).

Implements early stopping and adaptive learning rate scheduling.

🖥️ hpp_gui.py
Full-featured Tkinter GUI for non-technical users.

Automatically shows the most relevant predictor features per selected county.

Provides user input validation and error handling.

Displays prediction results clearly with explanations.

🛠️ inference.py
CLI-based prediction interface for script-based usage.

Auto-selects top 10 predictors based on correlation.

Supports manual input with fallback to default values.

Full logging of prediction steps for transparency.

Setup Instructions

Step 1: Install Dependencies

1.	Navigate to the project root directory:

	bash

	cd /path/to/project_dir

2.	Run the install_dependencies.sh script:

	bash

 	install_dependencies.sh # WSL
	install_dependencies.bat # Windows

3.	For Linux users, ensure Tkinter is installed:

	bash

	sudo apt-get install python3-tk
________________________________________
Step 2: Configure the Project
1.	Review the config/config.json file:
	o	Ensure paths are accurate for your environment (Windows or WSL).
	o	Modify paths as needed to match your directory structure.
________________________________________

Step 3: Run the Application

Preprocess Data
	•	Transform raw data into a processed Excel file and metadata JSON:

	bash

	python3 scripts/preprocessing.py

Train Models
	•	Train models for all counties:

	bash

	python3 scripts/house_price_predicting_mlm_v6_FT.py

Run Predictions
	•	Run the inference script:

	bash

	python3 scripts/inference.py

	•	Input the desired county and receive predictions.

Launch GUI
	•	Start the GUI application:

	bash

	python3 scripts/hpp_gui.py

________________________________________
Requirements
See requirements.txt for a detailed list of dependencies. Key requirements include:
•	Python 3.8+
•	numpy, pandas, tensorflow, joblib, scikit-learn, openpyxl
________________________________________

Testing and Validation
1.	Test the application in both WSL and Windows environments.
2.	Verify the GUI functionality for user inputs and prediction results.
3.	Validate the correctness of logs, outputs, and processed data.
________________________________________

Contributing
1.	Fork the repository and create a new branch.
2.	Ensure code adheres to PEP 8 standards.
3.	Submit a pull request with clear descriptions of changes.
________________________________________
License
This project is open-source and available under the Apache 2.0 License.
________________________________________
Acknowledgments
Special thanks to contributor, Patrick Hill, and resources used in developing this project.


