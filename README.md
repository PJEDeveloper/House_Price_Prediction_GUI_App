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

