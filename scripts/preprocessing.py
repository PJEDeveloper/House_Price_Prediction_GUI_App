import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from utils.config_utils import load_config
import re

# Load configuration
config = load_config()

# Paths from config
INPUT_CSV_PATH = config["data"]["raw"]["csv"]
OUTPUT_EXCEL_PATH = config["data"]["processed"]["excel"]
LOG_FILE_PATH = os.path.join(config["results"]["logs"], config["preprocessing_log"])
METADATA_DIR = config["metadata_dir"]

# Function to sanitize column names to match naming convention
def sanitize_column_name(name):
    return re.sub(r'[^\w]', '_', name)  # Replace non-alphanumeric characters with underscores

# Preprocessing function
def preprocess_county_data(input_csv_path, output_excel_path, log_file_path, metadata_dir):
    try:
        # Step 1: Load the data
        raw_data = pd.read_csv(input_csv_path)
        log_entries = ["Step 1: Data loaded successfully."]

        # Step 2: Validate the input structure
        required_columns = ['RegionID', 'RegionName', 'StateName', 'State'] + [col for col in raw_data.columns if col.startswith('20')]
        if not all(col in raw_data.columns for col in required_columns):
            raise ValueError("Input data is missing one or more required columns.")
        log_entries.append(f"Step 2: Validation passed. Required columns found.")

        # Step 3: Handle missing values (impute using median)
        time_series_columns = [col for col in raw_data.columns if col.startswith('20')]
        raw_data[time_series_columns] = raw_data[time_series_columns].apply(lambda x: x.fillna(x.median()), axis=0)
        log_entries.append(f"Step 3: Missing values imputed using the median for time-series columns.")

        # Step 4: Pivot data for time-series transformation
        pivot_data = raw_data.melt(id_vars=['RegionName', 'State'], value_vars=time_series_columns,
                                   var_name='Date', value_name='Value')
        pivot_data['County_State'] = pivot_data['RegionName'] + ', ' + pivot_data['State']

        # Sanitize column names to match naming convention
        pivot_data['County_State'] = pivot_data['County_State'].apply(sanitize_column_name)
        pivot_data = pivot_data.pivot(index='Date', columns='County_State', values='Value')
        pivot_data.reset_index(inplace=True)
        log_entries.append(f"Step 4: Data pivoted successfully with time-series transformation. Column names sanitized.")

        # Step 5: Save metadata separately
        metadata = raw_data[['RegionID', 'RegionName', 'StateName', 'State']].drop_duplicates()

        # Sanitize metadata column for consistency
        metadata['County_State'] = metadata['RegionName'] + ', ' + metadata['State']
        metadata['County_State'] = metadata['County_State'].apply(sanitize_column_name)
        log_entries.append(f"Step 5: Metadata extracted and formatted. Column names sanitized.")

        # Save metadata as a JSON file
        os.makedirs(metadata_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metadata_path = os.path.join(metadata_dir, f"metadata_{timestamp}.json")
        metadata.to_json(metadata_path, orient='records', indent=4)
        log_entries.append(f"Step 5.1: Metadata saved to JSON at {metadata_path}.")

        # Step 6: Save processed data to Excel
        with pd.ExcelWriter(output_excel_path) as writer:
            pivot_data.to_excel(writer, index=False, sheet_name='Time_Series_Data')
            metadata.to_excel(writer, index=False, sheet_name='Metadata')
        log_entries.append(f"Step 6: Processed data saved to Excel successfully at {output_excel_path}.")

        # Step 7: Save log file
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(log_file_path, 'w') as log_file:
            log_file.write("\n".join(log_entries))

        print("Preprocessing completed successfully. Logs written to:", log_file_path)
    except Exception as e:
        print(f"An error occurred: {e}")
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(log_file_path, 'w') as log_file:
            log_file.write(f"An error occurred: {e}\n")

# Run the preprocessing
if __name__ == "__main__":
    preprocess_county_data(INPUT_CSV_PATH, OUTPUT_EXCEL_PATH, LOG_FILE_PATH, METADATA_DIR)
