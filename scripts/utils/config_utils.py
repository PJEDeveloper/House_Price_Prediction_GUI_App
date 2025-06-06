import os
import json

def load_config() -> dict:
    """
    Load the configuration file from the project directory.
    Returns:
        dict: Parsed configuration data.
    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    # Define the path to the config file (relative to the project root)
    config_path = r"/mnt/d/house_price_prediction/config/config.json"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, "r") as f:
        return json.load(f)
