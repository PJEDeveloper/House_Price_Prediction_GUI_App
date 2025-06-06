#!/bin/bash

# Navigate to project directory
cd "$(dirname "$0")"

# Install dependencies
pip install -r requirements.txt

echo "Dependencies installed successfully."
