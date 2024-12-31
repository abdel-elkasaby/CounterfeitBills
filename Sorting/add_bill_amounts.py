import json
import re
import csv
from pathlib import Path

import json
import os

# Function to load configuration
def load_config():
    """Load the configuration JSON."""
    scripts_path = os.path.abspath(os.path.dirname(__file__))
    json_file = os.path.join(scripts_path, "..", "config.json")
    with open(json_file, "r") as config_file:
        return json.load(config_file)

def validate_file(file_path):
    """Ensure the specified file exists."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

# Load configuration
config = load_config()

# Extract paths from configuration
input_csv = config["labels_csv"]
output_csv = config["output_csv"]

# Validate input file
validate_file(input_csv)


# Function to extract the bill amount from the file path
def extract_amount(file_path):
    match = re.search(r'i(\d+)[a-zA-Z\d]*', Path(file_path).name)
    return int(match.group(1)) if match else None

# Read the input CSV, process each row, and write to the output CSV
with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
    csvreader = csv.reader(infile)
    csvwriter = csv.writer(outfile)

    for row in csvreader:
        if not row:  # Skip empty rows
            continue
        file_path, ribbon = row  # Unpack the current row
        amount = extract_amount(file_path)  # Extract the bill amount
        csvwriter.writerow([file_path, ribbon, amount])  # Write the updated row

print(f"Processed file saved to: {output_csv}")
