import pandas as pd

# File paths
old_file_path = 'C:/Users/abdel/OneDrive/Desktop/Momken/Bills/Labels/labels.csv'
new_file_path = 'C:/Users/abdel/OneDrive/Desktop/Momken/Bills/Labels/only ribbons.csv'

# Load the CSV file without headers
labels_df = pd.read_csv(old_file_path, header=None)

# Filter the rows where the second column (index 1) is 1
filtered_df = labels_df[labels_df[1] == 1]

# Drop the second column (index 1)
filtered_df = filtered_df.drop(columns=[1])

# Save the filtered DataFrame to the new file
filtered_df.to_csv(new_file_path, index=False, header=False)

print(f"Filtered CSV saved to: {new_file_path}")
