import pandas as pd

# Load the Excel file
excel_file = "updated_medicines.xlsx"  # Replace with your actual file path
df = pd.read_excel(excel_file)

# Define the output CSV file path
csv_file = "medicines.csv"  # Replace with your desired output file name

# Convert to CSV
df.to_csv(csv_file, index=False)

# Print the output file path
print(f"File saved as: {csv_file}")
