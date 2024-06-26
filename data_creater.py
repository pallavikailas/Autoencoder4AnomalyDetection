import pandas as pd

# File paths
path_1 = "C:/Users/209500/Desktop/SQLtest/data4.xlsx"  # Correct data points
path_2 = "C:/Users/209500/Desktop/SQLtest/filtered_data.xlsx"    # Wrong data points

# Read each Excel file
correct_data = pd.read_excel(path_1)
wrong_data = pd.read_excel(path_2)

# Concatenate them into a single DataFrame
combined_data = pd.concat([correct_data, wrong_data], axis=0)

# Optionally, save the combined dataset to a new Excel file
combined_path = "C:/Users/209500/Desktop/SQLtest/combined_data.xlsx"
combined_data.to_excel(combined_path, index=False)
