import pandas as pd
from excel_file import ExcelDataset

def create_dataset(file1, file2):
  correct_data = pd.read_excel(file1)
  wrong_data = pd.read_excel(file2)
  combined_data = pd.concat([correct_data, wrong_data], axis=0)
  dataset = ExcelDataset(combined_data)
  
  return dataset
