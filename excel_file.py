import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class ExcelDataset(Dataset):
    def __init__(self, excel_file):
        self.data1 = pd.read_excel(excel_file)
        self.label_encoders = {}
        self.data2 = pd.get_dummies(self.data1, columns=[col for col in self.data1.columns
                                                         if self.data1[col].dtype == 'object'])
        self.data = self.data2.astype(np.float32)
        self.columns = self.data.columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx].values.astype(np.float32)

    def inverse_transform(self, encoded_row):
        decoded_row = {}
        for i, value in enumerate(encoded_row):
            column_name = self.columns[i]
            decoded_row[column_name] = value
        return decoded_row
