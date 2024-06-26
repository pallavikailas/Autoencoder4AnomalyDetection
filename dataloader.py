from torch.utils.data import DataLoader
from excel_file import ExcelDataset
import torch

torch.manual_seed(42)

# File paths
dataset = "C:/Users/209500/Desktop/SQLtest/combined_data.xlsx"
dataset = ExcelDataset(dataset)

# Determine sizes for train, validation, and test sets
total_size = len(dataset)
train_size = int(0.6 * total_size)  # 60% for training
val_size = int(0.2 * total_size)    # 20% for validation
test_size = total_size - train_size - val_size  # Remaining for testing

# Split sequentially
train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, total_size))

# DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
