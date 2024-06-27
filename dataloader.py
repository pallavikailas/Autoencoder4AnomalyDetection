import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from excel_file import ExcelDataset

torch.manual_seed(42)

def create_dataloaders(correct_data_path, wrong_data_path, batch_size=64):
    # Read the uploaded Excel files
    correct_data = pd.read_excel(correct_data_path)
    wrong_data = pd.read_excel(wrong_data_path)

    # Concatenate the two datasets
    combined_data = pd.concat([correct_data, wrong_data], axis=0)

    # Create the dataset
    dataset = ExcelDataset(combined_data)

    # Train-Val-Test split
    total_size = len(dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    # Split sequentially
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, total_size))

    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
