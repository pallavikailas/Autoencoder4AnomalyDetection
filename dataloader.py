import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from data_creater import create_dataset
from hyperparameter import batch_size

torch.manual_seed(42)

def create_dataloaders(correct_data_path, wrong_data_path, batch_size=batch_size):
    dataset = create_dataset(correct_data_path, wrong_data_path)
    
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
