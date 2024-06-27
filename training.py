import torch
import torch.nn as nn
import torch.optim as optim
from model import VAE
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataloader import create_dataloaders
from hyperparameter import num_epochs, learning_rate, multiplier, patience

def train_model(train_dataloader, val_dataloader):
    input_size = len(train_dataloader.dataset[0])
    autoencoder = VAE(input_size)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(autoencoder.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

    # Early stopping parameters
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        autoencoder.train()
        train_loss = 0.0
        kl_loss_total = 0.0

        for batch_idx, data in enumerate(train_dataloader):
            inputs = data
            optimizer.zero_grad()
            recon_batch, mu, logvar = autoencoder(inputs)
            reconstruction_loss = criterion(recon_batch, inputs)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction_loss + kl_loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            kl_loss_total += kl_loss.item()

        train_loss /= len(train_dataloader.dataset)
        kl_loss_total /= len(train_dataloader.dataset)

        # Validation loss
        autoencoder.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_idx, data in enumerate(val_dataloader):
                inputs = data
                recon_batch, mu, logvar = autoencoder(inputs)
                reconstruction_loss = criterion(recon_batch, inputs)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                val_loss += (reconstruction_loss + kl_loss).item()

        val_loss /= len(val_dataloader.dataset)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], '
            f'Train Loss: {train_loss:.4f}, '
            f'KL Loss: {kl_loss_total:.4f}, '
            f'Val Loss: {val_loss:.4f}'
        )

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping. No improvement in validation loss for {patience} epochs.')
                break

        scheduler.step()

    return autoencoder
