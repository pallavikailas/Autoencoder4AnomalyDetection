import torch
import torch.nn as nn
import torch.nn.functional as f


class VAE(nn.Module):
    def __init__(self, input_size, latent_size=16):
        super(VAE, self).__init__()

        # Encoder layers
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc31 = nn.Linear(32, latent_size)
        self.fc32 = nn.Linear(32, latent_size)

        # Decoder layers
        self.fc4 = nn.Linear(latent_size, 32)
        self.fc5 = nn.Linear(32, 64)
        self.fc6 = nn.Linear(64, 128)
        self.fc7 = nn.Linear(128, input_size)

        # Additional layers for encoding and decoding
        self.fc34 = nn.Linear(latent_size, latent_size // 4)
        self.fc35 = nn.Linear(latent_size // 4, latent_size // 16)
        self.fc42 = nn.Linear(latent_size // 16, latent_size // 4)
        self.fc43 = nn.Linear(latent_size // 4, latent_size)

        # Dropout layer (typically used during training)
        self.dropout = nn.Dropout(p=0.01)

    def encode(self, x):
        h1 = f.elu(self.fc1(x))
        h1 = self.dropout(h1)
        h2 = f.elu(self.fc2(h1))
        h2 = self.dropout(h2)
        h3 = f.elu(self.fc3(h2))
        h3 = self.dropout(h3)
        return self.fc31(h3), self.fc32(h3)

    def encode2(self, x):
        h2 = f.elu(self.fc34(x))
        h2 = self.dropout(h2)
        return self.fc35(h2)

    def decode(self, z):
        h4 = f.elu(self.fc4(z))
        h4 = self.dropout(h4)
        h5 = f.elu(self.fc5(h4))
        h5 = self.dropout(h5)
        h6 = f.elu(self.fc6(h5))
        h6 = self.dropout(h6)
        return torch.sigmoid(self.fc7(h6))

    def decode2(self, z):
        h5 = f.elu(self.fc42(z))
        h5 = self.dropout(h5)
        return self.fc43(h5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        mu, logvar = self.encode(x)
        mu = self.encode2(mu)
        logvar = self.encode2(logvar)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        epsstd = eps * std
        mu = self.decode2(mu)
        logvar = self.decode2(logvar)
        epsstd = self.decode2(epsstd)
        z = mu + epsstd
        recon_x = self.decode(z)
        return recon_x, mu, logvar
