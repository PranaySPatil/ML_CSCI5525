import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.datasets as dsets
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

# configure training config
num_epochs = 10
batch_size = 64
img_size = 784
z_size = 20
lr = 0.001

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Sequential( 
            nn.Linear(img_size, 400),
            nn.ReLU()
        )

        self.mu = nn.Linear(400, z_size)
        self.var = nn.Linear(400, z_size)

    def forward(self, x):
        z = self.fc1(x)
        z_mu = self.mu(z)
        z_var = self.var(z)

        return z_mu, z_var


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Sequential( 
            nn.Linear(z_size, 400),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(400, img_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x


class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        # encode
        z_mu, z_var = self.encoder(x)

        # sample from the distribution having latent parameters z_mu, z_var
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)

        # reparameterize
        z = eps.mul(std).add_(z_mu)

        # decode
        x_dash = self.decoder(z)

        return x_dash, z_mu, z_var

def train_VAE(train_data_loader):
    # initialize
    VAE_net = VAE()
    optimizer = optim.Adam(VAE_net.parameters())
    loss = nn.BCELoss(reduction='sum')
    train_losses = []

    for e in range(num_epochs):
        avg_loss = 0

        for n_batch, (real_batch, _) in enumerate(train_data_loader):
            # reset gradients
            optimizer.zero_grad()

            # reshape image
            images = real_batch.view(-1, 28 * 28)

            # run the network
            denoised_images, z_mu, z_var = VAE_net(images)

            # compute bce loss
            bce_loss = loss(denoised_images, images)

            # compute kl divergence loss
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)

            # total loss
            curr_loss = bce_loss + kl_loss

            # backward pass
            curr_loss.backward()
            avg_loss += curr_loss.item()

            # update the weights
            optimizer.step()

        train_losses.append(avg_loss/len(train_data_loader))
    
    return VAE_net, train_losses

def test_VAE(VAE_net):
    # randomly generate z
    z = torch.randn(16, z_size)

    # run decoder
    reconstructed_img = VAE_net.decoder(z)
    reconstructed_img = reconstructed_img.view(reconstructed_img.size(0), 1, 28, 28)

    horizontal_grid = vutils.make_grid(reconstructed_img, nrow=4, normalize=True, scale_each=True)
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(np.moveaxis(horizontal_grid.detach().numpy(), 0, -1))
    plt.show()

def plot_losses(losses):
    plt.plot(losses, label="loss")
    plt.title("Loss vs EPochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

if __name__ == "__main__":
    train_data = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

    test_data = dsets.MNIST(root='./data', 
                            train=False, 
                            transform=transforms.ToTensor(),
                            download=True)

    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)

    VAE_net, train_losses = train_VAE(train_data_loader)
    torch.save(VAE_net.state_dict(), "hw5_VAE.pth")

    plot_losses(train_losses)
    test_VAE(VAE_net)
