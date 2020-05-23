import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import torchvision.datasets as dsets
import torchvision.utils as vutils

# configure training config
num_epochs = 10
batch_size = 64
img_size = 784
z_size = 20
lr = 0.0002

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Sequential( 
            nn.Linear(img_size, 400),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(400, z_size),
            nn.ReLU()   
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        return x

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

class dAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        x_dash = self.decoder(z)

        return x_dash

def train_dAE(train_data_loader):
    # initialize
    dAE_net = dAE()
    loss = nn.BCELoss()
    optimizer = optim.Adam(dAE_net.parameters())
    train_losses = []
    
    for epoch in range(num_epochs):
        avg_loss = 0

        for image, _ in train_data_loader:
            # reset gradients
            optimizer.zero_grad()

            # initialize noisy image
            noise = Variable(torch.randn(image.size(0), 1,28,28))
            image_noisy = image+0.1 * noise
            
            # run the network
            output = dAE_net(image_noisy.reshape(image_noisy.size(0), -1))

            # reformate original
            reshaped_image = image.reshape(image.size(0), -1)

            # compute loss
            curr_loss = loss(output, reshaped_image)
            avg_loss += curr_loss.item()

            # back propogation
            curr_loss.backward()

            # Update weights with gradients
            optimizer.step()
            
        train_losses.append(avg_loss/len(train_data_loader))

    return dAE_net, train_losses

def plot_losses(losses):
    plt.plot(losses, label="loss")
    plt.title("Loss vs EPochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

def test_dAE(dAE_net, test_data_loader):
    for img, _ in test_data_loader:
        # initialize noisy image
        noise = Variable(torch.randn(img.size(0), 1,28,28))
        img = img + 0.1 * noise

        # run network
        output = dAE_net(img.reshape(img.size(0), -1))
        reconstructed_img = output.view(output.size(0), 1, 28, 28)
        
        # create image grid
        concated_images = torch.cat([img, reconstructed_img], dim=0)
        img_grid = vutils.make_grid(concated_images, nrow=5, normalize=True, scale_each=True)
        fig = plt.figure(figsize=(4, 4))
        plt.imshow(np.moveaxis(img_grid.detach().numpy(), 0, -1))
        plt.show()
        
        break

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
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=5, shuffle=False)

    dAE_net, train_losses = train_dAE(train_data_loader)
    torch.save(dAE_net.state_dict(), "hw5_dAE.pth")

    plot_losses(train_losses)

    test_dAE(dAE_net, test_data_loader)

