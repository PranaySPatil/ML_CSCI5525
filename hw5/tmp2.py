import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np

# configure training config
num_epochs = 50
batch_size = 100
img_size = 784
dis_output_size = 1
z_size = 128
lr = 0.0002
loss = nn.BCELoss()

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        # define network
        self.fc1 = nn.Sequential( 
            nn.Linear(img_size, 512),
            nn.LeakyReLU(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, dis_output_size),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        # define network
        self.fc1 = nn.Sequential(
            nn.Linear(z_size, 256),
            nn.LeakyReLU(0.2)
        )
        self.fc2 = nn.Sequential(            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, img_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

def plot_losses(gen_loss, dis_losses):
    # plot losses vs epochs
    plt.plot(range(len(gen_loss)+1)[1:], gen_loss, label="Generator loss")
    plt.plot(range(len(gen_loss)+1)[1:], dis_losses, label="Discriminator loss")
    plt.title("Loss vs Epochs")
    plt.xlabel('Loss')
    plt.ylabel('Epochs')
    plt.legend()
    plt.show()

def plot_generated_images(images_gen):
    # plot the figures
    for imgs in images_gen:
        horizontal_grid = vutils.make_grid(imgs, normalize=True, scale_each=True,nrow=4)
        fig = plt.figure(figsize=(4, 4))
        plt.imshow(np.moveaxis(horizontal_grid.detach().numpy(), 0, -1))
        plt.show()


def train_discriminator(discriminator, optimizer, real_data, fake_data, noise_decay):
    # reset gradients
    optimizer.zero_grad()
    
    # initializing real data with noise * decay
    noise1 = noise_decay * torch.distributions.Uniform(0,1).sample_n(real_data.size(0)*img_size).view(real_data.size(0),img_size)
    noise2 = noise_decay * torch.distributions.Uniform(0,1).sample_n(real_data.size(0)*img_size).view(real_data.size(0),img_size)

    prediction_real = discriminator(real_data+noise1)

    # calculate error and backpropagate, using label smoothing
    error_real = loss(prediction_real, Variable(torch.ones(real_data.size(0), 1)*0.9))
    error_real.backward()

    # initializing fake data with noise * decay
    prediction_fake = discriminator(fake_data+noise2)

    # calculate error and backpropagate
    error_fake = loss(prediction_fake, Variable(torch.zeros(real_data.size(0), 1)))
    error_fake.backward()
    
    # update weights
    optimizer.step()
    
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(discriminator, optimizer, fake_data, noise_decay):
    # reset gradients
    optimizer.zero_grad()

    # initializing fake data with noise * decay
    noise = noise_decay * torch.distributions.Uniform(0,1).sample_n(fake_data.size(0)*img_size).view(fake_data.size(0),img_size)
    prediction = discriminator(fake_data+noise)

    # calculate error and backpropagate, using label smoothing
    error = loss(prediction, torch.ones(fake_data.size(0), 1)*0.9)
    error.backward()

    # update weights
    optimizer.step()
    
    return error

def train_GAN(train_loader):
    # initialization
    dis_losses = []
    gen_losses = []
    images_gen = []
    discriminator = Discriminator()
    generator = Generator()

    dis_optimizer = optim.Adam(discriminator.parameters(), lr)
    gen_optimizer = optim.Adam(generator.parameters(), lr)


    for epoch in range(num_epochs):
        avg_gen_loss = 0
        avg_dis_loss = 0
        noise_decay = 0

        for n_batch, (real_batch,_) in enumerate(train_loader, 0):
            # decaying noise
            noise_decay = 0.1*noise_decay

            N = real_batch.size(0)

            # vecorize image
            real_data = Variable(real_batch.view(N, img_size))

            # generate fake data
            fake_data = generator(Variable(torch.randn(N, 128))).detach()

            # train Discriminator
            dis_loss, d_pred_real, d_pred_fake = train_discriminator(discriminator, dis_optimizer, real_data, fake_data, noise_decay)

            # generate fake data, with random z
            fake_data = generator(Variable(torch.randn(N, 128)))

            # train Generator
            gen_loss = train_generator(discriminator, gen_optimizer, fake_data, noise_decay)

            avg_gen_loss += gen_loss.item()
            avg_dis_loss += dis_loss.item()

        gen_losses.append(avg_gen_loss)
        dis_losses.append(dis_loss)

        if (epoch+1) % 10 == 0:
            # generate images using generator
            fake_data = generator(Variable(torch.randn(16, 128)))
            images_gen.append(fake_data.view(fake_data.size(0), 1, 28, 28))

    return generator, discriminator, gen_losses, dis_losses, images_gen 

if __name__ == "__main__":
    data = dsets.MNIST(root='./data', 
                                train=True, 
                                transform=transforms.ToTensor(),
                                download=True)
    train_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)

    generator, discriminator, gen_losses, dis_losses, images_gen  = train_GAN(train_loader)
    torch.save(generator, "hw5_gan_gen.pth")
    torch.save(discriminator, "hw5_gan_dis.pth")

    plot_losses(gen_losses, dis_losses)
    plot_generated_images(images_gen)