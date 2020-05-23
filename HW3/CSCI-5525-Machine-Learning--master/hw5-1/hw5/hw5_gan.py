# -*- coding: utf-8 -*-
"""hw5_gan.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11s8bayKOzP5G6n_k3K06Jsx1ebj04DIL
"""

import pandas as pd
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np

#normalize the data
data = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# total Num batches
num_batches = len(data_loader)

class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 784
        n_out = 1
        
        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, 512),
            nn.LeakyReLU(0.2),
            
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            
        )
        
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.out(x)
        return x

class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 128
        n_out = 784
        
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
       
        
        self.out = nn.Sequential(
            nn.Linear(512, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        #x = self.hidden2(x)
        x = self.out(x)
        return x

discriminator = DiscriminatorNet()#initilization of discriminator and Generator net
generator = GeneratorNet()

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# Loss function
loss = nn.BCELoss()

# Number of epochs
num_epochs = 50

def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data

def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data

def train_discriminator(optimizer, real_data, fake_data):
    # Reset gradients
    optimizer.zero_grad()
    
    #  Train  Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    #  Train  Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()
    
    #  Update weights with gradients
    optimizer.step()
    
    # Return error
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data):
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error

def images_to_vectors(images):
  #convert images to vectors
  return images.view(images.size(0), 784)

def vectors_to_images(vectors):
  #convert  vectors to images
  return vectors.view(vectors.size(0), 1, 28, 28)
def noise(size):
  #genrate random noise
  n = Variable(torch.randn(size, 128))
  return n

generator_error=[];
discriminator_error=[];
fake_images=[]
for epoch in range(num_epochs):
  total_gerror=0;
  total_derror=0;
  counter=0;
  for n_batch, (real_batch,_) in enumerate(data_loader,0):
        counter+=100;
        # Train Discriminator
        real_data = Variable(images_to_vectors(real_batch))
        
        fake_data = generator(noise(real_data.size(0))).detach()
        
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer,
                                                                real_data, fake_data)

        # Train Generator
        # Generate fake data
        fake_data = generator(noise(real_batch.size(0)))
        # Train G
        g_error = train_generator(g_optimizer, fake_data)
        total_gerror+=g_error;
        total_derror+=d_error;
  #loss per epoch
  generator_error.append(total_gerror.item()/n_batch);
  discriminator_error.append(total_derror.item()/n_batch);
  if epoch%10==0:
    #store images with each %10 iteration
    fake_data=generator(noise(16))
    fake_images.append(fake_data);

plt.figure()
plt.plot(discriminator_error,label="Discriminator")
plt.title("Discriminator plot")
plt.xlabel("Epochs")
plt.ylabel("Error")

plt.figure()
plt.plot(generator_error,label="Generator")

plt.title("Generator plot")
plt.xlabel("Epochs")
plt.ylabel("Error")

torch.save(generator, "hw5_gan_gen.pth")
torch.save(discriminator, "hw5_gan_dis.pth")

#plot the figures
for i in fake_images:
  images=vectors_to_images(i)
  horizontal_grid = vutils.make_grid(
              images, normalize=True, scale_each=True,nrow=4)
  fig = plt.figure(figsize=(4, 4))
  plt.imshow(np.moveaxis(horizontal_grid.detach().numpy(), 0, -1))


