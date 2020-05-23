# -*- coding: utf-8 -*-
"""hw5_dAE.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IVPQVKVfj5V5uQxvzYambAyk4NuWc1cD
"""

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

train_data = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_data = dsets.MNIST(root='./data', 
                            train=False, 
                            transform=transforms.ToTensor(),
                            download=True)

class EncoderDecoderNet(torch.nn.Module):
    """
    A four hidden-layer Encoder Decoder neural network
    """
    def __init__(self):
        super(EncoderDecoderNet, self).__init__()
        n_features = 784
        n_out = 1
        
        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, 400),
            nn.ReLU(),
            
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(400, 20),
            nn.ReLU(),
            
        )
        self.hidden2 = nn.Sequential( 
            nn.Linear(20, 400),
            nn.ReLU(),
            
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(400, n_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        return x

encoderDecoderNet = EncoderDecoderNet()#initialize the network
#batch size
batch_size=64
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=5, shuffle=False)

optimizer = optim.Adam(encoderDecoderNet.parameters())#adam optimizer
# Loss function
loss = nn.BCELoss()

def noise(size):
    #genrate random noise
    n = Variable(torch.randn(size, 1,28,28))
    return n

train_loss_list = []#list collect all the training loss for 
num_epochs = 10
for epoch in range(num_epochs):
    current_loss = 0
    for image,label in train_data_loader:
       
        image_noisy = image+0.1 * noise(image.size(0))#add noise to the image
         # Reset gradients
        optimizer.zero_grad()
        output = encoderDecoderNet(image_noisy.reshape(image_noisy.size(0), -1))
        image1 = image.reshape(image.size(0), -1)
        # Calculate error and backpropagate
        loss_f = loss(output,image1)
        current_loss += loss_f.item()
        loss_f.backward()
        # Update weights with gradients
        optimizer.step()
        
    train_loss_list.append(current_loss/len(train_data_loader))#add total loss
torch.save(encoderDecoderNet.state_dict(),"hw5_dAE.pth")

plt.figure()
plt.plot(train_loss_list,label="loss")
plt.title("loss plot")
plt.xlabel("Epochs")
plt.ylabel("Error")

def vectors_to_images(vectors):
    #convert  vector to images    
    return vectors.view(vectors.size(0), 1, 28, 28)

for image_test,label in test_data_loader:
        image_noisy_test = image_test+0.1 * noise(image_test.size(0))#torch.mul(image+0.25, 0.1 * noise(image.size(0)))

        output = encoderDecoderNet(image_noisy_test.reshape(image_noisy_test.size(0), -1))
        image1 = image.reshape(image.size(0), -1)
        
        reconstructed_img = vectors_to_images(output)
        all_img = torch.cat([image_noisy_test, reconstructed_img], dim=0)
        horizontal_grid = vutils.make_grid(
                  all_img, nrow=5, normalize=True, scale_each=True)
        fig = plt.figure(figsize=(4, 4))
        plt.imshow(np.moveaxis(horizontal_grid.detach().numpy(), 0, -1))
        break
