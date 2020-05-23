# -*- coding: utf-8 -*-
"""hw3mnistcnn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vJDQ5Ge6cfWdjJDSTq52YZAzOkbue-jW
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import datetime
import torch.nn.functional as F

train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())
time_list=[]

batch_size = 32

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=20,kernel_size=3,stride=1)
        self.fc1 = nn.Linear(20*13*13, 128)
        self.fc2 = nn.Linear(128, 10)
        

    def forward(self, x):
      x=self.conv1(x);
      x=F.max_pool2d(x,2);
      x=F.relu(x);
      x = x.view(x.size(0), -1)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return x#F.log_softmax(x)

model = Net()

criterion = nn.CrossEntropyLoss()

learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
#optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

losslist=[]
traing_accuracy_list=[]
oldloss=100;
newloss=50;
a = datetime.datetime.now()
#for epoch in range(num_epochs):
while oldloss-newloss>=0.0001:
    total = 0
    correct=0
    total_loss=0
    for i, (images, labels) in enumerate(train_loader):
        # Load images
        images = images.requires_grad_()

        # Clear gradients 
        optimizer.zero_grad()

        # Forward pass 
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        # Calculate Loss:
        loss = criterion(outputs, labels)

        # Getting gradients
        loss.backward()

        # Updating parameters
        optimizer.step()

        total_loss+=loss;

        correct += (predicted == labels).sum()
        total += labels.size(0)
    traing_accuracy_list.append(100 * correct.item() / total)
    losslist.append(100*total_loss.item()/total)
    oldloss=newloss;
    newloss=total_loss.item()/total
    print(oldloss-newloss)
    
b = datetime.datetime.now()
time_list.append((b-a).seconds)

import matplotlib.pyplot as plt
plt.plot(losslist)
plt.title("SGD loss vs epochs")
plt.ylabel("loss%")
plt.xlabel("epochs")

plt.plot(traing_accuracy_list)
plt.title("SGD Accuracy vs epochs")
plt.ylabel("Accuracy %")
plt.xlabel("epochs")

losslist

torch.save(model, "mnist-cnn")
model = Net()
model = torch.load( "mnist-cnn")

# Calculate Accuracy         
correct = 0
total = 0

for images, labels in test_loader:
                # Load images
  images = images.requires_grad_()

                # Forward pass 
  outputs = model(images)

                # Get predictions
  _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
  total += labels.size(0)

                # Total correct predictions
  correct += (predicted == labels).sum()

  accuracy = 100 * correct / total
print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))

batch_list=[32,64,96,128]
batch_list=[32,64,96,128]
plt.plot(epoch_list,time_list)
plt.title("time vs batchsize")
plt.xlabel("Batch Size")
plt.ylabel("Time(in sec)")

time_list
