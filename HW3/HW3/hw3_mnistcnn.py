import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import timeit, datetime
import numpy as np

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3, 1, 1)
        self.fc1 = nn.Linear(14*14*20, 128, True)
        self.fc2 = nn.Linear(128, 10, False)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.softmax(self.fc2(x), dim=1)
        return x

def load_data(batch_size):
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    trainset = datasets.MNIST(root="./data", train=True, download=True,transform=transform)
    testset = datasets.MNIST(root="./data", train=False, download=True,transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    classes = list(range(10))

    return train_loader, test_loader

def train_epoch(train_loader, net, optimizer):
    net.train()
    correct = 0
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        epoch_loss += loss.item()

    epoch_accuracy = correct/len(train_loader.dataset)
    epoch_loss /= len(train_loader.dataset)

    return epoch_accuracy, epoch_loss

def test(test_loader, net):
    with torch.no_grad():
        net.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            output = net(data)
            # sum up batch loss
            test_loss += F.cross_entropy(output, target).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        acc = 100*correct/len(test_loader.dataset)
        print('Average loss: '+str(test_loss))
        print('Accuracy '+str(acc)+'%')

def plot_graph(accuracies, losses):
    plt.plot(list(range(0, len(accuracies))), accuracies, label='Accuracies')
    plt.xlabel('#epoch')
    plt.ylabel('Accuracy')
    plt.show()
    plt.plot(list(range(0, len(losses))), losses, label='Losses')
    plt.xlabel('#epoch')
    plt.ylabel('Loss')
    plt.show()

def plot_graph_2(accuracies, losses, optimizer):
    plt.plot(list(range(0, len(accuracies))), accuracies, label='Accuracies')
    plt.xlabel('#epoch')
    plt.ylabel('Accuracy')
    plt.title('Epoch vs Accuracy-'+optimizer)
    plt.show()
    plt.plot(list(range(0, len(losses))), losses, label='Losses')
    plt.xlabel('#epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss-'+optimizer)
    plt.show()

def train_SGD():
    net = Net()
    train_loader, test_loader = load_data(32)
    epochs = 30
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    accuracies = []
    losses = []

    for i in range(epochs):
        accuracy, loss = train_epoch(train_loader, net, optimizer)
        print("Epochs: "+str(i+1))
        if i>0 and abs(loss-losses[-1])<0.00005:
            break

        accuracies.append(accuracy)
        losses.append(loss)

    plot_graph(accuracies, losses)
    torch.save(net, "mnist-cnn")
    # test(test_loader, net)
    print('Finished Training')

def test_SGD():
    train_loader, test_loader = load_data(32)
    net = torch.load('mnist-cnn')
    test(test_loader, net)

def get_runtime_over_batch_size():
    batch_sizes = [32, 64, 96, 128]
    epochs = 30
    runtimes = []

    for batch_size in batch_sizes:
        net = Net()
        losses = []
        optimizer = optim.SGD(net.parameters(), lr=0.1)
        train_loader, test_loader = load_data(batch_size)
        start = timeit.default_timer()
        start = datetime.datetime.now()
        for i in range(epochs):
            accuracy, loss = train_epoch(train_loader, net, optimizer)
            print("Epochs: "+str(i+1))
            if i>0 and abs(loss-losses[-1])<0.00005:
                break

            losses.append(loss)

        stop = timeit.default_timer()
        stop = datetime.datetime.now()
        runtimes.append((stop - start).seconds)
    
    x_pos = np.arange(len(batch_sizes))

    plt.bar(x_pos, runtimes, align='center', alpha=0.5)
    plt.xticks(x_pos, batch_sizes)
    plt.ylabel('Runtime in seconds')
    plt.xlabel('Batch size')
    plt.title('Batch Size vs Runtime')
    plt.show()

def train_3_optimizers():
    networks = [
        Net(),
        Net(),
        Net()
    ]
    optimizers = [ 
        optim.SGD(networks[0].parameters(), lr=0.001),
        optim.Adam(networks[1].parameters(), lr=0.001),
        optim.Adagrad(networks[2].parameters(), lr=0.001)
    ]
    optimizers_labels = ["SGD", "Adam", "AdaGrad"]
    train_loader, test_loader = load_data(32)
    epochs = 30
    
    for o in range(len(optimizers)):
        optimizer = optimizers[o]
        accuracies = []
        losses = []
        for i in range(epochs):
            accuracy, loss = train_epoch(train_loader, networks[o], optimizer)
            print("Epochs: "+str(i+1))
            if i>0 and abs(loss-losses[-1])<0.00005:
                break

            accuracies.append(accuracy)
            losses.append(loss)

        plot_graph_2(accuracies, losses, optimizers_labels[o])
        print('Finished Training for '+optimizers_labels[o])


if __name__ == "__main__":
    # train_SGD()
    test_SGD()
    # get_runtime_over_batch_size()
    # train_3_optimizers()