import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

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

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128, True)
        self.fc2 = nn.Linear(128, 10, False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

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

    epoch_accuracy = 100*correct/len(train_loader.dataset)
    epoch_loss /= len(train_loader.dataset)
    
    return epoch_accuracy, epoch_loss

def test_model(test_loader, net):
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
    print(accuracies)
    print(losses)
    plt.plot(list(range(0, len(accuracies))), accuracies, label='Accuracies')
    plt.xlabel('#epoch')
    plt.ylabel('Accuracy')
    plt.show()
    plt.plot(list(range(0, len(losses))), losses, label='Losses')
    plt.xlabel('#epoch')
    plt.ylabel('Loss')
    plt.show()

def train():
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    train_loader, test_loader = load_data(32)
    accuracies = []
    losses = []
    epochs = 30

    for i in range(epochs):
        accuracy, loss = train_epoch(train_loader, net, optimizer)
        print("Epochs: "+str(i+1))
        if i>0 and abs(loss-losses[-1])<0.00005:
            break

        accuracies.append(accuracy)
        losses.append(loss)

    plot_graph(accuracies, losses)
    torch.save(net, "mnist-fc")
    # test_model(test_loader, net)
    print('Finished Training')

def test():
    net = torch.load("mnist-fc")
    train_loader, test_loader = load_data(32)

    test_model(test_loader, net)

if __name__ == "__main__":
    # train()
    test()