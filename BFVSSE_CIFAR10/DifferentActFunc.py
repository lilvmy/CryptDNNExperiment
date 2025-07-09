import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import logging as log


# python verison lower than 3.9


device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.ToTensor()

train_set = torchvision.datasets.CIFAR10(
    root = './data',
    train=True,
    download=True,
    transform=transform
)

test_set = torchvision.datasets.CIFAR10(
    root = './data',
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=50,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=50,
    shuffle=True
)


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def train_net(network, epochs, device):
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    for epoch in range(epochs):

        total_loss = 0
        total_correct = 0

        for batch in train_loader:  # Get Batch
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            preds = network(images)  # Pass Batch
            loss = F.cross_entropy(preds, labels)  # Calculate Loss

            optimizer.zero_grad()
            loss.backward()  # Calculate Gradients
            optimizer.step()  # Update Weights

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)


def test_net(network, device):
    network.eval()
    total_loss = 0
    total_correct = 0

    with torch.no_grad():
        for batch in test_loader:  # Get Batch
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            preds = network(images)  # Pass Batch
            loss = F.cross_entropy(preds, labels)  # Calculate Loss

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)

        accuracy = round(100. * (total_correct / len(test_loader.dataset)), 4)

    return total_correct / len(test_loader.dataset)


experiments = 10

# Initial LeNet-1: two ReLU
accuracies = []
for i in range(0, experiments):
    for j in range(0, experiments):
        CIFAR10_ReLU2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),

            nn.Flatten(),

            nn.Linear(576, 128),
            nn.Linear(128, 10),
        )

        CIFAR10_ReLU2.to(device)
        train_net(CIFAR10_ReLU2, 10, device)
        acc = test_net(CIFAR10_ReLU2, device)
        accuracies.append(acc)

    m = np.array(accuracies)
    print(f"Mean accuracy of {i+1} epoch under two ReLU on test set: {np.mean(m)}")

# Initial LeNet-1: two Tanh
accuracies = []
for i in range(0,experiments):
    for j in range(0, experiments):
        CIFAR10_Tanh2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),

            nn.Flatten(),

            nn.Linear(576, 128),
            nn.Linear(128, 10),
        )

        CIFAR10_Tanh2.to(device)
        train_net(CIFAR10_Tanh2, i+1, device)
        acc = test_net(CIFAR10_Tanh2, device)
        accuracies.append(acc)

    m = np.array(accuracies)
    print(f"Mean accuracy  of {i+1} epoch under two Tanh on test set: {np.mean(m)}")


# Initial LeNet-1: two Softplus
accuracies = []
for i in range(0,experiments):
    for j in range(0, experiments):
        CIFAR10_SP2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.Softplus(),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.Softplus(),
            nn.AvgPool2d(kernel_size=2),

            nn.Flatten(),

            nn.Linear(576, 128),
            nn.Linear(128, 10),
        )

        CIFAR10_SP2.to(device)
        train_net(CIFAR10_SP2, i+1, device)
        acc = test_net(CIFAR10_SP2, device)
        accuracies.append(acc)

    m = np.array(accuracies)
    print(f"Mean accuracy  of {i+1} epoch under two Softplus on test set: {np.mean(m)}")


class TaylorExpansion4(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t):
        t1 = torch.mul(t, 0.5002)
        t2 = torch.add(0.1992, t1)
        t3 = torch.pow(t, 2)
        t4 = torch.add(t2, torch.mul(t3, 0.1997))

        return t4

# Initial LeNet-1: two 2-degree Taylor expansion of ReLU
accuracies = []
for i in range(0,experiments):
    for j in range(0, experiments):
        CIFAR10_TayloyEx4 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            TaylorExpansion4(),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            TaylorExpansion4(),
            nn.AvgPool2d(kernel_size=2),

            nn.Flatten(),

            nn.Linear(576, 128),
            nn.Linear(128, 10),
        )

        CIFAR10_TayloyEx4.to(device)
        train_net(CIFAR10_TayloyEx4, i+1, device)
        acc = test_net(CIFAR10_TayloyEx4, device)
        accuracies.append(acc)

    m = np.array(accuracies)
    print(f"Mean accuracy  of {i+1} epoch on test set: {np.mean(m)}")