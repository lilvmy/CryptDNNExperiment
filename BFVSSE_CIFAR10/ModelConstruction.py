import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import logging as log


# python verison lower than 3.9
root_logger = log.getLogger()
root_logger.setLevel(log.DEBUG) # or whatever
handler = log.FileHandler('ModelConstruction.log', 'w', 'utf-8') # or whatever
handler.setFormatter(log.Formatter(fmt='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')) # or whatever
root_logger.addHandler(handler)

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

# Initial CIFAR-10: two ReLU
accuracies = []
for i in range(0, experiments):
    VGG_16 = nn.Sequential(
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

    VGG_16.to(device)
    train_net(VGG_16, 50, device)
    acc = test_net(VGG_16, device)
    accuracies.append(acc)

m = np.array(accuracies)
log.info(f"Results for VGG_16:")
log.info(f"10 accuracy are: {m}")
log.info(f"Mean accuracy on test set: {np.mean(m)}")
log.info(f"Var: {np.var(m)}")
torch.save(VGG_16, "VGG_16.pt")


# Initial CIFAR-10: single ReLU
accuracies = []
for i in range(0, experiments):
    VGG_single_relu_16 = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2),

        nn.Conv2d(32, 64, kernel_size=3, stride=2),
        nn.AvgPool2d(kernel_size=2),

        nn.Flatten(),

        nn.Linear(576, 128),
        nn.Linear(128, 10),
    )

    VGG_single_relu_16.to(device)
    train_net(VGG_single_relu_16, 50, device)
    acc = test_net(VGG_single_relu_16, device)
    accuracies.append(acc)

m = np.array(accuracies)
log.info(f"Results for VGG_single_16:")
log.info(f"10 accuracy are: {m}")
log.info(f"Mean accuracy on test set: {np.mean(m)}")
log.info(f"Var: {np.var(m)}")
torch.save(VGG_single_relu_16, "VGG_single_16.pt")

# Initial CIFAR-10: single square
class Square(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t):
        return torch.pow(t, 2)

accuracies = []
for i in range(0, experiments):
    VGG_single_square_16 = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5),
        Square(),
        nn.AvgPool2d(kernel_size=2),

        nn.Conv2d(32, 64, kernel_size=3, stride=2),
        nn.AvgPool2d(kernel_size=2),

        nn.Flatten(),

        nn.Linear(576, 128),
        nn.Linear(128, 10),
    )

    VGG_single_square_16.to(device)
    train_net(VGG_single_square_16, 50, device)
    acc = test_net(VGG_single_square_16, device)
    accuracies.append(acc)

m = np.array(accuracies)
log.info(f"Results for VGG_single_square_16:")
log.info(f"10 accuracy are: {m}")
log.info(f"Mean accuracy on test set: {np.mean(m)}")
log.info(f"Var: {np.var(m)}")
torch.save(VGG_single_square_16, "VGG_single_square_16.pt")




