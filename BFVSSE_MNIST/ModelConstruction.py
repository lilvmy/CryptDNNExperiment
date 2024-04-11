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

train_set = torchvision.datasets.MNIST(
    root = './data',
    train=True,
    download=True,
    transform=transform
)

test_set = torchvision.datasets.MNIST(
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

class Square(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t):
        return torch.pow(t, 2)


# class TanhPlus(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, t):
#         tmp1 = torch.multiply(t, 0.5)
#         tmp2 = torch.multiply(torch.pow(t, 2), 1/4)
#         tmp3 = torch.multiply(torch.pow(t, 4), 1/24)
#         cons = torch.multiply(torch.log(torch.tensor(2)), 1/2)
#         sum1 = torch.add(cons, tmp1)
#         sum2 = torch.add(sum1, tmp2)
#         res = torch.sub(sum2, tmp3)
#         return res
#         # return torch.divide((torch.log(torch.exp(t) + torch.exp(-t))) + t, 2.0)

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
    LeNet1_ReLU2 = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=5),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2),

        nn.Conv2d(4, 12, kernel_size=5),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2),

        nn.Flatten(),

        nn.Linear(192, 10),
    )

    LeNet1_ReLU2.to(device)
    train_net(LeNet1_ReLU2, 10, device)
    acc = test_net(LeNet1_ReLU2, device)
    accuracies.append(acc)

m = np.array(accuracies)
log.info(f"Results for LeNet-1_ReLU2:")
log.info(f"Mean accuracy on test set: {np.mean(m)}")
log.info(f"Var: {np.var(m)}")

# Optional: save the last trained LeNet-1_ReLU2:
torch.save(LeNet1_ReLU2, "LeNet1_ReLU2.pt")


# LeNet-1 with a single ReLU
accuracies = []
for i in range(0, experiments):
    LeNet1_singleReLU = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=5),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2),

        nn.Conv2d(4, 12, kernel_size=5),
        # nn.Tanh(),
        nn.AvgPool2d(kernel_size=2),

        nn.Flatten(),

        nn.Linear(192, 10),
    )

    LeNet1_singleReLU.to(device)
    train_net(LeNet1_singleReLU, 15, device)
    acc = test_net(LeNet1_singleReLU, device)
    accuracies.append(acc)

m = np.array(accuracies)
log.info(f"Results for LeNet-1 (single ReLU):")
log.info(f"Mean accuracy on test set: {np.mean(m)}")
log.info(f"Var: {np.var(m)}")

# Optional: save the last trained LeNet-1 (single tanh):
torch.save(LeNet1_singleReLU, "LeNet1_single_ReLU.pt")


# Approximated LeNet-1 (single Square)
accuracies = []
for i in range(0, experiments):
    Approx_LeNet1_Single_Square = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=5),
        Square(),
        nn.AvgPool2d(kernel_size=2),

        nn.Conv2d(4, 12, kernel_size=5),
        # nn.Tanh(),
        nn.AvgPool2d(kernel_size=2),

        nn.Flatten(),

        nn.Linear(192, 10),
    )

    Approx_LeNet1_Single_Square.to(device)
    train_net(Approx_LeNet1_Single_Square, 15, device)
    acc = test_net(Approx_LeNet1_Single_Square, device)
    accuracies.append(acc)

m = np.array(accuracies)
log.info(f"Results for approximated LeNet-1 (single Square):")
log.info(f"Mean accuracy on test set: {np.mean(m)}")
log.info(f"Var: {np.var(m)}")

# Optional: save the last trained approximated LeNet-1:
torch.save(Approx_LeNet1_Single_Square, "LeNet1_Approx_single_Square.pt")

# Approximated LeNet-1 (Square2)
accuracies = []
for i in range(experiments):
    Approx_LeNet1_Square2 = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=5),
        Square(),
        nn.AvgPool2d(kernel_size=2),

        nn.Conv2d(4, 12, kernel_size=5),
        Square(),
        nn.AvgPool2d(kernel_size=2),

        nn.Flatten(),

        nn.Linear(192, 10),
    )

    Approx_LeNet1_Square2.to(device)
    train_net(Approx_LeNet1_Square2, 15, device)
    acc = test_net(Approx_LeNet1_Square2, device)
    accuracies.append(acc)

m = np.array(accuracies)
log.info(f"Results for approximated LeNet-1 (Square):")
log.info(f"Mean accuracy on test set: {np.mean(m)}")
log.info(f"Var: {np.var(m)}")

# Optional: save the last trained approximated LeNet-1:
torch.save(Approx_LeNet1_Square2, "LeNet1_Approx_Square2.pt")

# Approximated LeNet-1 (single TanhPlus) - the one saved and used by the encrypted processing
model = torch.load("LeNet1_ReLU2.pt")
model.eval()
model.to(device)
acc = test_net(model, device)
log.info(f"Results for approximated LeNet-1_single_relu - the one saved to file: {acc}")




