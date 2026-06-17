import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

from exercise5 import train_loader

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset.data.unsqueeze(1), batch_size=128, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset.data.unsqueeze(1), batch_size=128, shuffle=True
)


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc = nn.Linear(10, 64 * 7 * 7)
        self.conv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.conv2 = nn.ConvTranspose2d(32, 1, 4, 2, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.fc(x))
        x = x.reshape((batch_size, 64, 7, 7))
        x = F.relu(self.conv1(x))
        x = F.sigmoid(self.conv2(x))
        return x


class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_model(
    model: nn.Module, criterion=None, optimizer=None, dataloader=train_loader
) -> nn.Module:

    return model


def test_model(model: nn.Module, dataloader: test_loader) -> None:

    return
