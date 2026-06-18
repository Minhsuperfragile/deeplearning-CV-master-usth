from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from mpmath import re
from sklearn.externals.array_api_compat.numpy import rec
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sympy.geometry.entity import x
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_dataset.data.unsqueeze(1)),
    batch_size=128,
    shuffle=True,
)

test_example = test_dataset.data[0].unsqueeze(0).unsqueeze(0).to("cuda")

# test_loader = torch.utils.data.DataLoader(
#    torch.utils.data.TensorDataset(test_dataset.data.unsqueeze(1)),
#    batch_size=128,
#    shuffle=True,
# )


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
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    dataloader: torch.utils.data.DataLoader = train_loader,
    epoch=10,
    device="cuda",
) -> nn.Module:
    if optimizer is None:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    model = model.to(device)
    model.train()
    for e in range(epoch):
        pbar = tqdm(enumerate(dataloader))
        pbar.set_description(f"Training model using {device}")
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Loss: {loss.item()}")
    return model


def test_model(
    model: nn.Module, test_example=test_example
) -> tuple[torch.Tensor, torch.Tensor]:
    """Test model on a single image"""
    model.eval()
    with torch.no_grad():
        reconstructed_image = model(test_example)
    return test_example.squeeze(0), reconstructed_image.squeeze(0)


def plot_images(
    *args, layout=(1, 2), titles=[""], save_path="./images/plot.png"
) -> None:
    """Plot multiple images
    Args:
        args: Images to be plotted
        layout: Tuple of (rows, columns)
        save_path: Path to save the plot
    Returns:
        fig, axes: Figure and axes objects
    """
    if len(titles) < len(args):
        titles = [f"Image {i}" for i in range(len(args))]
    if len(args) > 1:
        layout = (1, len(args))
    else:
        layout = (1, 1)

    fig, axes = plt.subplots(*layout)
    # Plot images
    for ax, image in zip(axes, args):
        ax.imshow(image, cmap="gray")
        ax.axis("off")

    plt.savefig(save_path)
    plt.show(block=False)

    return


def part_1_2(model: nn.Module = ConvolutionalAutoEncoder()) -> ConvolutionalAutoEncoder:
    """Experiment with Conv AutoEncoder on MNIST"""
    # Train model with MSE
    criterion = nn.MSELoss()
    model = train_model(model=model, criterion=criterion)

    # Visualize reconstructed image
    reconstructed_image, test_example = test_model(model)

    plot_images(
        reconstructed_image,
        test_example,
        layout=(1, 2),
        save_path="images/part_1_2.png",
    )
    return model


def part_1_3(model: nn.Module = ConvolutionalAutoEncoder()) -> ConvolutionalAutoEncoder:
    criterion = nn.BCELoss()

    class Wrapper(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

            self.model = model

        def forward(self, x):
            return F.sigmoid(self.model(x))

    model = Wrapper()
    model = train_model(model, criterion)

    # Visualize reconstructed image
    reconstructed_image, test_example = test_model(model)
    plot_images(
        reconstructed_image.cpu(),
        test_example.cpu(),
        layout=(1, 2),
        save_path="images/part_1_3.png",
    )

    return model


def part_1_4(*args, test_example: torch.Tensor = test_example) -> None:
    """Add noise to test image and reconstruct it. Pass in 1 or more models into args"""
    noise = torch.rand_like(test_example) - 0.5
    noisy_x = torch.clamp(test_example + 0.5 * noise, 0, 1).to("cuda")

    reconstructed_images = set()
    titles = set()
    for model in args:
        reconstructed_image = model(noisy_x)
        reconstructed_images.add(reconstructed_image.cpu())
        titles.add(model.__class__.__name__)

    plot_images(
        test_example,
        *reconstructed_images,
        titles=["Original", *titles],
        layout=(1, len(args) + 1),
        save_path="images/part_1_4.png",
    )
    return None


def part_1_5(*args, test_example: torch.Tensor = test_example) -> None:
    """Add noise to test image and reconstruct it. Pass in 1 or more models into args"""
    noisy_x = torch.rand_like(test_example)
    latent_space = args[0].encoder(noisy_x)
    noisy_latent = torch.rand_like(latent_space).to("cuda")

    reconstructed_images = set()
    titles = set()
    for model in args:
        reconstructed_image = model.decoder(noisy_latent)
        reconstructed_images.add(reconstructed_image.cpu())
        titles.add(model.__class__.__name__)

    plot_images(
        *reconstructed_images,
        titles=[*titles],
        layout=(1, len(args)),
        save_path="images/part_1_5.png",
    )
    return None


def part_1_6(*args: ConvolutionalAutoEncoder, x1: torch.Tensor, x2: torch.Tensor):
    # TODO: Implement this function
    reconstructed_images = []
    titles = []
    alpha = np.arange(0.1, 0.9, 0.1)  # Interpolation factor
    x1 = x1.repeat(len(alpha))
    x2 = x2.repeat(len(alpha))

    for model in args:
        z1 = model.encoder(x1)
        z2 = model.encoder(x2)
        # Interpolate between z1 and z2
        z_interp = (1 - alpha) * z1 + alpha * z2
        # Decode the interpolated latent vector
        reconstructed_image = model.decoder(z_interp)
        reconstructed_images.append(reconstructed_image.cpu())
        titles.append(model.__class__.__name__)

    plot_images(
        *reconstructed_images,
        titles=[*titles],
        layout=(2, len(args)),
        save_path="images/part_1_6.png",
    )
    return None


if __name__ == "__main__":
    mse_cae_model: ConvolutionalAutoEncoder = part_1_2()
    bse_cae_model: ConvolutionalAutoEncoder = part_1_3()
    part_1_4(mse_cae_model, bse_cae_model)
    part_1_5(mse_cae_model, bse_cae_model)

    x1 = test_dataset.data[0].unsqueeze(0).unsqueeze(0).to("cuda")
    x2 = test_dataset.data[1].unsqueeze(0).unsqueeze(0).to("cuda")

    part_1_6(mse_cae_model, bse_cae_model, x1=x1, x2=x2)

    print("All parts completed successfully!")
