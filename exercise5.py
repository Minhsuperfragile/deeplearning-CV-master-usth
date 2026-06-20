import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
)

test_example = test_dataset.data[0].unsqueeze(0).unsqueeze(0).to("cuda").to(torch.float)

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

    def forward(self, x) -> torch.Tensor:
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

    def forward(self, x) -> torch.Tensor:
        batch_size = x.shape[0]
        x = F.relu(self.fc(x))
        x = x.reshape((batch_size, 64, 7, 7))
        x = F.relu(self.conv1(x))
        x = F.sigmoid(self.conv2(x))
        return x


class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self, loss_function: str = "") -> None:
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.used_loss = loss_function

    def forward(self, x) -> torch.Tensor:
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
        pbar = tqdm(dataloader)
        pbar.set_description(f"Training model using {device}")
        for data, _ in pbar:
            data = data.to(device).to(torch.float)

            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data)
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
    *args, layout: tuple[int, int] = (1, 2), titles=[""], save_path="./images/plot.png"
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

    fig, axes = plt.subplots(*layout)
    axes = axes.flatten()
    # Plot images
    for ax, image, title in zip(axes, args, titles):
        if image.dim() == 4:
            image = image.squeeze(0)  # quick fix
        ax.imshow(image.permute(1, 2, 0).cpu(), cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show(block=False)

    return


def part_1_2(
    model: nn.Module = ConvolutionalAutoEncoder(loss_function="MSE"),
) -> ConvolutionalAutoEncoder:
    """Experiment with Conv AutoEncoder on MNIST"""
    print("----Running 1.2----")

    # Train model with MSE
    criterion = nn.MSELoss()
    model = train_model(model=model, criterion=criterion)

    test_example, reconstructed_image = test_model(model)
    plot_images(
        reconstructed_image,
        test_example,
        layout=(1, 2),
        titles=["Reconstructed", "Original"],
        save_path="images/part_1_2.png",
    )
    return model


def part_1_3(
    model: nn.Module = ConvolutionalAutoEncoder(loss_function="BCE"),
) -> ConvolutionalAutoEncoder:
    print("----Running 1.3----")
    criterion = nn.BCELoss()

    class Wrapper(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

            self.model = model

        def forward(self, x):
            return F.sigmoid(self.model(x))

    model = Wrapper()
    model = train_model(model, criterion).model

    test_example, reconstructed_image = test_model(model)
    plot_images(
        reconstructed_image.cpu(),
        test_example.cpu(),
        layout=(1, 2),
        titles=["Reconstructed", "Original"],
        save_path="images/part_1_3.png",
    )

    return model


def part_1_4(
    *args: ConvolutionalAutoEncoder, test_example: torch.Tensor = test_example
) -> None:
    """Add noise to test image and reconstruct it. Pass in 1 or more models into args"""
    print("----Running 1.4----")
    noise = torch.rand_like(test_example) - 0.5
    noisy_x = torch.clamp(test_example + 0.5 * noise, 0, 1).to("cuda")

    reconstructed_images = set()
    titles = []
    for model in args:
        _, reconstructed_image = test_model(model=model, test_example=noisy_x)
        reconstructed_images.add(reconstructed_image.cpu())
        titles.append(model.used_loss)

    plot_images(
        test_example,
        *reconstructed_images,
        titles=["Original"] + titles,
        layout=(1, len(args) + 1),
        save_path="images/part_1_4.png",
    )
    return None


def part_1_5(
    *args: ConvolutionalAutoEncoder, test_example: torch.Tensor = test_example
) -> None:
    """Add noise to test image and reconstruct it. Pass in 1 or more models into args"""
    print("----Running 1.5----")
    noisy_x = torch.rand_like(test_example)
    latent_space = args[0].encoder(noisy_x)
    noisy_latent = torch.rand_like(latent_space).to("cuda")

    reconstructed_images = set()
    titles = []
    for model in args:
        reconstructed_image = model.decoder(noisy_latent)
        reconstructed_images.add(reconstructed_image.cpu().detach())
        titles.append(model.used_loss)

    plot_images(
        *reconstructed_images,
        titles=titles,
        layout=(1, len(args)),
        save_path="images/part_1_5.png",
    )
    return None


def part_1_6(*args: ConvolutionalAutoEncoder, x1: torch.Tensor, x2: torch.Tensor):
    print("----Running 1.6----")
    reconstructed_images = []
    alpha = torch.arange(0.1, 1.0, 0.1).to("cuda")  # Interpolation factor
    batch = len(alpha)
    x1 = x1.expand(batch, -1, -1, -1)
    x2 = x2.expand(batch, -1, -1, -1)

    titles = [str(round(float(i), 1)) for i in list(alpha.expand(2, -1).flatten())]
    for model in args:
        z1 = model.encoder(x1)
        z2 = model.encoder(x2)
        # Interpolate between z1 and z2
        z_interp = (1 - alpha).unsqueeze(1) * z1 + alpha.unsqueeze(1) * z2
        # Decode the interpolated latent vector
        reconstructed_image = model.decoder(z_interp)
        reconstructed_images.extend(list(reconstructed_image.cpu().detach()))

    plot_images(
        *reconstructed_images,
        titles=titles,
        layout=(len(args), batch),
        save_path="images/part_1_6.png",
    )
    return None


# if __name__ == "__main__":
#     mse_cae_model: ConvolutionalAutoEncoder = part_1_2()
#     bse_cae_model: ConvolutionalAutoEncoder = part_1_3()
#     part_1_4(mse_cae_model, bse_cae_model)
#     part_1_5(mse_cae_model, bse_cae_model)

#     indices_1 = (test_dataset.targets == 1).nonzero(as_tuple=True)[0]
#     indices_8 = (test_dataset.targets == 8).nonzero(as_tuple=True)[0]

#     x1 = (
#         test_dataset.data[indices_1[0]]
#         .unsqueeze(0)
#         .unsqueeze(0)
#         .to("cuda")
#         .to(torch.float)
#     )
#     x2 = (
#         test_dataset.data[indices_8[0]]
#         .unsqueeze(0)
#         .unsqueeze(0)
#         .to("cuda")
#         .to(torch.float)
#     )

#     part_1_6(mse_cae_model, bse_cae_model, x1=x1, x2=x2)

#     print("All parts completed successfully!")


# VAE
class VariationalAutoEncoder(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = Encoder()
        self.fc_mu = nn.Linear(10, 10, bias=False)
        self.fc_var = nn.Linear(10, 10, bias=False)
        self.decoder = Decoder()

    def encode(self, x):
        """Return a mu,var vector of size (batch, latent dim)"""
        z: torch.Tensor = self.encoder(x)
        mu = self.fc_mu(z)
        var = self.fc_var(z)

        return mu, var  # (batch, vector)

    def sampling(self, mean, var_log):
        """Sampling a latent distribution from mu and var vector"""
        std = torch.exp(0.5 * var_log)
        z = mean + var_log * torch.randn_like(std)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, var = self.encode(x)
        z = self.sampling(mu, var)
        out = self.decode(z)
        return out


def part_2_2(
    model: VariationalAutoEncoder = VariationalAutoEncoder(),
) -> VariationalAutoEncoder:
    criterion = nn.MSELoss()
    model = train_model(model, criterion)

    return model


if __name__ == "__main__":
    model = VariationalAutoEncoder().to("cuda")
    test_batch = test_example.expand(16, -1, -1, -1)
    mu, lv = model.encode(test_batch)
    print(mu.shape, lv.shape)
    z = model.sampling(mu, lv)
    print(z.shape)
