from re import X
from typing import Dict, List, Tuple
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        input_size: int,
        hidden_channels: List,
        latent_dimension: int,
        padding: int,
        padding_mode: str,
        kernel_size: int,
        pooling_size: int,
        activation: str,
    ):
        super().__init__()

        activation = getattr(torch.nn, activation)()
        self.conv_list = nn.ModuleList([])

        self.conv_list.add_module(
            "block_0",
            nn.Sequential(
                # nn.BatchNorm1d(input_channels),
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=hidden_channels[0],
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode="circular",
                ),
                activation,
                nn.AvgPool2d(kernel_size=[pooling_size, 1]),
                nn.BatchNorm2d(hidden_channels[0]),
            ),
        )

        for i in range(len(hidden_channels) - 1):
            self.conv_list.add_module(
                f"block_{i+1}",
                nn.Sequential(
                    # nn.BatchNorm1d(input_channels),
                    nn.Conv2d(
                        in_channels=hidden_channels[i],
                        out_channels=hidden_channels[i + 1],
                        kernel_size=kernel_size,
                        padding=padding,
                        padding_mode="circular",
                    ),
                    activation,
                    nn.AvgPool2d(kernel_size=[pooling_size, 1]),
                    nn.BatchNorm2d(hidden_channels[i + 1]),
                ),
            )

        self.final_mu = nn.Sequential(
            nn.Linear(
                hidden_channels[-1]
                * int(
                    (input_size[0] * input_size[1])
                    / (pooling_size ** len(hidden_channels))
                ),
                100,
            ),
            activation,
            nn.Linear(100, 50),
            activation,
            nn.Linear(50, latent_dimension),
        )
        self.final_logsigma = nn.Sequential(
            nn.Linear(
                hidden_channels[-1]
                * int(
                    (input_size[0] * input_size[1] * input_channels)
                    / (pooling_size ** len(hidden_channels))
                ),
                100,
            ),
            activation,
            nn.Linear(100, 50),
            activation,
            nn.Linear(50, latent_dimension),
        )

    def forward(self, x: torch.Tensor) -> Tuple:
        for conv in self.conv_list:
            print("x.shape=", x.shape)
            x = conv(x)
        print("x.shape=", x.shape)
        x = x.view(x.shape[0], -1)

        x_mu = self.final_mu(x)
        x_logstd = self.final_logsigma(x)
        return x_mu, x_logstd


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        hidden_channels: List,
        output_channels: int,
        output_size: int,
        padding: int,
        padding_mode: str,
        kernel_size: int,
        pooling_size: int,
        activation: str,
    ):
        super().__init__()

        activation = getattr(torch.nn, activation)()

        self.output_size = output_size
        self.pooling_size = pooling_size

        self.recon_block = nn.Sequential(
            nn.Linear(
                latent_dimension,
                int(
                    (output_size[0] * output_size[1])
                    / (pooling_size) ** len(hidden_channels)
                )
                * hidden_channels[0],
            ),
        )

        self.conv_list = nn.ModuleList([])

        for i in range(len(hidden_channels) - 1):
            self.conv_list.add_module(
                f"block_{i}",
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=hidden_channels[i],
                        out_channels=hidden_channels[i + 1],
                        kernel_size=[kernel_size[0] + 1, kernel_size[1]],
                        stride=[2, 1],
                        padding=padding,
                    ),
                    activation,
                    nn.BatchNorm2d(hidden_channels[i + 1]),
                ),
            )

        self.conv_list.add_module(
            f"block_{i+1}",
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=hidden_channels[-1],
                    out_channels=output_channels,
                    kernel_size=[kernel_size[0] + 1, kernel_size[1]],
                    stride=[2, 1],
                    padding=padding,
                ),
            ),
        )

        self.hidden_channel = hidden_channels

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        print("x.shape=", z.shape)
        x = self.recon_block(z)
        print("x.shape=", x.shape)
        x = x.view(
            -1,
            self.hidden_channel[0],
            int(
                (self.output_size[0]) / (self.pooling_size ** len(self.hidden_channel))
            ),
            self.output_size[1],
        )
        for conv in self.conv_list:
            print("x.shape", x.shape)
            x = conv(x)
        print('x.shape',x.shape)
        return x


class VarAE(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        hidden_channel: int,
        input_channels: int,
        input_size: int,
        padding: int,
        padding_mode: str,
        kernel_size: int,
        Loss: nn.Module,
        pooling_size: int,
        activation: nn.Module,
    ):
        super().__init__()

        self.encoder = Encoder(
            input_channels=input_channels,
            input_size=input_size,
            hidden_channels=hidden_channel,
            latent_dimension=latent_dimension,
            padding=padding,
            padding_mode=padding_mode,
            kernel_size=kernel_size,
            pooling_size=pooling_size,
            activation=activation,
        )

        self.decoder = Decoder(
            output_channels=input_channels,
            output_size=input_size,
            hidden_channels=hidden_channel,
            latent_dimension=latent_dimension,
            padding=padding,
            padding_mode=padding_mode,
            kernel_size=kernel_size,
            pooling_size=pooling_size,
            activation=activation,
        )

        self.loss = Loss

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)

        return x_recon, latent_mu, latent_logvar

    def _latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = (logvar * 0.5).exp()
            return torch.distributions.Normal(loc=mu, scale=std).rsample()
            # std = logvar.mul(0.5).exp_()
            # eps = torch.empty_like(std).normal_()
            # return eps.mul(std).add_(mu)
        else:
            return mu

    def train_step(self, batch: Tuple, device: str):
        x = batch[0]
        x = x.unsqueeze(1).to(device=device)
        latent_mu, latent_logvar = self.encoder(x)
        latent = self._latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        loss, kldiv = self.loss(x_recon, x, latent_mu, latent_logvar)
        return loss, kldiv

    def valid_step(self, batch: Tuple, device: str):
        x = batch[0]
        x = x.unsqueeze(1).to(device=device)
        latent_mu, latent_logvar = self.encoder(x)
        latent = self._latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        loss, kldiv = self.loss(x_recon, x, latent_mu, latent_logvar)
        return loss, kldiv
