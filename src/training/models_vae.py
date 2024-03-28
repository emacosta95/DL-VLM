from re import X
from typing import Dict, List, Tuple
import torch
import torch.nn as nn



class Encode(nn.Module):
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
                nn.AvgPool2d(kernel_size=pooling_size),
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
                    nn.AvgPool2d(kernel_size=pooling_size),
                    nn.BatchNorm2d(hidden_channels[i + 1]),
                ),
            )

        self.final_mu = nn.Sequential(
            nn.Linear(
                hidden_channels[-1]
                * int((input_size[0]*input_size[1])/ (pooling_size ** len(hidden_channels))),
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
                * int((input_size[0]*input_size[1]) / (pooling_size ** len(hidden_channels))),
                100,
            ),
            activation,
            nn.Linear(100, 50),
            activation,
            nn.Linear(50, latent_dimension),
        )

    def forward(self, x: torch.Tensor) -> Tuple:
        for conv in self.conv_list:
            x = conv(x)
        x = x.view(x.shape[0], -1)

        x_mu = self.final_mu(x)
        x_logstd = self.final_logsigma(x)
        return x_mu, x_logstd


class Encode3D(nn.Module):
    def __init__(
        self,
        input_channels: int,
        input_size: List,
        hidden_channels: List,
        latent_dimension: int,
        padding: List,
        padding_mode: str,
        kernel_size: List,
        pooling_size: List,
        activation: str,
    ):
        super().__init__()

        activation = getattr(torch.nn, activation)()
        self.conv_list = nn.ModuleList([])

        self.conv_list.add_module(
            "block_0",
            nn.Sequential(
                # nn.BatchNorm1d(input_channels),
                nn.Conv3d(
                    in_channels=input_channels,
                    out_channels=hidden_channels[0],
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode="circular",
                ),
                activation,
                nn.AvgPool3d(kernel_size=pooling_size),
                nn.BatchNorm3d(hidden_channels[0]),
            ),
        )

        for i in range(len(hidden_channels) - 1):
            self.conv_list.add_module(
                f"block_{i+1}",
                nn.Sequential(
                    # nn.BatchNorm1d(input_channels),
                    nn.Conv3d(
                        in_channels=hidden_channels[i],
                        out_channels=hidden_channels[i + 1],
                        kernel_size=kernel_size,
                        padding=padding,
                        padding_mode="circular",
                    ),
                    activation,
                    nn.AvgPool3d(kernel_size=pooling_size),
                    nn.BatchNorm3d(hidden_channels[i + 1]),
                ),
            )

        self.final_mu = nn.Sequential(
            nn.Linear(
                hidden_channels[-1]
                * int(
                    (input_size[0] // (pooling_size[0] ** len(hidden_channels)))
                    * (input_size[1] // (pooling_size[1] ** len(hidden_channels)))
                    * (input_size[2] // (pooling_size[2] ** len(hidden_channels)))
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
                    (input_size[0] // (pooling_size[0] ** len(hidden_channels)))
                    * (input_size[1] // (pooling_size[1] ** len(hidden_channels)))
                    * (input_size[2] // (pooling_size[2] ** len(hidden_channels)))
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
            x = conv(x)
        x = x.view(x.shape[0], -1)

        x_mu = self.final_mu(x)
        x_logstd = self.final_logsigma(x)
        return x_mu, x_logstd


class Encode3db(nn.Module):
    def __init__(
        self,
        input_channels: int,
        linear_input_size: int,
        hidden_channels: int,
        latent_dimension: int,
        padding: int,
        padding_mode: str,
        kernel_size: int,
        pooling_size: int,
        activation: str,
    ):
        super().__init__()

        activation = getattr(torch.nn, activation)()

        self.block_1 = nn.Sequential(
            # nn.BatchNorm1d(input_channels),
            nn.Conv3d(
                in_channels=input_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="circular",
            ),
            activation,
            nn.AvgPool3d(kernel_size=pooling_size),
            nn.BatchNorm3d(hidden_channels),
        )
        self.block_2 = nn.Sequential(
            nn.Conv3d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="circular",
            ),
            activation,
            nn.AvgPool3d(kernel_size=pooling_size),
            nn.BatchNorm3d(hidden_channels),
        )
        self.block_3 = nn.Sequential(
            nn.Conv3d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="circular",
            ),
            activation,
            nn.AvgPool3d(kernel_size=pooling_size),
            nn.BatchNorm3d(hidden_channels),
        )
        self.final_mu = nn.Sequential(
            nn.Linear(
                hidden_channels * int(linear_input_size / (pooling_size**3)) ** 3,
                100,
            ),
            activation,
            nn.Linear(100, 50),
            activation,
            nn.Linear(50, latent_dimension),
        )
        self.final_logsigma = nn.Sequential(
            nn.Linear(
                hidden_channels * int(linear_input_size / (pooling_size**3)) ** 3,
                100,
            ),
            activation,
            nn.Linear(100, 50),
            activation,
            nn.Linear(50, latent_dimension),
        )

    def forward(self, x: torch.Tensor) -> Tuple:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
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
        dx: float,
    ):
        super().__init__()

        activation = getattr(torch.nn, activation)()

        self.output_size = output_size
        self.pooling_size = pooling_size
        self.dx = dx

        self.recon_block = nn.Sequential(
            nn.Linear(
                latent_dimension,
                int((output_size[0]*output_size[1]) / (pooling_size[0]) ** len(hidden_channels))
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
                        kernel_size=[kernel_size[0] + 1,kernel_size[1]],
                        stride=[2,1],
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
                    kernel_size=[kernel_size[0] + 1,kernel_size[1]],
                    stride=[2,1],
                    padding=padding,
                ),
            ),
        )

        self.hidden_channel = hidden_channels

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.recon_block(z)
        x = x.view(
            -1,
            self.hidden_channel[0],
            int((self.output_size[0]*self.output_size[1]) / (self.pooling_size[0] ** len(self.hidden_channel))),
        )
        for conv in self.conv_list:
            x = conv(x)

        return x


class DecodeNorm3D(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        hidden_channels: List,
        output_channels: int,
        output_size: List,
        padding: List,
        padding_mode: str,
        kernel_size: List,
        pooling_size: int,
        activation: str,
        dx: float,
    ):
        super().__init__()

        activation = getattr(torch.nn, activation)()

        self.output_size = output_size
        self.pooling_size = pooling_size
        self.dx = dx

        self.recon_block = nn.Sequential(
            nn.Linear(
                latent_dimension,
                int(
                    (output_size[0] // (pooling_size[0]) ** len(hidden_channels))
                    * (output_size[1] // (pooling_size[1]) ** len(hidden_channels))
                    * (output_size[2] // (pooling_size[2]) ** len(hidden_channels))
                )
                * hidden_channels[0],
            ),
        )

        self.conv_list = nn.ModuleList([])

        self.adaptive_pooling = nn.AdaptiveAvgPool3d(output_size=output_size)

        for i in range(len(hidden_channels) - 1):
            if i == 0:
                self.conv_list.add_module(
                    f"block_{i}",
                    nn.Sequential(
                        nn.ConvTranspose3d(
                            in_channels=hidden_channels[i],
                            out_channels=hidden_channels[i + 1],
                            kernel_size=[
                                kernel_size[0] + 1,
                                kernel_size[1] + 1,
                                kernel_size[2] + 1,
                            ],
                            stride=2,
                            padding=padding,
                        ),
                        activation,
                        nn.BatchNorm3d(hidden_channels[i + 1]),
                    ),
                )
            else:
                self.conv_list.add_module(
                    f"block_{i}",
                    nn.Sequential(
                        nn.ConvTranspose3d(
                            in_channels=hidden_channels[i],
                            out_channels=hidden_channels[i + 1],
                            kernel_size=[
                                kernel_size[0] + 1,
                                kernel_size[1] + 1,
                                kernel_size[2] + 1,
                            ],
                            stride=2,
                            padding=padding,
                        ),
                        activation,
                        nn.BatchNorm3d(hidden_channels[i + 1]),
                    ),
                )

        self.conv_list.add_module(
            f"block_{i+1}",
            nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels=hidden_channels[-1],
                    out_channels=output_channels,
                    kernel_size=[
                        kernel_size[0] + 1,
                        kernel_size[1] + 1,
                        kernel_size[2] + 1,
                    ],
                    stride=2,
                    padding=padding,
                ),
            ),
        )

        self.hidden_channel = hidden_channels

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.recon_block(z)
        x = x.view(
            -1,
            self.hidden_channel[0],
            int(
                self.output_size[0]
                // (self.pooling_size[0] ** len(self.hidden_channel))
            ),
            int(
                self.output_size[1]
                // (self.pooling_size[1] ** len(self.hidden_channel))
            ),
            int(
                self.output_size[2]
                // (self.pooling_size[2] ** len(self.hidden_channel))
            ),
        )
        for conv in self.conv_list:
            x = conv(x)
        # positivity
        x = torch.sigmoid(x)
        x = self.adaptive_pooling(x)
        # x = nn.functional.gelu(x)
        # normalization
        # condition
        norm = torch.sum(x, dim=(2, 3, 4)) * self.dx**3
        x = x / norm[:, :, None, None, None]

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
    ):
        super().__init__()

        self.encoder = Encode(
            input_channels=input_channels,
            input_size=input_size,
            hidden_channel=hidden_channel,
            latent_dimension=latent_dimension,
            padding=padding,
            padding_mode=padding_mode,
            kernel_size=kernel_size,
        )

        self.decoder = Decoder(
            output_channels=input_channels,
            output_size=input_size,
            hidden_channel=hidden_channel,
            latent_dimension=latent_dimension,
            padding=padding,
            padding_mode=padding_mode,
            kernel_size=kernel_size,
        )

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)

        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = (logvar * 0.5).exp()
            return torch.distributions.Normal(loc=mu, scale=std).rsample()
            # std = logvar.mul(0.5).exp_()
            # eps = torch.empty_like(std).normal_()
            # return eps.mul(std).add_(mu)
        else:
            return mu
