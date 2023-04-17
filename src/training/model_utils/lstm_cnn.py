import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from typing import Tuple


class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: Tuple,
    ) -> None:
        super().__init__()

        self.x_i = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.h_i = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.c_i = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )

        self.x_f = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.h_f = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.c_f = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )

        self.x_c = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.h_c = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.c_c = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )

        self.x_o = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.h_o = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.c_o = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        i = F.gelu(self.x_i(x) + self.h_i(h) + self.c_i(c))
        f = torch.sigmoid(self.x_f(x) + self.h_f(h) + self.c_f(c))
        c = f * c + i * torch.tanh(self.x_c(x) + self.h_c(h))
        o = F.gelu(self.x_o(x) + self.h_o(h) + self.c_o(c))
        h = o * torch.tanh(c)
        return x, h, c


class Encoder1D(nn.Module):
    def __init__(
        self,
        n_conv: int,
        activation: nn.Module,
        hc: int,
        in_channels: int,
        kernel_size: int,
        padding_mode: str,
        latent_dimension: int,
        n_layers: int,
        pooling_size: int,
        hidden_neurons: int,
    ) -> None:
        super().__init__()

        self.conv_part = nn.Sequential()
        n_conv = len(hc)
        self.conv_part.add_module(
            f"conv_{-1}",
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=hc[0],
                kernel_size=kernel_size,
                padding_mode=padding_mode,
                padding=(kernel_size - 1) // 2,
            ),
        )
        self.conv_part.add_module(f"act {-1}", activation)
        self.conv_part.add_module(
            f"pooling {-1}", nn.AvgPool1d(kernel_size=pooling_size)
        )
        for i in range(n_conv - 1):
            self.conv_part.add_module(
                f"conv_{i}",
                nn.Conv1d(
                    in_channels=hc[i - 1],
                    out_channels=hc[i],
                    kernel_size=kernel_size,
                    padding_mode=padding_mode,
                    padding=(kernel_size - 1) // 2,
                ),
            )
            self.conv_part.add_module(f"act_{i}", activation)
            self.conv_part.add_module(
                f"pooling {i}", nn.AvgPool1d(kernel_size=pooling_size)
            )
        self.conv_part.add_module("global pooling", nn.AdaptiveAvgPool1d(1))

        self.latent_operator = nn.Sequential()
        for i in range(n_layers):
            self.latent_operator.add_module(
                f"layer {i}", nn.Linear(hc[-1], hidden_neurons)
            )
            self.latent_operator.add_module(f"dens act {i}", activation)
        self.latent_operator.add_module(
            "final layer", nn.Linear(hidden_neurons, latent_dimension)
        )

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        x = self.conv_part(x)
        x = x.squeeze()
        l = self.latent_operator(x)
        return l


class Decoder1D(nn.Module):
    def __init__(
        self,
        n_conv: int,
        activation: nn.Module,
        hc: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        latent_dimension: int,
        n_layers: int,
        hidden_neurons: int,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.conv_part = nn.Sequential()
        n_conv = len(hc)
        self.conv_part.add_module(
            f"conv_{-1}",
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=hc[0],
                kernel_size=kernel_size + 1,
                padding_mode="zeros",
                padding=(kernel_size - 1) // 2,
            ),
        )
        self.conv_part.add_module(f"act_{-1}", activation)
        for i in range(n_conv - 1):
            self.conv_part.add_module(
                f"conv_{i}",
                nn.ConvTranspose1d(
                    in_channels=hc[i - 1],
                    out_channels=hc[i],
                    kernel_size=kernel_size + 1,
                    padding_mode="zeros",
                    padding=(kernel_size - 1) // 2,
                ),
            )
            self.conv_part.add_module(f"act_{i}", activation)
        self.conv_part.add_module(
            f"conv_{i+1}",
            nn.ConvTranspose1d(
                in_channels=hc[-1],
                out_channels=out_channels,
                kernel_size=kernel_size + 1,
                padding_mode="zeros",
                padding=(kernel_size - 1) // 2,
            ),
        )

        self.latent_operator = nn.Sequential()

        self.latent_operator.add_module(
            f"layer {-1}", nn.Linear(latent_dimension, hidden_neurons)
        )
        self.latent_operator.add_module(f"dens act {-1}", activation)
        for i in range(n_layers):
            self.latent_operator.add_module(
                f"layer {i}", nn.Linear(hidden_neurons, hidden_neurons)
            )
            self.latent_operator.add_module(f"dens act {i}", activation)
        self.latent_operator.add_module(
            "final layer", nn.Linear(hidden_neurons, in_channels * (kernel_size + 1))
        )

    def forward(self, l: torch.Tensor):
        x = self.latent_operator(l)
        x = x.view(-1, self.in_channels, self.kernel_size + 1)
        x = self.conv_part(x)
        return x


class LSTMcell(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.f_h = nn.Linear(input_size, output_size)
        self.f_x = nn.Linear(input_size, output_size)
        self.i_h = nn.Linear(input_size, output_size)
        self.i_x = nn.Linear(input_size, output_size)
        self.o_x = nn.Linear(input_size, output_size)
        self.o_h = nn.Linear(input_size, output_size)
        self.c_h = nn.Linear(input_size, output_size)
        self.c_x = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):

        f = torch.sigmoid(self.f_x(x) + self.f_h(h))
        i = torch.sigmoid(self.i_x(x) + self.i_h(h))
        o = torch.sigmoid(self.o_x(x) + self.o_h(h))
        c_tilde = torch.tanh(self.c_x(x) + self.c_h(h))
        c = f * c + i * c_tilde
        h = o * c

        return o, h, c
