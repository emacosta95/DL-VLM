import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from typing import Tuple, List


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
        size_restriction: int,
    ) -> None:
        super().__init__()

        self.size_restriction = size_restriction
        self.conv_part = nn.ModuleList()
        n_conv = len(hc)
        block = nn.Sequential()
        block.add_module(
            f"conv_{-1}",
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=hc[0],
                kernel_size=kernel_size,
                padding_mode=padding_mode,
                padding=(kernel_size - 1) // 2,
            ),
        )
        block.add_module(
            f"bn_{-1}",
            nn.BatchNorm1d(hc[-1]),
        )
        block.add_module(f"act {-1}", activation)
        self.conv_part.add_module(f"block {-1}", block)
        for i in range(n_conv - 1):
            block = nn.Sequential()
            block.add_module(
                f"conv_{i}",
                nn.Conv1d(
                    in_channels=hc[i - 1],
                    out_channels=hc[i],
                    kernel_size=kernel_size,
                    padding_mode=padding_mode,
                    padding=(kernel_size - 1) // 2,
                ),
            )
            block.add_module(
                f"bn_{i}",
                nn.BatchNorm1d(hc[i]),
            )
            block.add_module(f"act_{i}", activation)
            self.conv_part.add_module(f"block {i}", block)

        self.latent_operator = nn.Sequential()
        for i in range(n_layers):
            if i == 0:
                self.latent_operator.add_module(
                    f"layer {i}",
                    nn.Linear(hc[-1] * self.size_restriction, hidden_neurons),
                )
                self.latent_operator.add_module(f"dens act {i}", activation)
            else:
                self.latent_operator.add_module(
                    f"layer {i}", nn.Linear(hidden_neurons, hidden_neurons)
                )
                self.latent_operator.add_module(f"dens act {i}", activation)

        self.latent_operator.add_module(
            "final layer", nn.Linear(hidden_neurons, latent_dimension)
        )

    def forward(self, x: torch.Tensor):
        outputs = []
        for i, conv in enumerate(self.conv_part):
            x = conv(x)
            outputs.append(x)
        x = F.adaptive_avg_pool1d(x, output_size=self.size_restriction)
        # print("x restriction", x.shape)
        x = x.view(x.shape[0], -1)
        l = self.latent_operator(x)
        return l, outputs


class Encoder2D(nn.Module):
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
        size_restriction: int,
    ) -> None:
        super().__init__()

        self.size_restriction = size_restriction
        self.conv_part = nn.ModuleList()
        n_conv = len(hc)
        block = nn.Sequential()
        block.add_module(
            f"conv_{-1}",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hc[0],
                kernel_size=[1, kernel_size],
                padding_mode=padding_mode,
                padding=[0, (kernel_size - 1) // 2],
            ),
        )
        block.add_module(
            f"bn_{-1}",
            nn.BatchNorm2d(hc[-1]),
        )
        block.add_module(f"act {-1}", activation)
        self.conv_part.add_module(f"block {-1}", block)
        for i in range(n_conv - 1):
            block = nn.Sequential()
            block.add_module(
                f"conv_{i}",
                nn.Conv2d(
                    in_channels=hc[i - 1],
                    out_channels=hc[i],
                    kernel_size=[1, kernel_size],
                    padding_mode=padding_mode,
                    padding=[0, (kernel_size - 1) // 2],
                ),
            )
            block.add_module(
                f"bn_{i}",
                nn.BatchNorm2d(hc[i]),
            )
            block.add_module(f"act_{i}", activation)
            self.conv_part.add_module(f"block {i}", block)

    def forward(self, x: torch.Tensor):
        outputs = []
        for i, conv in enumerate(self.conv_part):
            x = conv(x)
            outputs.append(x)
        x = F.adaptive_avg_pool2d(x, output_size=[x.shape[-2], self.size_restriction])
        # print("x restriction", x.shape)
        # x = x.view(x.shape[0], -1)
        # l = self.latent_operator(x)
        l = x.squeeze(1)
        l = l.view(l.shape[0], l.shape[-2], -1)
        return l, outputs


class Decoder1D(nn.Module):
    def __init__(
        self,
        n_conv: int,
        activation: nn.Module,
        hc: List,
        out_channels: int,
        kernel_size: int,
        latent_dimension: int,
        n_layers: int,
        hidden_neurons: int,
        input_size: int,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.in_channels = hc[-1]
        self.kernel_size = kernel_size

        self.conv_part = nn.ModuleList()
        n_conv = len(hc)
        self.n_conv = n_conv
        block = nn.Sequential()
        block.add_module(
            f"conv_{-1}",
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=hc[-1],
                kernel_size=kernel_size,
                padding_mode="circular",
                padding=(kernel_size - 1) // 2,
            ),
        )
        block.add_module(
            f"bn_{-1}",
            nn.BatchNorm1d(hc[-1]),
        )
        block.add_module(f"act_{-1}", activation)
        self.conv_part.add_module(f"block -1", block)
        for i in range(n_conv - 1):
            block = nn.Sequential()
            block.add_module(
                f"conv_{i}",
                nn.Conv1d(
                    in_channels=hc[-i - 1],
                    out_channels=hc[-i - 2],
                    kernel_size=kernel_size,
                    padding_mode="circular",
                    padding=(kernel_size - 1) // 2,
                ),
            )
            block.add_module(
                f"bn_{i}",
                nn.BatchNorm1d(hc[-i - 1]),
            )
            block.add_module(f"act_{i}", activation)
            self.conv_part.add_module(f"block {i}", block)
        self.conv_part.add_module(
            f"block_{i+1}",
            nn.Conv1d(
                in_channels=hc[0],
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding_mode="circular",
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
            "final layer",
            nn.Linear(hidden_neurons, self.in_channels * (self.input_size)),
        )

    def forward(self, l: torch.Tensor, outputs: List):
        x = self.latent_operator(l)
        x = x.view(-1, self.in_channels, self.input_size)
        # print("decoder initial x", x.shape)
        for i, conv in enumerate(self.conv_part):
            # print("out shape", outputs[-i - 1].shape, "x", x.shape)
            if i < self.n_conv:
                x = conv(x + outputs[-i - 1])
            elif i == self.n_conv:
                x = conv(x)
        return x


class Decoder2D(nn.Module):
    def __init__(
        self,
        n_conv: int,
        activation: nn.Module,
        hc: List,
        out_channels: int,
        kernel_size: int,
        latent_dimension: int,
        n_layers: int,
        hidden_neurons: int,
        input_size: int,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.in_channels = hc[-1]
        self.kernel_size = kernel_size

        self.conv_part = nn.ModuleList()
        n_conv = len(hc)
        self.n_conv = n_conv
        block = nn.Sequential()
        block.add_module(
            f"conv_{-1}",
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=hc[-1],
                kernel_size=[1, kernel_size],
                padding_mode="circular",
                padding=[0, (kernel_size - 1) // 2],
            ),
        )
        block.add_module(
            f"bn_{-1}",
            nn.BatchNorm2d(hc[-1]),
        )
        block.add_module(f"act_{-1}", activation)
        self.conv_part.add_module(f"block -1", block)
        for i in range(n_conv - 1):
            block = nn.Sequential()
            block.add_module(
                f"conv_{i}",
                nn.Conv1d(
                    in_channels=hc[-i - 1],
                    out_channels=hc[-i - 2],
                    kernel_size=[1, kernel_size],
                    padding_mode="circular",
                    padding=[0, (kernel_size - 1) // 2],
                ),
            )
            block.add_module(
                f"bn_{i}",
                nn.BatchNorm2d(hc[-i - 1]),
            )
            block.add_module(f"act_{i}", activation)
            self.conv_part.add_module(f"block {i}", block)
        self.conv_part.add_module(
            f"block_{i+1}",
            nn.Conv2d(
                in_channels=hc[0],
                out_channels=out_channels,
                kernel_size=[1, kernel_size],
                padding_mode="circular",
                padding=[0, (kernel_size - 1) // 2],
            ),
        )

    def forward(self, l: torch.Tensor, outputs: List):
        # x = self.latent_operator(l)
        x = l.view(l.shape[0], self.in_channels, -1, self.input_size)
        # print("decoder initial x", x.shape)
        for i, conv in enumerate(self.conv_part):
            # print("out shape", outputs[-i - 1].shape, "x", x.shape)
            if i < self.n_conv:
                x = conv(x + outputs[-i - 1])
            elif i == self.n_conv:
                x = conv(x)
        return x
