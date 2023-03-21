import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from src.training.model_utils.cnn_causal_blocks import CausalConv2d
from typing import List


class EncoderSeq2Seq(nn.Module):
    def __init__(
        self,
        n_conv: int,
        hc: int,
        in_channels: int,
        out_channels: int,
        kernel_size: List,
        padding_mode: str,
    ) -> None:
        super().__init__()

        self.conv_part = nn.ModuleList()
        self.n_conv = n_conv

        for i in range(n_conv):
            if i == 0:
                self.conv_part.append(
                    EncodeBlock(
                        kernel_size=kernel_size,
                        in_channels=in_channels,
                        hidden_channels=hc,
                        out_channels=hc,
                    )
                )
            elif (i != 0) and (i != n_conv - 1):
                self.conv_part.append(
                    EncodeBlock(
                        kernel_size=kernel_size,
                        in_channels=hc,
                        hidden_channels=hc,
                        out_channels=hc,
                    )
                )

            elif i == n_conv - 1:
                self.conv_part.append(
                    EncodeBlock(
                        kernel_size=kernel_size,
                        in_channels=hc,
                        hidden_channels=out_channels,
                        out_channels=out_channels,
                    )
                )

    def forward(self, x: torch.Tensor):
        for i, block in enumerate(self.conv_part):
            if i != 0 and i != self.n_conv - 1:
                x = block(x) + x  # skipped connection
            else:
                x = block(x)
        return x


# We have to write the decode block
class DecodeBlock(nn.Module):
    def __init__(
        self,
        kernel_size: List,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.out_channels = out_channels
        self.conv_space = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=[1, kernel_size[1]],
            padding=[0, (kernel_size[1] - 1) // 2],
            padding_mode="circular",
        )
        self.conv_t = CausalConv2d(
            in_channels=hidden_channels,
            out_channels=2 * out_channels,
            kernel_size=[kernel_size[0], 1],
        )
        self.activation = nn.GLU()

    def forward(self, x: torch.Tensor):
        x = self.conv_space(x)
        x = self.conv_t(x)
        x = x.view(-1, self.out_channels, x.shape[-2], 2 * x.shape[-1])
        x = self.activation(x)
        return x
