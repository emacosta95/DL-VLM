import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from src.training.model_utils.cnn_causal_blocks import MaskedConv2d, CausalConv2d


class ConvBlock(nn.Module):
    def __init__(
        self,
        n_conv: int,
        activation: nn.Module,
        hc: int,
        in_channels: int,
        kernel_size: int,
        padding_mode: str,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential()
        for i in range(n_conv):

            if i == 0:
                self.block.add_module(
                    f"conv_{i}",
                    CausalConv2d(
                        in_channels=in_channels,
                        out_channels=hc[i],
                        kernel_size=kernel_size,
                    ),
                )
            else:
                self.block.add_module(
                    f"conv_{i}",
                    CausalConv2d(
                        in_channels=hc[i - 1],
                        out_channels=hc[i],
                        kernel_size=kernel_size,
                    ),
                )
            self.block.add_module(f"act_{i}", activation)

        self.block.add_module(
            f"conv_{i+1}",
            CausalConv2d(
                in_channels=hc[i],
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
        )

    def forward(self, x: torch.Tensor):
        x = self.block(x)
        # x = torch.cos(x)  # values between -1 and 1
        return x


class ConvBlock1D(nn.Module):
    def __init__(
        self,
        n_conv: int,
        activation: nn.Module,
        hc: int,
        in_channels: int,
        kernel_size: int,
        padding_mode: str,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential()
        for i in range(n_conv):

            if i == 0:
                self.block.add_module(
                    f"conv_{i}",
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=hc[i],
                        kernel_size=kernel_size,
                        padding_mode=padding_mode,
                        padding=(kernel_size - 1) // 2,
                    ),
                )
            else:
                self.block.add_module(
                    f"conv_{i}",
                    nn.Conv1d(
                        in_channels=hc[i - 1],
                        out_channels=hc[i],
                        kernel_size=kernel_size,
                        padding_mode=padding_mode,
                        padding=(kernel_size - 1) // 2,
                    ),
                )
            self.block.add_module(f"act_{i}", activation)

        self.block.add_module(
            f"conv_{i}",
            nn.Conv1d(
                in_channels=hc[i],
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding_mode=padding_mode,
                padding=(kernel_size - 1) // 2,
            ),
        )

    def forward(self, x: torch.Tensor):
        x = self.block(x)
        # x = torch.cos(x)  # values between -1 and 1
        return x
