import torch
import torch.nn as nn
from torch.nn import functional as F
from src.training.model_utils.cnn_causal_blocks import CausalConv2d
from typing import List


class ConvBlock(nn.Module):
    def __init__(
        self,
        n_conv: int,
        activation: nn.Module,
        hc: List[int],
        in_channels: int,
        kernel_size: int,
        padding_mode: str,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential()
        self.block.add_module(
            f"conv_sp {-1}",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hc[0],
                kernel_size=[kernel_size[0], 1],
                padding_mode=padding_mode,
                padding=[(kernel_size[0] - 1) // 2, 0],
            ),
        )
        self.block.add_module(
            f"conv_t {-1}",
            CausalConv2d(
                in_channels=hc[0],
                out_channels=hc[0],
                kernel_size=[1, kernel_size[1]],
                bias=False,
            ),
        )

        self.block.add_module(
            f"batchnorm_{-1}",
            nn.BatchNorm2d(hc[0]),
        )
        for i in range(n_conv - 1):
            self.block.add_module(
                f"conv_sp{i}",
                nn.Conv2d(
                    in_channels=hc[i - 1],
                    out_channels=hc[i],
                    kernel_size=[kernel_size[0], 1],
                    padding_mode=padding_mode,
                    padding=[(kernel_size[0] - 1) // 2, 0],
                ),
            )
            self.block.add_module(
                f"conv_t{i}",
                CausalConv2d(
                    in_channels=hc[i],
                    out_channels=hc[i],
                    kernel_size=[1, kernel_size[1]],
                    bias=False,
                ),
            )

            self.block.add_module(
                f"batchnorm_{i}",
                nn.BatchNorm2d(hc[i]),
            )
            self.block.add_module(f"act_{i}", activation)

        self.block.add_module(
            f"conv_sp{n_conv+1}",
            nn.Conv2d(
                in_channels=hc[i],
                out_channels=hc[i],
                kernel_size=[kernel_size[0], 1],
                padding_mode=padding_mode,
                padding=[(kernel_size[0] - 1) // 2, 0],
            ),
        )
        self.block.add_module(
            f"conv_t{n_conv+1}",
            CausalConv2d(
                in_channels=hc[i],
                out_channels=out_channels,
                kernel_size=[1, kernel_size[1]],
                bias=False,
            ),
        )

    def forward(self, x: torch.Tensor):
        return self.block(x)


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
