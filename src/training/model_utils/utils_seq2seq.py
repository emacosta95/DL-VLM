import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from src.training.model_utils.cnn_causal_blocks import CausalConv2d, MaskedTimeConv2d
from typing import List, Optional


class EncoderSeq2Seq(nn.Module):
    def __init__(
        self,
        n_conv: int,
        hc: int,
        in_channels: int,
        out_channels: int,
        kernel_size: List,
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
                        hidden_channels=hc,
                        out_channels=out_channels,
                        activation=False,
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
class EncodeBlock(nn.Module):
    def __init__(
        self,
        kernel_size: List,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        activation: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.act_bool = activation
        if activation:
            self.activation = nn.GLU()
            self.out_channels = 2 * out_channels

        else:
            self.activation = nn.Identity()
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
            out_channels=self.out_channels,
            kernel_size=[kernel_size[0], 1],
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_space(x)
        x = self.conv_t(x)
        if self.act_bool:
            x = x.view(x.shape[0], x.shape[1] // 2, x.shape[-2], 2 * x.shape[-1])
        x = self.activation(x)

        return x


class DecodeBlock(nn.Module):
    def __init__(
        self,
        kernel_size: List,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        activation: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.act_bool = activation
        if activation:
            self.activation = nn.GLU()
            self.out_channels = 2 * out_channels

        else:
            self.activation = nn.Identity()
            self.out_channels = out_channels
        self.conv_space = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=[1, kernel_size[1]],
            padding=[0, (kernel_size[1] - 1) // 2],
            padding_mode="circular",
        )
        self.conv_t = MaskedTimeConv2d(
            in_channels=hidden_channels,
            out_channels=self.out_channels,
            kernel_size=[kernel_size[0], 1],
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_space(x)
        x = self.conv_t(x)
        if self.act_bool:
            x = x.view(x.shape[0], x.shape[1] // 2, x.shape[-2], 2 * x.shape[-1])
        x = self.activation(x)
        return x


class DecoderOperator(nn.Module):
    def __init__(
        self, out_channels: int, kernel_size: int, n_conv: int, hc: int
    ) -> None:
        super().__init__()

        self.n_conv = n_conv
        self.causal_embedding = MaskedTimeConv2d(
            in_channels=out_channels,
            out_channels=hc,
            kernel_size=[kernel_size[0], 1],
        )

        self.preprocessing_attention = nn.Conv2d(
            in_channels=hc,
            out_channels=hc,
            kernel_size=kernel_size,
            padding=[(kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2],
        )

        self.conv_part = nn.ModuleList()
        for i in range(n_conv):
            if i == 0:
                self.conv_part.append(
                    DecodeBlock(
                        kernel_size=kernel_size,
                        in_channels=hc,
                        hidden_channels=hc,
                        out_channels=hc,
                    )
                )
            elif (i != 0) and (i != n_conv - 1):
                self.conv_part.append(
                    DecodeBlock(
                        kernel_size=kernel_size,
                        in_channels=hc,
                        hidden_channels=hc,
                        out_channels=hc,
                    )
                )

            elif i == n_conv - 1:
                self.conv_part.append(
                    DecodeBlock(
                        kernel_size=kernel_size,
                        in_channels=hc,
                        hidden_channels=hc,
                        out_channels=out_channels,
                        activation=False,
                    )
                )

    def attention(self, e: torch.Tensor, h: torch.Tensor, x: torch.Tensor):
        a = torch.einsum("bhti,bhri->bhtr", e, h)
        c = F.softmax(a, dim=-1)
        c = torch.einsum("bhtr,bhri->bhti", c, (e + x))
        return c

    def forward(self, y: torch.Tensor, x: torch.Tensor, e: torch.Tensor):

        y = self.causal_embedding(y)
        for i, block in enumerate(self.conv_part):
            if i == 0:
                h = block(y)
            elif i != 0 and i < self.n_conv - 1:
                d = self.preprocessing_attention(h) + y
                c = self.attention(e=e, h=d, x=x)
                h = h + c
                h = block(h) + h
            elif i == self.n_conv - 1:
                y_tilde = block(h)
        return y_tilde
