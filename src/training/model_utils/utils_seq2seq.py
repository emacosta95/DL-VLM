import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from src.training.model_utils.cnn_causal_blocks import (
    CausalConv2d,
    MaskedTimeConv2d,
    GatedConv2D,
)
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
            self.activation = nn.Mish()
            self.out_channels = out_channels

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
        # if self.act_bool:
        #     x = x.view(x.shape[0], x.shape[1] // 2, x.shape[-2], 2 * x.shape[-1])
        x = self.activation(x)

        return x


class DecodeBlock(nn.Module):
    def __init__(
        self,
        kernel_size: List,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        mask_type: str,
        activation: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.act_bool = activation
        if activation:
            self.activation = nn.Mish()
            self.out_channels = out_channels

        else:
            self.activation = nn.Identity()
            self.out_channels = out_channels
        # self.conv_space = nn.Conv2d(
        #     in_channels=in_channels,
        #     out_channels=hidden_channels,
        #     kernel_size=[1, kernel_size[1]],
        #     padding=[0, (kernel_size[1] - 1) // 2],
        #     padding_mode="circular",
        # )
        # self.conv_t = MaskedTimeConv2d(
        #     in_channels=hidden_channels,
        #     out_channels=self.out_channels,
        #     kernel_size=[kernel_size[0], 1],
        # )

        self.conv = GatedConv2D(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            mask_type=mask_type,
        )

    def forward(self, x: torch.Tensor):
        # x = self.conv_space(x)
        # x = self.conv_t(x)
        # if self.act_bool:
        #     x = x.view(x.shape[0], x.shape[1] // 2, x.shape[-2], 2 * x.shape[-1])
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderOperator(nn.Module):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        n_conv: int,
        hc: int,
        in_channels: int,
    ) -> None:
        super().__init__()

        self.n_conv = n_conv
        self.causal_embedding = GatedConv2D(
            in_channels=in_channels,
            hidden_channels=hc,
            out_channels=hc,
            kernel_size=kernel_size,
            mask_type="A",
        )

        self.preprocessing_attention = GatedConv2D(
            in_channels=hc,
            hidden_channels=hc,
            out_channels=hc,
            kernel_size=kernel_size,
            mask_type="B",
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
                        mask_type="A",
                    )
                )
            elif (i != 0) and (i != n_conv - 1):
                self.conv_part.append(
                    DecodeBlock(
                        kernel_size=kernel_size,
                        in_channels=hc,
                        hidden_channels=hc,
                        out_channels=hc,
                        mask_type="B",
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
                        mask_type="B",
                    )
                )

    def attention(self, e: torch.Tensor, d: torch.Tensor, x: torch.Tensor):

        a = torch.einsum("bhti,bhri->bhtr", d, e)  # causal effect
        mask = torch.ones_like(a)
        mask[:, :, e.shape[-2] // 2 + 1 :, e.shape[-2] // 2 + 1 :] = 0.0
        a = a * mask
        c = F.softmax(a, dim=-1)
        c = torch.einsum("bhtr,bhri->bhti", c, (e + x))
        return c

    def forward(self, y: torch.Tensor, x: torch.Tensor, e: torch.Tensor):

        r = self.causal_embedding(y)  # autoregressive property checked
        for i, block in enumerate(self.conv_part):
            if i == 0:
                h = block(r)
            elif i != 0 and i < self.n_conv - 1:
                d = self.preprocessing_attention(h) + r
                c = self.attention(e=e, d=d, x=x)
                h = h + c
                h = block(h) + h
            elif i == self.n_conv - 1:
                y_tilde = block(h)
        return y_tilde


class ProbabilityHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        mask_type: int,
    ) -> None:
        super().__init__()

        self.conv_mu = GatedConv2D(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            mask_type=mask_type,
        )
        self.conv_logsigma = GatedConv2D(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            mask_type=mask_type,
        )

    def forward(self, h: torch.Tensor):
        mu = self.conv_mu(h).squeeze(1)
        logsigma = self.conv_logsigma(h).squeeze(1)
        return mu, logsigma

    def training_sample(self, mu: torch.Tensor, logsigma: torch.Tensor):
        std = (logsigma * 0.5).exp()
        return torch.distributions.Normal(loc=mu, scale=std).rsample()

    def prediction_sample(
        self,
        mu: torch.Tensor,
        logsigma: torch.Tensor,
        space_index: int,
        time_index: int,
    ):
        std = (logsigma * 0.5).exp()
        return torch.distributions.Normal(
            loc=mu[:, time_index, space_index],
            scale=std[:, time_index, space_index],
        ).rsample()
