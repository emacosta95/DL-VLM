import torch
import torch.nn as nn
from src.training.model_utils.utils_seq2seq import EncoderSeq2Seq, DecoderOperator
from typing import Tuple


class Seq2Seq(nn.Module):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        n_conv: int,
        hc: int,
        in_channel: int,
        loss: nn.Module,
    ) -> None:
        super().__init__()

        self.encoder = EncoderSeq2Seq(
            n_conv=n_conv,
            hc=hc,
            in_channels=in_channel,
            out_channels=hc,
            kernel_size=kernel_size,
        )

        self.decoder = DecoderOperator(
            out_channels=out_channels, kernel_size=kernel_size, n_conv=n_conv, hc=hc
        )

        self.loss = loss

    def forward(self, x: torch.Tensor, y: torch.Tensor):

        e = self.encoder(x)
        y_tilde = self.decoder(x=x, y=y, e=e)

        return y_tilde

    def train_step(self, batch: Tuple, device: str):
        x, y = batch
        x = x.to(device=device, dtype=torch.double)
        y = y.to(device=device, dtype=torch.double)
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        y_tilde = self.forward(x, y)
        loss = self.loss(y_tilde.squeeze(), y.squeeze())
        return loss

    def valid_step(self, batch: Tuple, device: str):
        x, y = batch
        x = x.to(device=device, dtype=torch.double)
        y = y.to(device=device, dtype=torch.double)
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        y_tilde = self.forward(x, y)
        loss = self.loss(y_tilde.squeeze(), y.squeeze())
        return loss

    def prediction_step(self, x: torch.Tensor, y: torch.Tensor):
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        for t in range(x.shape[-2] - 1):
            y_tilde = self.forward(x, y).squeeze()
            y = y_tilde[:, :, t + 1]
        y = y.squeeze()
        return y
