import torch
import torch.nn as nn
from src.training.model_utils.utils_seq2seq import (
    EncoderSeq2Seq,
    DecoderOperator,
    ProbabilityHead,
)
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
        regularization: float,
    ) -> None:
        super().__init__()

        self.regularization: float = regularization

        self.encoder = EncoderSeq2Seq(
            n_conv=n_conv,
            hc=hc,
            in_channels=in_channel,
            out_channels=hc,
            kernel_size=kernel_size,
        )

        self.decoder = DecoderOperator(
            out_channels=out_channels,
            kernel_size=kernel_size,
            n_conv=n_conv,
            hc=hc,
            in_channels=in_channel,
        )

        # self.probability_head = ProbabilityHead(
        #     in_channels=hc,
        #     hidden_channels=hc,
        #     out_channels=out_channels,
        #     kernel_size=kernel_size,
        #     mask_type="B",
        # )

        self.loss = loss

    def forward(self, x: torch.Tensor, y: torch.Tensor):

        e = self.encoder(x)
        y_tilde = self.decoder(x=x, y=y, e=e)
        # mu, logsigma = self.probability_head(h)

        return y_tilde  # mu, logsigma

    def train_step(self, batch: Tuple, device: str):
        x, y = batch
        x = x.to(device=device, dtype=torch.double)
        # create some noise to improve the universality
        noise_mu = torch.zeros_like(x)
        noise_sigma = self.regularization * torch.ones_like(x)
        noise = torch.normal(noise_mu, noise_sigma)
        y = y.to(device=device, dtype=torch.double)
        y_input = y + noise
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        y_input = y_input.unsqueeze(1)
        y_tilde = self.forward(x=x, y=y_input).squeeze(1)
        # y_tilde = self.probability_head.training_sample(mu, logsigma)
        loss = self.loss(y_tilde, y.squeeze(1))
        return loss

    def valid_step(self, batch: Tuple, device: str):
        x, y = batch
        x = x.to(device=device, dtype=torch.double)
        # create some noise to improve the universality
        noise_mu = torch.zeros_like(x)
        noise_sigma = self.regularization * torch.ones_like(x)
        noise = torch.normal(noise_mu, noise_sigma)
        y = y.to(device=device, dtype=torch.double)
        y_input = y + noise
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        y_input = y_input.unsqueeze(1)
        y_tilde = self.forward(x=x, y=y_input).squeeze()
        # y_tilde = self.probability_head.training_sample(mu, logsigma)
        loss = self.loss(y_tilde, y.squeeze())
        return loss

    def prediction_step(self, time_step_initial: int, x: torch.Tensor, y: torch.Tensor):
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        print(y.shape)
        for t in range(time_step_initial, x.shape[-2]):
            y_sample = self.forward(x=x, y=y)
            y[:, 0, t, :] = y_sample[:, 0, t, :]

        y = y.squeeze()
        return y

    def get_sample(self, x: torch.Tensor, y: torch.Tensor):
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        mu, logsigma = self.forward(x=x, y=y)
        y_tilde = self.probability_head.training_sample(mu, logsigma)
        return y_tilde
