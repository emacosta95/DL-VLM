import torch
import torch.nn as nn
from src.training.model_utils.utils_seq2seq import (
    EncoderSeq2Seq,
    DecoderOperator,
    ProbabilityHead,
)
from src.training.model_utils.lstm_cnn import Encoder1D, Decoder1D, Encoder2D, Decoder2D
from typing import Tuple, List


class UnetLSTM(nn.Module):
    def __init__(
        self,
        activation: nn.Module,
        hc: List,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        latent_dimension: int,
        n_layers: int,
        hidden_neurons: int,
        input_size: int,
        Loss: nn.Module,
        lstm_layers: int,
    ) -> None:
        super().__init__()

        self.loss = Loss
        self.lstm_layers = lstm_layers
        n_conv = len(hc)

        self.encoder = Encoder1D(
            n_conv=n_conv,
            activation=activation,
            hc=hc,
            in_channels=in_channels,
            kernel_size=kernel_size,
            padding_mode="circular",
            latent_dimension=latent_dimension,
            n_layers=n_layers,
            pooling_size=None,
            hidden_neurons=hidden_neurons,
            size_restriction=input_size,
        )

        self.decoder = Decoder1D(
            n_conv=n_conv,
            activation=activation,
            hc=hc,
            out_channels=in_channels,
            kernel_size=kernel_size,
            latent_dimension=latent_dimension,
            n_layers=n_layers,
            hidden_neurons=hidden_neurons,
            input_size=input_size,
        )

        self.lstm = nn.LSTM(
            input_size=latent_dimension,
            hidden_size=hidden_neurons,
            num_layers=self.lstm_layers,
            proj_size=latent_dimension,
            batch_first=True,
        )

    def forward(self, b: torch.Tensor):
        t = b.shape[-2]
        bs = b.shape[0]
        # batch for t
        b = b.contiguous().view(t * bs, -1)
        # normalize
        b = (b - b.mean()) / b.std()
        b = b.unsqueeze(1)
        # print("b shape", b.shape)
        lt, outputs = self.encoder(b)
        lt = lt.view(bs, t, -1)
        # print("lt_shape part 1", lt.shape)
        lt, _ = self.lstm(lt)
        lt = lt.reshape(bs * t, -1)
        # print("lt shape", lt.shape)
        z = self.decoder(lt, outputs)
        z = z.squeeze(1)
        z = z.view(bs, t, -1)
        return z

    def train_step(self, batch: Tuple, device: str):
        x, y = batch
        x = x.to(device=device, dtype=torch.double)
        # create some noise to improve the universality
        y = y.to(device=device, dtype=torch.double)
        y_tilde = self.forward(x)
        # y_tilde = self.probability_head.training_sample(mu, logsigma)
        loss = self.loss(y_tilde, y)
        return loss

    def valid_step(self, batch: Tuple, device: str):
        x, y = batch
        x = x.to(device=device, dtype=torch.double)
        # create some noise to improve the universality
        y = y.to(device=device, dtype=torch.double)
        y_tilde = self.forward(x)
        # y_tilde = self.probability_head.training_sample(mu, logsigma)
        loss = self.loss(y_tilde, y)
        return loss


class UnetLSTM_beta(nn.Module):
    def __init__(
        self,
        activation: nn.Module,
        hc: List,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        latent_dimension: int,
        n_layers: int,
        hidden_neurons: int,
        input_size: int,
        Loss: nn.Module,
        lstm_layers: int,
    ) -> None:
        super().__init__()

        self.loss = Loss
        self.lstm_layers = lstm_layers
        n_conv = len(hc)
        self.hc = hc

        self.encoder = Encoder2D(
            n_conv=n_conv,
            activation=activation,
            hc=hc,
            in_channels=in_channels,
            kernel_size=kernel_size,
            padding_mode="circular",
            latent_dimension=latent_dimension,
            n_layers=n_layers,
            pooling_size=None,
            hidden_neurons=hidden_neurons,
            size_restriction=input_size,
        )

        self.decoder = Decoder2D(
            n_conv=n_conv,
            activation=activation,
            hc=hc,
            out_channels=in_channels,
            kernel_size=kernel_size,
            latent_dimension=latent_dimension,
            n_layers=n_layers,
            hidden_neurons=hidden_neurons,
            input_size=input_size,
        )

        self.lstm = nn.LSTM(
            input_size=latent_dimension * hc[-1],
            hidden_size=hidden_neurons,
            num_layers=self.lstm_layers,
            proj_size=latent_dimension,
            batch_first=True,
        )

    def forward(self, b: torch.Tensor):
        t = b.shape[-2]
        bs = b.shape[0]
        # batch for t
        # normalize
        b = b.unsqueeze(1)
        # print("b shape", b.shape)
        lt, outputs = self.encoder(b)
        # lt = lt.view(bs, t, -1)
        # print("lt_shape part 1", lt.shape)
        lt, _ = self.lstm(lt)
        lt = lt.unsqueeze(1)
        lt = lt.expand(-1, self.hc[-1], -1, -1)
        # lt = lt.reshape(bs * t, -1)
        # print("lt shape", lt.shape)
        z = self.decoder(lt, outputs)
        z = z.squeeze(1)
        # z = z.view(bs, t, -1)
        return z

    def train_step(self, batch: Tuple, device: str):
        x, y = batch
        x = x.to(device=device, dtype=torch.double)
        # create some noise to improve the universality
        y = y.to(device=device, dtype=torch.double)
        y_tilde = self.forward(x)
        # y_tilde = self.probability_head.training_sample(mu, logsigma)
        loss = self.loss(y_tilde, y)
        return loss

    def valid_step(self, batch: Tuple, device: str):
        x, y = batch
        x = x.to(device=device, dtype=torch.double)
        # create some noise to improve the universality
        y = y.to(device=device, dtype=torch.double)
        y_tilde = self.forward(x)
        # y_tilde = self.probability_head.training_sample(mu, logsigma)
        loss = self.loss(y_tilde, y)
        return loss
