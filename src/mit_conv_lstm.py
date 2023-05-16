from src.training.model_utils.mit_cnn_lstm import ConvLSTM
import torch
import torch.nn as nn
from typing import Tuple, List


class MITConvLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        n_conv,
        Loss: nn.Module,
    ) -> None:
        super().__init__()
        self.loss = Loss
        self.cnnlstm = ConvLSTM(
            input_dim,
            hidden_dim=hidden_dim,
            kernel_size=(1, kernel_size),
            num_layers=n_conv,
            batch_first=True,
        )

    def forward(self, b: torch.Tensor):
        return self.cnnlstm(b)

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
