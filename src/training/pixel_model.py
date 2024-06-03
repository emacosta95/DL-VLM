# %% Libraries
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from typing import Tuple
from src.training.conv_block_model import ConvBlock, PixelConv

from src.training.model_utils.cnn_causal_blocks import (
    CausalConv2d,
)
from tqdm import trange
import matplotlib.pyplot as plt


class PixelCNN(nn.Module):
    def __init__(
        self,
        in_features: int = None,
        in_channels: int = None,
        hidden_channels: list = None,
        out_features: int = None,
        out_channels: int = None,
        ks: int = None,
        padding: int = None,
        padding_mode: str = None,
        pooling_size: int = None,
        Activation: nn.Module = None,
        Loss: nn.Module = None,
        t_interval_range: int = None,
    ) -> None:
        super().__init__()

        self.t_interval_range = t_interval_range
        self.kernel_size = ks[0]

        self.CNNBlock = ConvBlock(
            n_conv=len(hidden_channels),
            activation=Activation,
            hc=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            padding_mode=padding_mode,
            kernel_size=ks,
        )

        self.PixelCONV_initial = PixelConv(
            n_conv=len(hidden_channels),
            activation=Activation,
            hc=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            padding_mode=padding_mode,
            kernel_size=ks,
            initial_block=True,
        )

        self.PixelCONV_final = PixelConv(
            n_conv=len(hidden_channels),
            activation=Activation,
            hc=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            padding_mode=padding_mode,
            kernel_size=ks,
            initial_block=False,
        )

        self.Gated = nn.GLU(dim=-1)

        self.n_conv_layers = len(hidden_channels)
        self.in_features = in_features
        self.in_channels = in_channels
        self.out_features = out_features
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.ks = ks
        self.padding = padding
        self.padding_mode = padding_mode
        self.Activation = Activation
        self.loss = Loss
        self.pooling_size = pooling_size

    def forward(self, x: torch.tensor) -> torch.tensor:

        h_eff = x[:, 0].unsqueeze(1)
        h = x[:, 1].unsqueeze(1)

        h = self.CNNBlock(h)
        h_eff = self.PixelCONV_initial(h_eff)

        y = torch.cat((h_eff, h), axis=-1)

        y = self.Gated(y)

        y = self.PixelCONV_final(y)
        return y.squeeze()

    def train_step(self, batch: Tuple, device: str):
        loss = 0
        x, y = batch
        x = x.to(device=device)
        x = x[:, :, :, : self.t_interval_range]  # + noise
        y = y.to(device=device)
        y = y[:, :, : self.t_interval_range]

        y_hat = self.forward(x.double())
        y_hat = y_hat.squeeze()
        y = y.squeeze()
        loss = +self.loss(y_hat, y)
        return loss

    def valid_step(self, batch: Tuple, device: str):
        loss = 0
        x, y = batch

        x = x.to(device=device)
        x = x[:, :, :, : self.t_interval_range]
        y = y.to(device=device)
        y = y[:, :, : self.t_interval_range]
        y_hat = self.forward(x.double())
        y_hat = y_hat.squeeze()
        y = y.squeeze()
        loss = +self.loss(y_hat, y)
        return loss

    def save(
        self,
        path: str,
        epoch: int = None,
        dataset_name: str = None,
        r_valid: float = None,
        r_train: float = None,
    ):
        """the saving routine included into the Model class. We adopt the state dict mode in order to use a more flexible saving method
        Arguments:
        path[str]: the path of the torch.file
        """
        torch.save(
            {
                "Activation": self.Activation,
                "n_conv_layers": self.n_conv_layers,
                "hidden_channels": self.hidden_channel,
                "in_features": self.in_features,
                "in_channels": self.in_channels,
                "out_features": self.out_features,
                "out_channels": self.out_channels,
                "padding": self.padding,
                "ks": self.ks,
                "padding_mode": self.padding_mode,
                "n_block_layers": self.n_block_layers,
                "model_state_dict": self.state_dict(),
                "epoch": epoch,
                "r_valid": r_valid,
                "r_train": r_train,
                "dataset_name": dataset_name,
            },
            path,
        )

    def load(self, path: str):
        data = torch.load(path)
        self.__init__(
            n_conv_layers=data["n_conv_layers"],
            in_features=data["in_features"],
            in_channels=data["in_channels"],
            hidden_channels=data["hidden_channels"],
            out_features=data["out_features"],
            out_channels=data["out_channels"],
            ks=data["ks"],
            padding=data["padding"],
            padding_mode=data["padding_mode"],
            Activation=data["Activation"],
            n_block_layers=data["n_block_layers"],
        )
        print(
            f"other information \n epochs={data['epoch']}, \n r_valid_value={data['r_valid']} and r_train_value={data['r_train']} on the dataset located in: {data['dataset_name']}"
        )
        self.load_state_dict(data["model_state_dict"])
