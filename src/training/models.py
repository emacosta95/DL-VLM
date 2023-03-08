# %% Libraries
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from typing import Tuple
from src.training.nn_blocks import ConvBlock
from tqdm import trange
import matplotlib.pyplot as plt


# class TDDFTRecurrent(nn.Module):
#     def __init__(
#         self,
#         n_conv_layers: int = None,
#         in_channels: int = None,
#         hidden_channels: list = None,
#         out_features: int = None,
#         ks: int = None,
#         padding_mode: str = None,
#         Activation: nn.Module = None,
#         Loss: nn.Module = None,
#     ) -> None:
#         """REconstruct DENsity profile via Transpose convolution

#         Argument:
#         n_conv_layers[int]: the number of layers of the architecture.
#         in_features [int]: the number of features of the input data.
#         in_channels[int]: the number of channels of the input data.
#         hidden_channels[list]: the list of hidden channels for each layer [C_1,C_2,...,C_N] with C_i referred to the i-th layer.
#         out_features[int]: the number of features of the output data
#         out_channels[int]: the number of channels of the output data.
#         ks[int]: the kernel size for each layer.
#         padding[int]: the list of padding for each layer.
#         padding_mode[str]: the padding_mode (according to the pytorch documentation) for each layer.
#         Activation[nn.Module]: the activation function that we adopt
#         n_block_layers[int]: number of conv layers for each norm
#         """

#         super().__init__()

#         self.potential_nn = TConvBlock(
#             n_conv=n_conv_layers,
#             activation=Activation,
#             hc=hidden_channels,
#             kernel_size=ks,
#             padding_mode=padding_mode,
#             in_channels=in_channels,
#             construction=False,
#         )
#         self.density_nn = TConvBlock(
#             n_conv=n_conv_layers,
#             activation=Activation,
#             hc=hidden_channels,
#             kernel_size=ks,
#             padding_mode=padding_mode,
#             in_channels=in_channels,
#             construction=False,
#         )
#         self.n_conv_layers = n_conv_layers
#         self.in_channels = in_channels
#         self.out_features = out_features
#         self.hidden_channels = hidden_channels
#         self.ks = ks
#         self.padding_mode = padding_mode
#         self.Activation = Activation
#         self.loss = Loss

#     def forward(self, v: torch.tensor, n_init: torch.tensor) -> torch.tensor:

#         # one for the image
#         v = v.unsqueeze(1)
#         x = torch.zeros_like(v).copy()
#         x[:, :, 0, :] = n_init
#         results = [n_init]
#         for i in range(v.shape[-2] - 1):
#             y = x + v[:, :, : i + 1]
#             # initial state information
#             r = self.potential_nn(y)
#             print(r.shape)
#             x = self.density_nn(r + x)
#             results.append()
#         print(x.shape)
#         x.squeeze(1)
#         return x

#     def train_step(self, batch: Tuple, device: str):
#         loss = 0
#         x, y = batch
#         x = x.to(device=device, dtype=torch.double)
#         # print(x.shape)
#         y = y.to(device=device, dtype=torch.double)
#         x = self.forward(x, y[:, 0])
#         loss = +self.loss(x, y)
#         return loss

#     def r2_computation(self, batch: Tuple, device: str, r2):
#         for i, bt in enumerate(batch):
#             x, y = bt
#             x = self.forward(x.to(dtype=torch.double, device=device))
#             y = y.double()
#             # print(y.shape,x.shape)
#             r2.update(x.cpu().detach().view(-1), y.cpu().detach().view(-1))
#         return r2

#     def save(
#         self,
#         path: str,
#         epoch: int = None,
#         dataset_name: str = None,
#         r_valid: float = None,
#         r_train: float = None,
#     ):
#         """the saving routine included into the Model class. We adopt the state dict mode in order to use a more flexible saving method
#         Arguments:
#         path[str]: the path of the torch.file
#         """
#         torch.save(
#             {
#                 "Activation": self.Activation,
#                 "n_conv_layers": self.n_conv_layers,
#                 "hidden_channels": self.hidden_channel,
#                 "in_features": self.in_features,
#                 "in_channels": self.in_channels,
#                 "out_features": self.out_features,
#                 "out_channels": self.out_channels,
#                 "padding": self.padding,
#                 "ks": self.ks,
#                 "padding_mode": self.padding_mode,
#                 "n_block_layers": self.n_block_layers,
#                 "model_state_dict": self.state_dict(),
#                 "epoch": epoch,
#                 "r_valid": r_valid,
#                 "r_train": r_train,
#                 "dataset_name": dataset_name,
#             },
#             path,
#         )

#     def load(self, path: str):
#         data = torch.load(path)
#         self.__init__(
#             n_conv_layers=data["n_conv_layers"],
#             in_features=data["in_features"],
#             in_channels=data["in_channels"],
#             hidden_channels=data["hidden_channels"],
#             out_features=data["out_features"],
#             out_channels=data["out_channels"],
#             ks=data["ks"],
#             padding=data["padding"],
#             padding_mode=data["padding_mode"],
#             Activation=data["Activation"],
#             n_block_layers=data["n_block_layers"],
#         )
#         print(
#             f"other information \n epochs={data['epoch']}, \n r_valid_value={data['r_valid']} and r_train_value={data['r_train']} on the dataset located in: {data['dataset_name']}"
#         )
#         self.load_state_dict(data["model_state_dict"])


class TDDFTCNN(nn.Module):
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
    ) -> None:
        """REconstruct DENsity profile via Transpose convolution

        Argument:
        n_conv_layers[int]: the number of layers of the architecture.
        in_features [int]: the number of features of the input data.
        in_channels[int]: the number of channels of the input data.
        hidden_channels[list]: the list of hidden channels for each layer [C_1,C_2,...,C_N] with C_i referred to the i-th layer.
        out_features[int]: the number of features of the output data
        out_channels[int]: the number of channels of the output data.
        ks[int]: the kernel size for each layer.
        padding[int]: the list of padding for each layer.
        padding_mode[str]: the padding_mode (according to the pytorch documentation) for each layer.
        Activation[nn.Module]: the activation function that we adopt
        n_block_layers[int]: number of conv layers for each norm
        """

        super().__init__()

        self.CNNBlock = ConvBlock(
            n_conv=len(hidden_channels),
            activation=Activation,
            hc=hidden_channels,
            in_channels=in_channels,
            padding_mode=padding_mode,
            kernel_size=ks,
        )
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
        x = self.CNNBlock(x)
        x.squeeze(1)
        return x

    def prediction(
        self,
        v: torch.tensor,
        n_init: torch.tensor,
        t_initial: int,
    ) -> torch.tensor:
        zero = torch.zeros_like(v)

        k2 = zero.unsqueeze(1).clone()
        k2[:, 0, 0, :] = n_init

        k1 = zero.unsqueeze(1).clone()
        k1[:, 0, 0, :] = v[:, 0, :]

        z = torch.cat((k1, k2), dim=1)
        # print(x)

        for t in trange(t_initial, v.shape[-2]):
            for i in trange(v.shape[-1]):
                new_n = self.CNNBlock(z)
                k2[:, :, t, i] = new_n[:, :, t, i]
                k1[:, 0, t, i] = v[:, t, i]
                z = torch.cat((k1, k2), dim=1)

        return k2.squeeze()

    def train_step(self, batch: Tuple, device: str):
        loss = 0
        x, y = batch
        x = x.to(device=device, dtype=torch.double)
        # print(x.shape)
        y = y.to(device=device, dtype=torch.double)
        x = self.forward(x)
        x = x.squeeze()
        y = y.squeeze()
        loss = +self.loss(x, y)
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


class TDDFTCNNNoMemory(nn.Module):
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
    ) -> None:
        """REconstruct DENsity profile via Transpose convolution

        Argument:
        n_conv_layers[int]: the number of layers of the architecture.
        in_features [int]: the number of features of the input data.
        in_channels[int]: the number of channels of the input data.
        hidden_channels[list]: the list of hidden channels for each layer [C_1,C_2,...,C_N] with C_i referred to the i-th layer.
        out_features[int]: the number of features of the output data
        out_channels[int]: the number of channels of the output data.
        ks[int]: the kernel size for each layer.
        padding[int]: the list of padding for each layer.
        padding_mode[str]: the padding_mode (according to the pytorch documentation) for each layer.
        Activation[nn.Module]: the activation function that we adopt
        n_block_layers[int]: number of conv layers for each norm
        """

        super().__init__()

        self.CNNBlock = ConvBlock(
            n_conv=len(hidden_channels),
            activation=Activation,
            hc=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            padding_mode=padding_mode,
            kernel_size=ks,
        )
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
        x = x.unsqueeze(1)
        x = self.CNNBlock(x)
        x.squeeze(1)
        return x

    def train_step(self, batch: Tuple, device: str):
        loss = 0
        x, y = batch
        x = x.to(device=device, dtype=torch.double)
        # print(x.shape)
        y = y.to(device=device, dtype=torch.double)
        x = self.forward(x)
        x = x.squeeze()
        y = y.squeeze()
        loss = +self.loss(x, y)
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
