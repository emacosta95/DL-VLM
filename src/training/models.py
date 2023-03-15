# %% Libraries
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from typing import Tuple
from src.training.conv_block_model import (
    ConvBlock,
)
from src.training.model_utils.cnn_causal_blocks import (
    CausalConv2d,
    MaskedConv2d,
    MaskedTConv2d,
)
from tqdm import trange
import matplotlib.pyplot as plt


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
        t_interval_range: int = None,
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

        t_int = np.random.randint(self.kernel_size + 1, self.t_interval_range, size=1)[
            0
        ]
        print(t_int)
        x = x.to(device=device, dtype=torch.double)
        x = x[:, :t_int]
        y = y.to(device=device, dtype=torch.double)
        y = y[:, :t_int]
        x = self.forward(x)
        x = x.squeeze()
        y = y.squeeze()
        loss = +self.loss(x, y)
        return loss

    def train_step(self, batch: Tuple, device: str):
        loss = 0
        x, y = batch

        t_int = np.random.randint(5, self.t_interval_range, size=1)[0]
        x = x.to(device=device, dtype=torch.double)
        x = x[:, :t_int]
        y = y.to(device=device, dtype=torch.double)
        y = y[:, :t_int]
        x = self.forward(x)
        x = x.squeeze()
        y = y.squeeze()
        loss = +self.loss(x, y)
        return loss

    def valid_step(self, batch: Tuple, device: str):
        loss = 0
        x, y = batch

        x = x.to(device=device, dtype=torch.double)
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


class Causal_REDENT2D(nn.Module):
    def __init__(
        self,
        n_conv_layers: int = None,
        in_features: int = None,
        in_channels: int = None,
        hidden_channels: list = None,
        out_features: int = None,
        out_channels: int = None,
        ks: int = None,
        padding: int = None,
        padding_mode: str = None,
        Activation: nn.Module = None,
        n_block_layers: int = None,
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

        self.conv_downsample = nn.ModuleList()
        self.conv_upsample = nn.ModuleList()
        self.n_conv_layers = n_conv_layers
        self.in_features = in_features
        self.in_channels = in_channels
        self.out_features = out_features
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.ks = ks
        self.padding = padding
        self.padding_mode = padding_mode
        self.Activation = Activation
        self.n_block_layers = n_block_layers
        self.loss = Loss
        if self.n_conv_layers != None:
            for i in range(n_conv_layers):
                if i == 0:
                    block = nn.Sequential()
                    block.add_module(
                        f"conv{i+1}",
                        CausalConv2d(
                            dilation=1,
                            stride=1,
                            in_channels=in_channels,
                            out_channels=hidden_channels[i],
                            kernel_size=ks,
                        ),
                    )
                    block.add_module(
                        f"batch_norm {i+1}", nn.BatchNorm2d(hidden_channels[i])
                    )
                    block.add_module(f"activation {i+1}", self.Activation)
                    for j in range(self.n_block_layers):
                        block.add_module(
                            f"conv_{i+1}_{j+1}",
                            CausalConv2d(
                                dilation=1,
                                stride=1,
                                in_channels=self.hidden_channels[i],
                                out_channels=self.hidden_channels[i],
                                kernel_size=ks,
                            ),
                        )
                        block.add_module(
                            f"batch_norm {i+1}_{j+1}",
                            nn.BatchNorm2d(self.hidden_channels[i]),
                        )
                        block.add_module(f"activation_{i+1}_{j+1}", self.Activation)
                    block.add_module(f"pooling {i+1}", nn.AvgPool2d(kernel_size=[2, 1]))
                    self.conv_downsample.append(block)

                elif (i > 0) and (i < n_conv_layers - 1):
                    block = nn.Sequential()
                    block.add_module(
                        f"conv{i+1}",
                        CausalConv2d(
                            dilation=1,
                            stride=1,
                            in_channels=hidden_channels[i - 1],
                            out_channels=hidden_channels[i],
                            kernel_size=ks,
                        ),
                    )
                    # block.add_module(
                    #    f"batch_norm {i+1}", nn.BatchNorm2d(hidden_channels[i])
                    # )
                    block.add_module(f"activation {i+1}", self.Activation)
                    for j in range(self.n_block_layers):
                        block.add_module(
                            f"conv_{i+1}_{j+1}",
                            CausalConv2d(
                                dilation=1,
                                stride=1,
                                in_channels=self.hidden_channels[i],
                                out_channels=self.hidden_channels[i],
                                kernel_size=ks,
                            ),
                        )
                        # block.add_module(
                        #     f"batch_norm {i+1}_{j+1}",
                        #     nn.BatchNorm2d(self.hidden_channels[i]),
                        # )
                        block.add_module(f"activation_{i+1}_{j+1}", self.Activation)
                    block.add_module(f"pooling {i+1}", nn.AvgPool2d(kernel_size=[2, 1]))
                    self.conv_downsample.append(block)
                elif i == n_conv_layers - 1:

                    block = nn.Sequential()

                    block.add_module(
                        f"conv{i+1}",
                        CausalConv2d(
                            dilation=1,
                            stride=1,
                            in_channels=hidden_channels[i - 1],
                            out_channels=hidden_channels[i],
                            kernel_size=ks,
                        ),
                    )
                    # block.add_module(
                    #     f"batch_norm {i+1}", nn.BatchNorm2d(hidden_channels[i])
                    # )
                    block.add_module(f"activation_{i+1}", self.Activation)

                    for j in range(self.n_block_layers):

                        block.add_module(
                            f"conv_{i+1}_{j+1}",
                            CausalConv2d(
                                dilation=1,
                                stride=1,
                                in_channels=self.hidden_channels[i],
                                out_channels=self.hidden_channels[i],
                                kernel_size=ks,
                            ),
                        )
                        block.add_module(
                            f"batch_norm {i+1}_{j+1}",
                            nn.BatchNorm2d(self.hidden_channels[i]),
                        )
                        block.add_module(f"activation_{i+1}_{j+1}", self.Activation)
                    block.add_module(f"pooling {i+1}", nn.AvgPool2d(kernel_size=[2, 1]))
                    self.conv_downsample.append(block)

            # We should add a final block of dense layers

            for i in range(self.n_conv_layers):
                if i == 0:
                    block = nn.Sequential()
                    block.add_module(
                        f"trans_conv{i+1}",
                        nn.ConvTranspose2d(
                            stride=(2, 1),
                            in_channels=hidden_channels[n_conv_layers - 1 - i],
                            out_channels=hidden_channels[n_conv_layers - 1 - (i + 1)],
                            kernel_size=[ks[0] + 1, ks[1]],
                            padding=padding,
                            padding_mode="zeros",
                        ),
                    )
                    # block.add_module(
                    #     f"batch_norm {i+1}",
                    #     nn.BatchNorm2d(
                    #         self.hidden_channels[n_conv_layers - 1 - (i + 1)]
                    #     ),
                    # )
                    block.add_module(f"activation {i+1}", self.Activation)
                    for j in range(self.n_block_layers):
                        block.add_module(
                            f"conv_{i+1}_{j+1}",
                            nn.ConvTranspose2d(
                                dilation=1,
                                stride=1,
                                in_channels=self.hidden_channels[
                                    n_conv_layers - 1 - (i + 1)
                                ],
                                out_channels=self.hidden_channels[
                                    n_conv_layers - 1 - (i + 1)
                                ],
                                kernel_size=ks,
                            ),
                        )
                        # block.add_module(
                        #     f"batch_norm {i+1}_{j+1}",
                        #     nn.BatchNorm2d(
                        #         self.hidden_channels[n_conv_layers - 1 - (i + 1)]
                        #     ),
                        # )
                        block.add_module(f"activation_{i+1}_{j+1}", self.Activation)
                    self.conv_upsample.append(block)
                elif (i > 0) and (i < n_conv_layers - 1):
                    block = nn.Sequential()
                    block.add_module(
                        f"trans_conv{i+1}",
                        nn.ConvTranspose2d(
                            stride=(2, 1),
                            in_channels=hidden_channels[n_conv_layers - 1 - (i)],
                            out_channels=hidden_channels[n_conv_layers - 1 - (i + 1)],
                            kernel_size=[ks[0] + 1, ks[1]],
                            padding=padding,
                            padding_mode="zeros",
                        ),
                    )
                    # block.add_module(
                    #     f"batch_norm {i+1}",
                    #     nn.BatchNorm2d(
                    #         self.hidden_channels[n_conv_layers - 1 - (i + 1)]
                    #     ),
                    # )
                    block.add_module(f"activation {i+1}", self.Activation)
                    for j in range(self.n_block_layers):
                        block.add_module(
                            f"conv_{i+1}_{j+1}",
                            CausalConv2d(
                                dilation=1,
                                stride=1,
                                in_channels=self.hidden_channels[
                                    n_conv_layers - 1 - (i + 1)
                                ],
                                out_channels=self.hidden_channels[
                                    n_conv_layers - 1 - (i + 1)
                                ],
                                kernel_size=ks,
                            ),
                        )
                        block.add_module(
                            f"batch_norm {i+1}_{j+1}",
                            nn.BatchNorm2d(
                                self.hidden_channels[n_conv_layers - 1 - (i + 1)]
                            ),
                        )
                        block.add_module(f"activation_{i+1}_{j+1}", self.Activation)
                    self.conv_upsample.append(block)
                elif i == n_conv_layers - 1:
                    block = nn.Sequential()
                    for j in range(self.n_block_layers):
                        block.add_module(
                            f"conv_{i+1}_{j+1}",
                            CausalConv2d(
                                dilation=1,
                                stride=1,
                                in_channels=self.hidden_channels[
                                    n_conv_layers - 1 - (i)
                                ],
                                out_channels=self.hidden_channels[
                                    n_conv_layers - 1 - (i)
                                ],
                                kernel_size=ks,
                            ),
                        )
                        # block.add_module(
                        #     f"batch_norm {i+1}_{j+1}",
                        #     nn.BatchNorm2d(
                        #         self.hidden_channels[n_conv_layers - 1 - (i)]
                        #     ),
                        # )
                        block.add_module(f"activation_bis_{i+1}_{j+1}", self.Activation)

                    block.add_module(
                        f"trans_conv{i+1}",
                        nn.ConvTranspose2d(
                            stride=(2, 1),
                            in_channels=hidden_channels[0],
                            out_channels=self.in_channels,
                            kernel_size=[ks[0] + 1, ks[1]],
                            padding=padding,
                            padding_mode="zeros",
                        ),
                    )
                    # block.add_module(f"activation_final{i+1}", self.Activation)
                    # block.add_module(
                    #     f"conv{i+1}",
                    #     CausalConv2d(
                    #         in_channels=hidden_channels[-1],
                    #         out_channels=self.in_channels,
                    #         kernel_size=[ks[0], ks[1]],
                    #     ),
                    # )

                    # block.add_module(
                    #     f'batch_norm {i+1}', nn.BatchNorm1d(self.out_channels))
                    self.conv_upsample.append(block)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = torch.unsqueeze(x, dim=1)
        outputs = []
        for block in self.conv_downsample:
            x = block(x)
            outputs.append(x)
        for i, block in enumerate(self.conv_upsample):
            if i == 0:
                x = block(x)
            else:
                x = x + outputs[self.n_conv_layers - 1 - i]
                x = block(x)
        x = torch.squeeze(x)
        # x = torch.sigmoid(x)  # we want to prove the Cross Entropy
        return x

    def train_step(self, batch: Tuple, device: str):
        x, y = batch
        x = x.to(device=device, dtype=torch.double)
        y = y.to(device=device, dtype=torch.double)
        x = self.forward(x).squeeze()
        loss = self.loss(x, y)
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
                "hidden_channels": self.hidden_channels,
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


class REDENTnopooling(nn.Module):
    def __init__(
        self,
        n_conv_layers: int = None,
        in_features: int = None,
        in_channels: int = None,
        hidden_channels: list = None,
        out_features: int = None,
        out_channels: int = None,
        ks: int = None,
        padding: int = None,
        padding_mode: str = None,
        Activation: nn.Module = None,
        n_block_layers: int = None,
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

        self.conv_downsample = nn.ModuleList()
        self.conv_upsample = nn.ModuleList()
        self.n_conv_layers = n_conv_layers
        self.in_features = in_features
        self.in_channels = in_channels
        self.out_features = out_features
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.ks = ks
        self.padding = padding
        self.padding_mode = padding_mode
        self.Activation = Activation
        self.n_block_layers = n_block_layers
        self.loss = Loss
        if self.n_conv_layers != None:
            for i in range(n_conv_layers):
                if i == 0:
                    block = nn.Sequential()
                    block.add_module(
                        f"conv{i+1}",
                        nn.Conv1d(
                            dilation=1,
                            stride=1,
                            in_channels=in_channels,
                            out_channels=hidden_channels[i],
                            kernel_size=ks,
                            padding=padding,
                            padding_mode=padding_mode,
                        ),
                    )
                    # block.add_module(
                    #     f"batch_norm {i+1}", nn.BatchNorm1d(hidden_channels[i])
                    # )
                    block.add_module(f"activation {i+1}", self.Activation)
                    for j in range(self.n_block_layers):
                        block.add_module(
                            f"conv_{i+1}_{j+1}",
                            nn.Conv1d(
                                dilation=1,
                                stride=1,
                                in_channels=self.hidden_channels[i],
                                out_channels=self.hidden_channels[i],
                                kernel_size=ks,
                                padding=padding,
                                padding_mode=padding_mode,
                            ),
                        )
                        # block.add_module(
                        #     f"batch_norm {i+1}_{j+1}",
                        #     nn.BatchNorm1d(self.hidden_channels[i]),
                        # )
                        block.add_module(f"activation_{i+1}_{j+1}", self.Activation)
                    # block.add_module(f"pooling {i+1}", nn.#AvgPool1d(kernel_size=2))
                    self.conv_downsample.append(block)

                elif (i > 0) and (i < n_conv_layers - 1):
                    block = nn.Sequential()
                    block.add_module(
                        f"conv{i+1}",
                        nn.Conv1d(
                            dilation=1,
                            stride=1,
                            in_channels=hidden_channels[i - 1],
                            out_channels=hidden_channels[i],
                            kernel_size=ks,
                            padding=padding,
                            padding_mode=padding_mode,
                        ),
                    )
                    # block.add_module(
                    #     f"batch_norm {i+1}", nn.BatchNorm1d(hidden_channels[i])
                    # )
                    block.add_module(f"activation {i+1}", self.Activation)
                    for j in range(self.n_block_layers):
                        block.add_module(
                            f"conv_{i+1}_{j+1}",
                            nn.Conv1d(
                                dilation=1,
                                stride=1,
                                in_channels=self.hidden_channels[i],
                                out_channels=self.hidden_channels[i],
                                kernel_size=ks,
                                padding=padding,
                                padding_mode=padding_mode,
                            ),
                        )
                        # block.add_module(
                        #     f"batch_norm {i+1}_{j+1}",
                        #     nn.BatchNorm1d(self.hidden_channels[i]),
                        # )
                        block.add_module(f"activation_{i+1}_{j+1}", self.Activation)
                    # block.add_module(f"pooling {i+1}", nn.#AvgPool1d(kernel_size=2))
                    self.conv_downsample.append(block)
                elif i == n_conv_layers - 1:

                    block = nn.Sequential()

                    block.add_module(
                        f"conv{i+1}",
                        nn.Conv1d(
                            dilation=1,
                            stride=1,
                            in_channels=hidden_channels[i - 1],
                            out_channels=hidden_channels[i],
                            kernel_size=ks,
                            padding=padding,
                            padding_mode=padding_mode,
                        ),
                    )
                    # block.add_module(
                    #     f"batch_norm {i+1}", nn.BatchNorm1d(hidden_channels[i])
                    # )
                    block.add_module(f"activation_{i+1}", self.Activation)

                    for j in range(self.n_block_layers):

                        block.add_module(
                            f"conv_{i+1}_{j+1}",
                            nn.Conv1d(
                                dilation=1,
                                stride=1,
                                in_channels=self.hidden_channels[i],
                                out_channels=self.hidden_channels[i],
                                kernel_size=ks,
                                padding=padding,
                                padding_mode=padding_mode,
                            ),
                        )
                        # block.add_module(
                        #     f"batch_norm {i+1}_{j+1}",
                        #     nn.BatchNorm1d(self.hidden_channels[i]),
                        # )
                        block.add_module(f"activation_{i+1}_{j+1}", self.Activation)
                    # block.add_module(f"pooling {i+1}", nn.AvgPool1d(kernel_size=2))
                    self.conv_downsample.append(block)

            for i in range(self.n_conv_layers):
                if i == 0 and self.n_conv_layers != 1:
                    block = nn.Sequential()
                    block.add_module(
                        f"trans_conv{i+1}",
                        nn.Conv1d(
                            stride=1,
                            in_channels=hidden_channels[n_conv_layers - 1 - i],
                            out_channels=hidden_channels[n_conv_layers - 1 - (i + 1)],
                            kernel_size=ks,
                            padding=padding,
                            padding_mode="circular",
                        ),
                    )
                    # block.add_module(
                    #     f"batch_norm {i+1}",
                    #     nn.BatchNorm1d(
                    #         self.hidden_channels[n_conv_layers - 1 - (i + 1)]
                    #     ),
                    # )
                    block.add_module(f"activation {i+1}", self.Activation)
                    for j in range(self.n_block_layers):
                        block.add_module(
                            f"conv_{i+1}_{j+1}",
                            nn.Conv1d(
                                dilation=1,
                                stride=1,
                                in_channels=self.hidden_channels[
                                    n_conv_layers - 1 - (i + 1)
                                ],
                                out_channels=self.hidden_channels[
                                    n_conv_layers - 1 - (i + 1)
                                ],
                                kernel_size=ks,
                                padding=padding,
                                padding_mode=padding_mode,
                            ),
                        )
                        # block.add_module(
                        #     f"batch_norm {i+1}_{j+1}",
                        #     nn.BatchNorm1d(
                        #         self.hidden_channels[n_conv_layers - 1 - (i + 1)]
                        #     ),
                        # )
                        block.add_module(f"activation_{i+1}_{j+1}", self.Activation)
                    self.conv_upsample.append(block)
                elif (i > 0) and (i < n_conv_layers - 1):
                    block = nn.Sequential()
                    block.add_module(
                        f"trans_conv{i+1}",
                        nn.Conv1d(
                            stride=1,
                            in_channels=hidden_channels[n_conv_layers - 1 - (i)],
                            out_channels=hidden_channels[n_conv_layers - 1 - (i + 1)],
                            kernel_size=ks,
                            padding=padding,
                            padding_mode="circular",
                        ),
                    )
                    # block.add_module(
                    #     f"batch_norm {i+1}",
                    #     nn.BatchNorm1d(
                    #         self.hidden_channels[n_conv_layers - 1 - (i + 1)]
                    #     ),
                    # )
                    block.add_module(f"activation {i+1}", self.Activation)
                    for j in range(self.n_block_layers):
                        block.add_module(
                            f"conv_{i+1}_{j+1}",
                            nn.Conv1d(
                                dilation=1,
                                stride=1,
                                in_channels=self.hidden_channels[
                                    n_conv_layers - 1 - (i + 1)
                                ],
                                out_channels=self.hidden_channels[
                                    n_conv_layers - 1 - (i + 1)
                                ],
                                kernel_size=ks,
                                padding=padding,
                                padding_mode=padding_mode,
                            ),
                        )
                        # block.add_module(
                        #     f"batch_norm {i+1}_{j+1}",
                        #     nn.BatchNorm1d(
                        #         self.hidden_channels[n_conv_layers - 1 - (i + 1)]
                        #     ),
                        # )
                        block.add_module(f"activation_{i+1}_{j+1}", self.Activation)
                    self.conv_upsample.append(block)
                elif i == n_conv_layers - 1:
                    block = nn.Sequential()
                    for j in range(self.n_block_layers):
                        block.add_module(
                            f"conv_{i+1}_{j+1}",
                            nn.Conv1d(
                                dilation=1,
                                stride=1,
                                in_channels=self.hidden_channels[
                                    n_conv_layers - 1 - (i)
                                ],
                                out_channels=self.hidden_channels[
                                    n_conv_layers - 1 - (i)
                                ],
                                kernel_size=ks,
                                padding=padding,
                                padding_mode=padding_mode,
                            ),
                        )
                        # block.add_module(
                        #     f"batch_norm {i+1}_{j+1}",
                        #     nn.BatchNorm1d(
                        #         self.hidden_channels[n_conv_layers - 1 - (i)]
                        #     ),
                        # )
                        block.add_module(f"activation_bis_{i+1}_{j+1}", self.Activation)

                    block.add_module(
                        f"trans_conv{i+1}",
                        nn.Conv1d(
                            stride=1,
                            in_channels=hidden_channels[n_conv_layers - 1 - (i)],
                            out_channels=self.out_channels,
                            kernel_size=ks,
                            padding=padding,
                            padding_mode="zeros",
                        ),
                    )
                    # block.add_module(
                    #     f'batch_norm {i+1}', nn.BatchNorm1d(self.out_channels))
                    self.conv_upsample.append(block)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = torch.unsqueeze(x, dim=1)
        outputs = []
        for block in self.conv_downsample:
            x = block(x)
            outputs.append(x)
        for i, block in enumerate(self.conv_upsample):
            if i == 0:
                x = block(x)
            else:
                x = x + outputs[self.n_conv_layers - 1 - i]
                x = block(x)
        x = torch.squeeze(x)
        # x = torch.sigmoid(x)  # we want to prove the Cross Entropy
        return x

    def functional(self, x: torch.tensor):
        x = torch.unsqueeze(x, dim=1)
        outputs = []
        for block in self.conv_downsample:
            x = block(x)
            outputs.append(x)
        for i, block in enumerate(self.conv_upsample):
            if i == 0:
                x = block(x)
            else:
                x = x + outputs[self.n_conv_layers - 1 - i]
                x = block(x)
        x = torch.squeeze(x)
        # x = torch.sigmoid(x)  # we want to prove the Cross Entropy
        return x.mean(-1)

    def train_step(self, batch: Tuple, device: str):
        x, y = batch
        x = x.to(device=device, dtype=torch.double)
        y = y.to(device=device, dtype=torch.double)
        x = self.forward(x).squeeze()
        loss = self.loss(x, y)
        return loss

    def r2_computation(self, batch: Tuple, device: str, r2):
        x, y = batch
        x = self.forward(x.to(dtype=torch.double, device=device))
        y = y.double()
        # print(y.shape,x.shape)
        r2.update(x.cpu().detach().view(-1), y.cpu().detach().view(-1))
        return r2

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
                "hidden_channels": self.hidden_channels,
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


# %%
