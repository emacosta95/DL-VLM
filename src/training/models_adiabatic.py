import torch
import torch.nn as nn
from typing import Tuple, Optional
from torch.nn.functional import sigmoid


class TDDFTadiabaticModel(nn.Module):
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
        if self.in_channels == 1:
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
        # +-1 outputs
        x = torch.sin(x)
        # x = torch.sigmoid(x)  # we want to prove the Cross Entropy
        return x

    def functional(self, x: torch.tensor):
        if self.in_channels == 1:
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
        x = torch.sin(x)
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


class Energy_XXZX(nn.Module):
    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        super().__init__()

        self.model: nn.Module = model

    def forward(self, z: torch.Tensor, h: torch.Tensor):
        # single take
        if z.shape[0] != 1:
            f = self.model(z)  # batch x 2 x size
        else:
            f = self.model(z).unsqueeze(0)
        # sum of zz + x
        f_tot = f[:, :]  # + torch.roll(z[:, 0, :], shifts=-1, dims=-1) * z[:, 0, :]
        # longitudinal external term
        e_ext = h * z  # batch x size
        # transverse external term

        return e_ext.sum(-1).sum(-1) + f_tot.sum(
            -1
        )  # mean for the static dft / sum for tddft

    def functional_value(self, z: torch.Tensor):
        if z.shape[0] != 1:
            f = self.model(z)
        else:
            f = self.model(z).unsqueeze(0)
        return f


class Energy_reduction_XXZX(nn.Module):
    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        super().__init__()

        self.model: nn.Module = model

    def forward(self, z: torch.Tensor, h: torch.Tensor):
        # single take
        if z.shape[0] != 1:
            f = self.model(z)  # batch x 2 x size
        else:
            f = self.model(z).unsqueeze(0)
        # sum of zz + x
        f_tot = f[:, :]  # + torch.roll(z[:, 0, :], shifts=-1, dims=-1) * z[:, 0, :]
        # longitudinal external term
        e_ext = h * z  # batch x size
        df = torch.roll(z[:, 0, :], shifts=-1, dims=-1) * z[:, 0, :]
        # transverse external term

        return (
            e_ext.sum(-1).sum(-1) + f_tot.sum(-1) + df.sum(-1)
        )  # mean for the static dft / sum for tddft

    def functional_value(self, z: torch.Tensor):
        if z.shape[0] != 1:
            f = self.model(z)
        else:
            f = self.model(z).unsqueeze(0)
        return f


class Energy_XXZX_1input(nn.Module):
    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        super().__init__()

        self.model: nn.Module = model

    def forward(self, z: torch.Tensor, h: torch.Tensor):
        # single take
        if z.shape[0] != 1:
            f = self.model(z)  # batch x 2 x size
        else:
            f = self.model(z).squeeze(0)
        # sum of zz + x
        f_tot = f[0, :]  # + torch.roll(z[:, 0, :], shifts=-1, dims=-1) * z[:, 0, :]
        # longitudinal external term

        e_ext = h[0, :] * z  # batch x size
        # transverse external term
        e_ext_transverse = f[1, :] * h[1, :]

        return e_ext.sum(-1) + f_tot.sum(-1) + e_ext_transverse.sum(-1)

    def batch_estimation(self, z: torch.Tensor, h: torch.Tensor):
        # single take
        if z.shape[0] != 1:
            f = self.model(z)  # batch x 2 x size
        else:
            f = self.model(z).unsqueeze(0)
        # sum of zz + x
        f_tot = f[:, 0, :]  # + torch.roll(z[:, 0, :], shifts=-1, dims=-1) * z[:, 0, :]
        # longitudinal external term

        e_ext = h[:, 0, :] * z  # batch x size
        # transverse external term
        e_ext_transverse = f[:, 1, :] * h[:, 1, :]

        return e_ext.sum(-1) + f_tot.sum(-1) + e_ext_transverse.sum(-1)

    def functional_value(self, z: torch.Tensor):
        if z.shape[0] != 1:
            f = self.model(z)
        else:
            f = self.model(z).unsqueeze(0)
        return f


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
        # x = torch.unsqueeze(x, dim=1)
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
        # x = torch.unsqueeze(x, dim=1)
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
