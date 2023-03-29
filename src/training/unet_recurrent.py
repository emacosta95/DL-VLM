import torch
import torch.nn as nn
from src.training.model_utils.utils_seq2seq import (
    EncoderSeq2Seq,
    DecoderOperator,
    ProbabilityHead,
)
from typing import Tuple


class UnetRNN(nn.Module):
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
        y = torch.squeeze(x)
        # x = torch.sigmoid(x)  # we want to prove the Cross Entropy
        return y

    def train_step(self, batch: Tuple, device: str):
        x, y = batch
        x = x.to(device=device, dtype=torch.double)
        y = y.to(device=device, dtype=torch.double)
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        y_tilde = torch.zeros_like(x)
        for t in range(x.shape[-2] - 1):
            input = torch.cat((x[:, :, t, :], y_tilde[:, :, t, :]), dim=1)
            output = self.forward(input)
            y_tilde[:, 0, t + 1, :] = output
        loss = self.loss(y_tilde.squeeze(), y)
        return loss

    def valid_step(self, batch: Tuple, device: str):
        x, y = batch
        x = x.to(device=device, dtype=torch.double)
        y = y.to(device=device, dtype=torch.double)
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        y_tilde = torch.zeros_like(x)
        for t in range(x.shape[-2] - 1):
            input = torch.cat((x[:, :, t, :], y_tilde[:, :, t, :]), dim=1)
            output = self.forward(input)
            y_tilde[:, 0, t + 1, :] = output
        loss = self.loss(y_tilde.squeeze(), y)
        return loss

    def prediction_step(self, time_step_initial: int, x: torch.Tensor, y: torch.Tensor):
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        print(y.shape)
        for t in range(time_step_initial, x.shape[-2] - 1):
            for s in range(x.shape[-1]):
                mu, logsigma = self.forward(x=x, y=y)
                y_sample = self.probability_head.prediction_sample(mu, logsigma, s, t)
                y[:, 0, t, s] = y_sample
                # y = (y[:, :, :, :] + torch.roll(y[:, :, :, :], shifts=1, dims=-2)) * 0.5
        y = y.squeeze()
        return y
