import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from typing import Tuple, List
from src.training.model_utils.lstm_cnn import LSTMcell, Encoder1D, Decoder1D
from tqdm import trange
import matplotlib.pyplot as plt


class CNNLSTM(nn.Module):
    def __init__(
        self,
        n_conv: int,
        activation: nn.Module,
        hidden_neurons: int,
        kernel_size: int,
        in_channels: int,
        latent_dimension: int,
        n_layers: int,
        pooling_size: int,
        hidden_channels: List[int],
        padding_mode: str,
        loss:nn.Module,
    ) -> None:
        super().__init__()
        
        self.loss=loss
        self.encoder = Encoder1D(
            n_conv=n_conv,
            activation=activation,
            hidden_neurons=hidden_neurons,
            kernel_size=kernel_size,
            in_channels=in_channels,
            latent_dimension=latent_dimension,
            n_layers=n_layers,
            pooling_size=pooling_size,
            padding_mode=padding_mode,
            hc=hidden_channels,
        )

        self.decoder = Decoder1D(
            n_conv=n_conv,
            activation=activation,
            in_channels=hidden_channels[-1],
            out_channels=in_channels,
            kernel_size=kernel_size,
            latent_dimension=latent_dimension,
            n_layers=n_layers,
            hidden_neurons=hidden_neurons,
            hc=hidden_channels,
        )

        self.input_to_lstm=nn.Linear(latent_dimension,hidden_neurons)
        self.lstmcell_1=LSTMcell(input_size=hidden_neurons,output_size=hidden_neurons)
        self.lstmcell_2=LSTMcell(input_size=hidden_neurons,output_size=hidden_neurons)
        self.output_from_lstm=nn.Linear(hidden_neurons,latent_dimension)
        
        
    def forward(self,d:torch.Tensor):
        
        z_total_evolution=torch.zeros_like(d)
        for t in range(d.shape[-2]):
            l=self.encoder(d[:,t])
            l_lstm=self.input_to_lstm(l)
            #initialization
            if t==0:
                c_a=torch.zeros_like(l_lstm)
                h_a=torch.zeros_like(l_lstm)
            o,c_a,h_a=self.lstmcell_1(l_lstm,c_a,h_a)
            
            if t==0:
                c_b=torch.zeros_like(o)
                h_b=torch.zeros_like(o)
            o,c_a,h_a=self.lstmcell_2(o,c_b,h_b)
            
            o=self.output_from_lstm(o)
            z=self.decoder(o)
            # the image size follows the input size (in this way we recover the scalability)
            z=F.adaptive_avg_pool1d(z,output_size=d.shape[-1])
            z_total_evolution[:,t,:]=z.squeeze(1)
            
        return z_total_evolution

    def train_step(self, batch: Tuple, device: str):
        x, y = batch
        x = x.to(device=device, dtype=torch.double)
        y = y.to(device=device, dtype=torch.double)
        x = self.forward(x).squeeze()
        loss = self.loss(x, y)
        return loss
            
                    