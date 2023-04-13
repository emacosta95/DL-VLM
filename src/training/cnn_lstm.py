import torch
from src.training.model_utils.lstm_cnn import ConvLSTMCell
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(
        self,
        n_conv: int,
        input_channels: int,
        output_channels: int,
        hidden_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()

        self.nconv=n_conv
        self.initial_embedding = nn.Conv1d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )

        self.final_embedding = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )

        self.cnnlstm=nn.ModuleList()
        
        for i in range(n_conv):
            self.cnnlstm.add_module(name=f'conv_block_{i}', ConvLSTMCell(hidden_channels,hidden_channels,hidden_channels,kernel_size))
            
    def forward(self,b:torch.Tensor):
        
        
        for t in range(b.shape[-2]):
        
            if t==0:    
                x=self.initial_embedding(b)
                h=torch.zeros_like(x)
                c=torch.zeros_like(x)
                
                hs=[h]*self.nconv
                cs=[c]*self.nconv
            for i,cnn in enumerate(self.cnnlstm):
                x,h,c=cnn(x,hs[i],cs[i])
                hs[i]=h
                cs[i]=c
                
            z=self.final_embedding(x)
            
            if t==0:
                z_t=z.unsqueeze(-2)
            else:
                z_t=torch.cat((z_t,z),dim=-2)

            
        
