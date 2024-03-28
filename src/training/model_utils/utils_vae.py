import torch as pt
from torch.nn import functional as F
import torch
from typing import Tuple

class VaeLoss(nn.Module):
    def __init__(self, variational_beta):

        super().__init__()
        self.variational_beta = variational_beta

    def forward(self, recon_x, x, mu, logvar):
        recon_loss = F.binary_cross_entropy(
            recon_x.view(recon_x.shape[0], -1),
            x.view(x.shape[0], -1),
            reduction="mean",
        )
        kldivergence = -0.5 * pt.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.variational_beta * kldivergence, kldivergence
    
    def train_step(self, batch: Tuple, device: str):
        x, y = batch
        x = x.to(device=device, dtype=torch.double)
        y = y.to(device=device, dtype=torch.double)
        x = self.forward(x)
        loss = self.loss(x, y)
        return loss
    
    def valid_step(self, batch: Tuple, device: str):
        x, y = batch
        x = x.to(device=device, dtype=torch.double)
        y = y.to(device=device, dtype=torch.double)
        x = self.forward(x)
        loss = self.loss(x, y)
        return loss