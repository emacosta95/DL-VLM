import torch
import torch.nn as nn


class modelLDA(nn.Module):
    def __init__(self, coeff: torch.Tensor) -> None:
        super().__init__()

        self.coeff = coeff

    def forward(self, m: torch.Tensor):
        z = m[:, 0, :]
        x = m[:, 1, :]
        z_poly = z[None, :, :] ** (torch.arange(0, 9, 2))[:, None, None]
        x_poly = x[None, :, :] ** (torch.arange(0, 9, 2))[:, None, None]
        up = torch.zeros_like(z)
        for i in range(5):
            for j in range(5):
                up = up + self.coeff[5 * i + j] * z_poly[i] * x_poly[j]
        down = torch.ones_like(z)
        for i in range(4):
            for j in range(4):
                down = down + self.coeff[25 + 4 * i + j] * z_poly[i] * x_poly[j]
        f_lda = up / down
        return f_lda
