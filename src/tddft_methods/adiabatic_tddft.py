import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from src.training.conv_block_model import ConvBlock1D
import numpy as np
from typing import Tuple


class AdiabaticTDDFT:
    def __init__(
        self, model: torch.nn.Module, h: torch.tensor, psi0: torch.tensor
    ) -> None:

        self.psi = psi0
        self.h = h
        self.functional = model.double()
        self.grad = 0.0

    def gradient_descent_step(
        self,
    ) -> tuple:
        """This routine computes the step of the gradient using both the positivity and the nomralization constrain

        Arguments:
        energy[nn.Module]: [the energy functional]
        phi[pt.tensor]: [the sqrt of the density profile]

        Returns:
            eng[pt.tensor]: [the energy value computed before the step]
            phi[pt.tensor]: [the wavefunction evaluated after the step]
        """

        # print(torch.typename(self.psi), torch.typename(self.sigmaz))
        # self.psi = self.psi.to(torch.double
        phi = self.psi.clone()

        w = 1 - 2 * torch.conj(phi) * phi
        print(w)
        w = w.to(torch.double)
        w.requires_grad_(True)
        f = self.functional(w)
        print("func=", f)
        f.backward(torch.ones_like(f))
        with torch.no_grad():

            grad = w.grad
            self.grad = grad.clone()
            # print("psi=", self.psi)
            # print("grad=", grad)

            w.grad.zero_()

    def kutta_step(self, vector: torch.Tensor, t: float, dt: float):

        vector.to(torch.complex128)
        field = self.h[:, int(t / dt) + 1] + self.grad
        output = torch.einsum("al,al->al", field, vector)
        # print(self.grad)
        return -1j * (output)

    def time_step(self, dt: float, t: float):

        # non linear term
        self.gradient_descent_step()

        # runge kutta
        k1 = self.kutta_step(self.psi.to(torch.complex128), t=t, dt=dt)
        # print("k1=", k1)
        k2 = self.kutta_step(self.psi + dt * k1 / 2, t=t + dt / 2, dt=dt)
        # print("k2=", k2)
        k3 = self.kutta_step(self.psi + dt * k2 / 2, t=t + dt / 2, dt=dt)
        # print("k2=", k3)
        k4 = self.kutta_step(self.psi + dt * k3, t + dt, dt)
        # print("k2=", k4)
        self.psi = self.psi + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * dt

    def compute_magnetization(self):
        z = 1 - 2 * torch.conj(self.psi) * self.psi
        return z.view(-1)


class AdiabaticTDDFTNN(nn.Module):
    def __init__(
        self,
        n_conv: int,
        activation: nn.Module,
        hc: int,
        in_channels: int,
        kernel_size: int,
        padding_mode: str,
        out_channels: int,
        j: float,
        tf: float,
        dt: float,
        time_interval: int,
        device: str,
    ) -> None:
        super().__init__()

        self.v_adiabatic = ConvBlock1D(
            n_conv=n_conv,
            hc=hc,
            in_channels=in_channels,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            out_channels=out_channels,
            activation=activation,
        )

        self.device = device

        self.j = j

        self.loss = nn.MSELoss()

        self.time = torch.linspace(0, tf, int(tf / dt))[:time_interval]

    def time_step(self, dt: float, t: float, psi: torch.Tensor, h: torch.Tensor):

        # non linear term
        z = 1 - 2 * torch.einsum("aml,alm->al", torch.conj(psi), psi)
        z = z.to(torch.double)
        v = self.forward(z)

        # runge kutta
        field = v + h[:, int(t / dt)]
        k1 = self.kutta_step(vector=psi.to(torch.complex128), field=field)
        field = v + h[:, int(t / dt + 1 / 2)]
        k2 = self.kutta_step(vector=psi + dt * k1 / 2, field=field)
        k3 = self.kutta_step(vector=psi + dt * k2 / 2, field=field)
        field = v + h[:, int(t / dt + 1)]
        k4 = self.kutta_step(vector=psi + dt * k3, field=field)
        psi = psi + (1 / 6) * (k1 * dt + 2 * k2 + 2 * k3 + k4) * dt
        return psi

    def kutta_step(self, field: torch.Tensor, vector: torch.Tensor):

        vector.to(torch.complex128)
        matrix = (
            -1j
            * (
                self.laplacian[None, :, :]
                + torch.einsum(
                    "lm,am->alm",
                    torch.eye(field.shape[-1]).to(torch.double).to(self.device),
                    field,
                )
            ).to(torch.complex128)
        )
        output = torch.einsum("aij,amj->ami", matrix, vector)

        return output

    def forward(self, x: torch.tensor):
        x = x.unsqueeze(1)
        x = self.v_adiabatic(x)
        x = x.squeeze(1)
        return x

    def get_magnetization(self, psi: torch.tensor):
        z = 1 - 2 * torch.einsum("ami,ami->ai", torch.conj(psi), psi).to(torch.double)

        return z

    def time_evolution(self, h: torch.Tensor):

        size = h.shape[-1]
        time = self.time
        # laplacian
        self.laplacian = torch.zeros(size=(size, size)).to(
            dtype=torch.double, device=self.device
        )
        self.laplacian[torch.arange(size), torch.arange(size)] = 2.0
        self.laplacian[
            (torch.arange(size) + 1) % size, (torch.arange(size)) % size
        ] = -1
        self.laplacian[
            (torch.arange(size) - 1) % size, (torch.arange(size)) % size
        ] = -1
        self.laplacian = self.j * self.laplacian

        z_ml = torch.zeros_like(h).to(torch.double).to(self.device)
        dt = torch.abs(time[1] - time[0])
        psi = (
            torch.zeros((h.shape[0], h.shape[-1], h.shape[-1]))
            .to(torch.double)
            .to(self.device)
        )
        psi[:, 0, :] = np.sqrt(0.5)

        z_ml = self.get_magnetization(psi).view(-1, 1, size)
        for i, t in enumerate(time[:-1]):
            psi = self.time_step(dt=dt, t=t, psi=psi, h=h)
            z_ml = torch.cat(
                (self.get_magnetization(psi).view(-1, 1, size), z_ml), axis=1
            )
        return z_ml, psi

    def initialization(self, h):
        size = h.shape[-1]
        time = self.time
        # laplacian
        self.laplacian = torch.zeros(size=(size, size)).to(
            dtype=torch.double, device=self.device
        )
        self.laplacian[torch.arange(size), torch.arange(size)] = 2.0
        self.laplacian[
            (torch.arange(size) + 1) % size, (torch.arange(size)) % size
        ] = -1
        self.laplacian[
            (torch.arange(size) - 1) % size, (torch.arange(size)) % size
        ] = -1
        self.laplacian = self.j * self.laplacian

        z_ml = torch.zeros_like(h).to(torch.double).to(self.device)
        dt = torch.abs(time[1] - time[0])
        psi = (
            torch.zeros((h.shape[0], h.shape[-1], h.shape[-1]))
            .to(torch.double)
            .to(self.device)
        )
        psi[:, 0, :] = np.sqrt(0.5)

        return z_ml, psi

    def train_step(self, batch: Tuple, device: str):
        h, z = batch
        h = h.to(device=device, dtype=torch.double)
        z = z.to(device=device, dtype=torch.double)
        z_ml = self.time_evolution(h=h, time=self.time)
        loss = self.loss(z, z_ml)
        return loss, z_ml
