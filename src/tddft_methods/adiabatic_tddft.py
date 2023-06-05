import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from src.training.conv_block_model import ConvBlock1D
import numpy as np
from typing import Tuple
from tqdm import tqdm


def pca_gradient(samples: torch.Tensor, gradient: torch.Tensor):
    # define X^T (Meyer et al. 2020)
    x = samples - gradient[None, :]
    # implement C
    c = torch.einsum("al,am->lm", x, x) / x.shape[0]
    # diagonalize C
    _, w = torch.linalg.eigh(c)
    projection = torch.einsum("kl,km->lm", w[:2], w[:2])
    return projection


class QuantumAnnealing:
    def __init__(
        self,
        z_init: torch.Tensor,
        h: torch.Tensor,
        e_target: torch.Tensor,
        energy: nn.Module,
        lr: float,
        annealing_steps: int,
    ) -> None:
        pass

        self.z_t: torch.Tensor = z_init  # batch x 0 x size
        self.h: torch.Tensor = h  # batch x t_steps x size
        self.e_target: torch.Tensor = e_target  # batch x t_steps x 0

        self.energy: nn.Module = energy

        self.lr: float = lr
        self.annealing_steps: int = annealing_steps

        self.z: torch.Tensor = torch.zeros_like(h)
        self.eng: torch.Tensor = torch.zeros(size=(h.shape[0], h.shape[1]))

    def annealing_step(self, z: torch.Tensor, h: torch.Tensor):
        # -1,1 constrain
        psi = torch.acos(z)
        psi.requires_grad_(True)
        for t in range(self.annealing_steps):
            z = torch.cos(psi)
            # energy functional
            eng = self.energy(z, h)
            eng.backward(torch.ones_like(eng))

            with torch.no_grad():
                grad = psi.grad  # batch x 0 x size
                psi -= self.lr * grad
                psi.grad.zero_()

        return torch.cos(psi).detach(), eng

    def run(
        self,
    ):
        # initial configuration
        self.z[:, 0, :] = self.z_t
        eng0 = self.energy(self.z_t, self.h[:, 0])
        self.eng[:, 0, :] = eng0

        t_bar = tqdm(range(self.h[:, 1:, :].shape[1]))

        # evolution
        for t in t_bar:
            self.z_t, eng_t = self.annealing_step(z=self.z_t, h=self.h[:, t + 1, :])
            self.z[:, t + 1] = self.z_t
            self.eng[:, t + 1] = eng_t.item()

            t_bar.set_description(
                f"de={torch.abs(self.eng[:,t+1]-self.e_target[:,t+1]).mean(0).item():.6f}"
            )
            t_bar.refresh()


class AdiabaticTDDFT:
    def __init__(
        self,
        model: torch.nn.Module,
        h: torch.tensor,
        omega: float,
        device: str,
        with_grad: bool == True,
    ) -> None:
        # self.psi = psi0.to(device=device, dtype=torch.complex128)
        self.h = h.to(device=device)
        self.with_grad = with_grad
        if self.with_grad:
            self.functional = model.double().to(device=device)
        self.grad = 0.0
        self.omega = omega
        self.x_operator = torch.tensor(
            [[0, 1], [1, 0]], dtype=torch.complex128, device=device
        )
        self.z_operator = torch.tensor(
            [[-1, 0], [0, 1]], dtype=torch.complex128, device=device
        )
        self.identity = torch.tensor(
            [[1, 0], [0, 1]], dtype=torch.complex128, device=device
        )

        # restore the gradient
        self.gradient_old = torch.ones(h.shape[-1])

        # save the f values
        self.f_values = 0.0

    def gradient_descent_step(self, psi: torch.Tensor) -> tuple:
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
        phi = psi.clone()

        w = torch.real(
            torch.conj(phi[:, 0]) * phi[:, 0] - torch.conj(phi[:, 1]) * phi[:, 1]
        )
        w = w.to(dtype=torch.double)
        w.requires_grad_(True)
        f = self.functional(w.view(1, -1))  # batch size form 1 x l
        # self.omega = f[1].detach().mean().clone()
        f = f[0].sum(-1)

        self.f_values = f.clone()
        f.backward(torch.ones_like(f))
        with torch.no_grad():
            grad = w.grad
            self.grad = grad.clone()
            # projection = pca_gradient(
            #    samples=self.sample_for_projection, gradient=self.grad
            # )
            # self.grad = torch.einsum("lm,m->l", projection, self.grad)
            w.grad.zero_()

    def time_step(self, dt: float, t: float, psi: torch.Tensor):
        # non linear term
        if self.with_grad:
            self.gradient_descent_step(psi=psi)

        # Crank-Nicholson algorithm
        field = self.h[int(t / dt)] + self.grad
        field = field.to(dtype=torch.complex128)
        ext_field = field[:, None, None] * self.z_operator[None, :, :]
        unitary_op = self.identity[None, :, :] + 0.5j * dt * (
            self.omega * self.x_operator[None, :, :] + ext_field
        )
        unitary_op_star = self.identity[None, :, :] - 0.5j * dt * (
            self.omega * self.x_operator[None, :, :] + ext_field
        )
        unitary = torch.einsum(
            "iab,ibc->iac", torch.linalg.inv(unitary_op), unitary_op_star
        )
        psi = torch.einsum("iab,ib->ia", unitary, psi)

        # impose the norm
        psi = psi / torch.linalg.norm(psi, dim=-1)[:, None]

        # self.psi = self.psi / torch.norm(self.psi, dim=-1)[:, None]

        return psi

    def compute_magnetization(self, psi: torch.Tensor):
        z = torch.conj(psi[:, 0]) * psi[:, 0] - torch.conj(psi[:, 1]) * psi[:, 1]
        return z.view(-1)


# class AdiabaticTDDFTNN(nn.Module):
#     def __init__(
#         self,
#         n_conv: int,
#         activation: nn.Module,
#         hc: int,
#         in_channels: int,
#         kernel_size: int,
#         padding_mode: str,
#         out_channels: int,
#         j: float,
#         tf: float,
#         dt: float,
#         time_interval: int,
#         device: str,
#     ) -> None:
#         super().__init__()

#         self.v_adiabatic = ConvBlock1D(
#             n_conv=n_conv,
#             hc=hc,
#             in_channels=in_channels,
#             kernel_size=kernel_size,
#             padding_mode=padding_mode,
#             out_channels=out_channels,
#             activation=activation,
#         )

#         self.device = device

#         self.j = j

#         self.loss = nn.MSELoss()

#         self.time = torch.linspace(0, tf, int(tf / dt))[:time_interval]

#     def time_step(self, dt: float, t: float, psi: torch.Tensor, h: torch.Tensor):
#         # non linear term
#         z = 1 - 2 * torch.einsum("aml,alm->al", torch.conj(psi), psi)
#         z = z.to(torch.double)
#         v = self.forward(z)

#         # runge kutta
#         field = v + h[:, int(t / dt)]
#         k1 = self.kutta_step(vector=psi.to(torch.complex128), field=field)
#         field = v + h[:, int(t / dt + 1 / 2)]
#         k2 = self.kutta_step(vector=psi + dt * k1 / 2, field=field)
#         k3 = self.kutta_step(vector=psi + dt * k2 / 2, field=field)
#         field = v + h[:, int(t / dt + 1)]
#         k4 = self.kutta_step(vector=psi + dt * k3, field=field)
#         psi = psi + (1 / 6) * (k1 * dt + 2 * k2 + 2 * k3 + k4) * dt
#         return psi

#     def kutta_step(self, field: torch.Tensor, vector: torch.Tensor):
#         vector.to(torch.complex128)
#         matrix = -1j * (
#             self.laplacian[None, :, :]
#             + torch.einsum(
#                 "lm,am->alm",
#                 torch.eye(field.shape[-1]).to(torch.double).to(self.device),
#                 field,
#             )
#         ).to(torch.complex128)
#         output = torch.einsum("aij,amj->ami", matrix, vector)

#         return output

#     def forward(self, x: torch.tensor):
#         x = x.unsqueeze(1)
#         x = self.v_adiabatic(x)
#         x = x.squeeze(1)
#         return x

#     def get_magnetization(self, psi: torch.tensor):
#         z = 1 - 2 * torch.einsum("ami,ami->ai", torch.conj(psi), psi).to(torch.double)

#         return z

#     def time_evolution(self, h: torch.Tensor):
#         size = h.shape[-1]
#         time = self.time
#         # laplacian
#         self.laplacian = torch.zeros(size=(size, size)).to(
#             dtype=torch.double, device=self.device
#         )
#         self.laplacian[torch.arange(size), torch.arange(size)] = 2.0
#         self.laplacian[
#             (torch.arange(size) + 1) % size, (torch.arange(size)) % size
#         ] = -1
#         self.laplacian[
#             (torch.arange(size) - 1) % size, (torch.arange(size)) % size
#         ] = -1
#         self.laplacian = self.j * self.laplacian

#         z_ml = torch.zeros_like(h).to(torch.double).to(self.device)
#         dt = torch.abs(time[1] - time[0])
#         psi = (
#             torch.zeros((h.shape[0], h.shape[-1], h.shape[-1]))
#             .to(torch.double)
#             .to(self.device)
#         )
#         psi[:, 0, :] = np.sqrt(0.5)

#         z_ml = self.get_magnetization(psi).view(-1, 1, size)
#         for i, t in enumerate(time[:-1]):
#             psi = self.time_step(dt=dt, t=t, psi=psi, h=h)
#             z_ml = torch.cat(
#                 (self.get_magnetization(psi).view(-1, 1, size), z_ml), axis=1
#             )
#         return z_ml, psi

#     def initialization(self, h):
#         size = h.shape[-1]
#         time = self.time
#         # laplacian
#         self.laplacian = torch.zeros(size=(size, size)).to(
#             dtype=torch.double, device=self.device
#         )
#         self.laplacian[torch.arange(size), torch.arange(size)] = 2.0
#         self.laplacian[
#             (torch.arange(size) + 1) % size, (torch.arange(size)) % size
#         ] = -1
#         self.laplacian[
#             (torch.arange(size) - 1) % size, (torch.arange(size)) % size
#         ] = -1
#         self.laplacian = self.j * self.laplacian

#         z_ml = torch.zeros_like(h).to(torch.double).to(self.device)
#         dt = torch.abs(time[1] - time[0])
#         psi = (
#             torch.zeros((h.shape[0], h.shape[-1], h.shape[-1]))
#             .to(torch.double)
#             .to(self.device)
#         )
#         psi[:, 0, :] = np.sqrt(0.5)

#         return z_ml, psi

#     def train_step(self, batch: Tuple, device: str):
#         h, z = batch
#         h = h.to(device=device, dtype=torch.double)
#         z = z.to(device=device, dtype=torch.double)
#         z_ml = self.time_evolution(h=h, time=self.time)
#         loss = self.loss(z, z_ml)
#         return loss, z_ml
