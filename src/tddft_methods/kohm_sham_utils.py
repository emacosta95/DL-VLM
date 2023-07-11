import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from typing import Tuple


def quench_field(
    h_i: torch.Tensor, h_f: torch.Tensor, lambd: float, time: torch.Tensor
):
    h = (
        h_i[None, :, :] * torch.exp(-time[:, None, None] * lambd)
        + (1 - torch.exp(-time[:, None, None] * lambd)) * h_f[None, :, :]
    )
    return h


def compute_the_inverse_jacobian(
    z: torch.Tensor, energy: nn.Module, tol: float = 10**-3
):
    z.requires_grad_(True)
    for i in range(z.shape[-1]):
        f = energy.functional_value(z)[:, 1, i]
        f.backward(torch.ones_like(f), retain_graph=True)
        with torch.no_grad():
            grad_x = z.grad.clone()
            z.grad.zero_()
            if i == 0:
                jacobian = grad_x.unsqueeze(-2)
            else:
                jacobian = torch.cat((jacobian, grad_x.unsqueeze(-2)), dim=-2)

    v, w = torch.linalg.eigh(jacobian)
    v_restored = v + tol
    identity = torch.eye(z.shape[-1])
    v_matrix = torch.einsum("bj,ij->bij", 1 / v_restored, identity)
    inverse_jacobian = torch.einsum("bji,bik,bkr->bjr", w, v_matrix, w.conj())

    return inverse_jacobian.squeeze(0)


def compute_the_gradient(
    m: torch.DoubleTensor, h: torch.DoubleTensor, energy: nn.Module, respect_to: str
) -> torch.DoubleTensor:
    m = m.detach()
    if respect_to == "z":
        z = m[:, 0, :]
        z.requires_grad_(True)
        input = torch.cat((z.unsqueeze(1), m[:, 1, :].unsqueeze(1)), dim=1)
    elif respect_to == "x":
        x = m[:, 1, :]
        x.requires_grad_(True)
        input = torch.cat((m[:, 0, :].unsqueeze(1), x.unsqueeze(1)), dim=1)
    eng = energy(z=input, h=h)
    eng.backward(torch.ones_like(eng))
    with torch.no_grad():
        if respect_to == "z":
            grad = z.grad.clone()
            z.grad.zero_()
        elif respect_to == "x":
            grad = x.grad.clone()
            x.grad.zero_()

    return grad.detach(), eng.squeeze().item()


def initialize_psi_from_z_and_x(
    z: torch.DoubleTensor, x: torch.DoubleTensor
) -> torch.ComplexType:
    psi = torch.zeros(size=(z.shape[-1], 2), dtype=torch.complex128)
    teta = torch.acos(x / torch.sqrt(1 - z**2))
    a = torch.sqrt((1 - z) / 2)
    b = torch.sqrt((1 + z) / 2)
    psi[:, 0] = torch.exp(-1j * teta) * a
    psi[:, 1] = b
    return psi


def build_hamiltonian(
    field_x: torch.DoubleTensor, field_z: torch.DoubleTensor
) -> torch.ComplexType:
    x_operator = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    z_operator = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)

    return (
        field_x[:, None, None] * x_operator[None, :, :]
        + field_z[:, None, None] * z_operator[None, :, :]
    )


def compute_the_magnetization(psi: torch.Tensor) -> Tuple[torch.DoubleTensor]:
    x_operator = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    z_operator = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)

    x = torch.einsum("li,ij,lj->l", torch.conj(psi), x_operator, psi).double()
    z = torch.einsum("li,ij,lj->l", torch.conj(psi), z_operator, psi).double()

    return z.detach(), x.detach()


def crank_nicolson_algorithm(
    hamiltonian: torch.ComplexType, psi: torch.ComplexType, dt: float
):
    identity = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.complex128)
    unitary_op = identity[None, :, :] + 0.5j * dt * hamiltonian
    unitary_op_star = identity[None, :, :] - 0.5j * dt * hamiltonian
    unitary = torch.einsum(
        "lab,lbc->lac", torch.linalg.inv(unitary_op), unitary_op_star
    )
    # unitary = torch.matrix_exp(-1j * dt * hamiltonian)
    psi = torch.einsum("lab,lb->la", unitary, psi)
    # psi = psi - 0.5j * dt * torch.einsum("lab,lb->la", hamiltonian, psi)
    # impose the norm
    psi = psi / torch.linalg.norm(psi, dim=-1)[:, None]
    return psi


def time_step_backward_algorithm(
    psi: torch.ComplexType,
    h: torch.Tensor,
    energy: nn.Module,
    dt: float,
    self_consistent_steps: int,
):
    psi0 = psi.clone()
    for i in range(self_consistent_steps):
        hamiltonian, eng = get_the_hamiltonian(psi, h=h, energy=energy)
        unitary = torch.matrix_exp(-1j * dt * hamiltonian)
        psi1 = torch.einsum("lab,lb->la", unitary, psi)
        # psi = psi0 - 1j * dt * torch.einsum("lab,lb->la", hamiltonian, psi)
        psi1 = psi1 / torch.linalg.norm(psi1, dim=-1)[:, None]

        hamiltonian1, eng1 = get_the_hamiltonian(psi1, h=h, energy=energy)
        hamiltonian = 0.5 * (hamiltonian + hamiltonian1)
        psi = torch.einsum("lab,lb->la", unitary, psi)
        psi = psi / torch.linalg.norm(psi, dim=-1)[:, None]
    return psi, eng


def get_the_hamiltonian(psi: torch.ComplexType, h: torch.Tensor, energy: nn.Module):
    x, z = compute_the_magnetization(psi=psi.clone())

    z = torch.cat((z.view(1, -1), x.view(1, -1)), dim=0)
    z = z.unsqueeze(0)  # the batch dimension

    h_eff, eng0 = compute_the_gradient(m=z, h=h, energy=energy, respect_to="z")
    omega_eff, _ = compute_the_gradient(m=z, h=h, energy=energy, respect_to="x")
    hamiltonian = build_hamiltonian(field_x=omega_eff[0], field_z=h_eff[0])

    return hamiltonian, eng0


def time_step_crank_nicolson_algorithm(
    psi: torch.ComplexType,
    h: torch.Tensor,
    h_plus: torch.Tensor,
    energy: nn.Module,
    dt: float,
    self_consistent_steps: int,
):
    for j in range(self_consistent_steps):
        hamiltonian, eng0 = get_the_hamiltonian(psi=psi.clone(), h=h, energy=energy)
        # print(hamiltonian[0, :], "\n")
        psi_1 = crank_nicolson_algorithm(
            hamiltonian=hamiltonian, psi=psi.clone(), dt=dt
        )

        hamiltonian_1, eng = get_the_hamiltonian(psi=psi_1, h=h_plus, energy=energy)
        psi = crank_nicolson_algorithm(
            hamiltonian=0.5 * (hamiltonian_1 + hamiltonian), psi=psi.clone(), dt=dt
        )

    return psi, eng
