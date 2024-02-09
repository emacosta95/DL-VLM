import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from typing import Tuple, List
from tqdm import trange


def quench_field(
    h_i: torch.Tensor, h_f: torch.Tensor, lambd: float, time: torch.Tensor
):
    h = (
        h_i[None, :, :] * torch.exp(-time[:, None, None] * lambd)
        + (1 - torch.exp(-time[:, None, None] * lambd)) * h_f[None, :, :]
    )
    return h


def compute_the_gradient(
    m: torch.DoubleTensor, h: torch.DoubleTensor, energy: nn.Module, respect_to: str
) -> torch.DoubleTensor:
    m = m.detach().double()
    if respect_to == "z":
        z = m[:, 2, :]
        z.requires_grad_(True)
        input = torch.cat(
            (m[:, 0, :].unsqueeze(1), m[:, 1, :].unsqueeze(1), z.unsqueeze(1)), dim=1
        )
    elif respect_to == "x":
        x = m[:, 0, :]
        x.requires_grad_(True)
        input = torch.cat(
            (x.unsqueeze(1), m[:, 1, :].unsqueeze(1), m[:, 2, :].unsqueeze(1)), dim=1
        )
    elif respect_to == "y":
        y = m[:, 1, :]
        y.requires_grad_(True)
        input = torch.cat(
            (m[:, 0, :].unsqueeze(1), y.unsqueeze(1), m[:, 2, :].unsqueeze(1)), dim=1
        )
    eng = energy(z=input, h=h)[0]
    eng.backward()
    with torch.no_grad():
        if respect_to == "z":
            grad = z.grad.clone()
            z.grad.zero_()
        elif respect_to == "x":
            grad = x.grad.clone()
            x.grad.zero_()
        elif respect_to == "y":
            grad = y.grad.clone()
            y.grad.zero_()
    return grad.detach(), eng.squeeze().item()


def compute_the_gradient_of_the_functional_ux_model(
    z: torch.DoubleTensor, model: nn.Module
) -> torch.DoubleTensor:
    if z.shape[0] == z.shape[-1]:
        z = z.unsqueeze(0)

    z.requires_grad_(True)
    f = model(z).sum(-1)
    f.backward()
    with torch.no_grad():
        grad = z.grad.clone()
        z.grad.zero_()
    return grad.detach()[0]


def initialize_psi_from_z(z: torch.DoubleTensor) -> torch.ComplexType:
    psi = torch.zeros(size=(2, z.shape[-1]), dtype=torch.complex128)
    a = torch.sqrt((1 - z) / 2)
    b = torch.sqrt((1 + z) / 2)
    psi[0, :] = a
    psi[1, :] = b
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
    y_operator = torch.tensor([[0.0, -1j], [1j, 0.0]], dtype=torch.complex128)

    x = torch.einsum("il,ij,jl->l", torch.conj(psi), x_operator, psi).double()
    z = torch.einsum("il,ij,jl->l", torch.conj(psi), z_operator, psi).double()
    y = torch.einsum("il,ij,jl->l", torch.conj(psi), y_operator, psi).double()

    return x.detach(), y.detach(), z.detach()


def crank_nicolson_algorithm(
    hamiltonian: torch.ComplexType, psi: torch.ComplexType, dt: float
):
    identity = torch.eye(psi.shape[0], dtype=torch.complex128)
    unitary_op = identity[None, :, :] + 0.5j * dt * hamiltonian
    unitary_op_star = identity[None, :, :] - 0.5j * dt * hamiltonian
    unitary = torch.einsum(
        "lab,lbc->lac", torch.linalg.inv(unitary_op), unitary_op_star
    )
    # unitary = torch.matrix_exp(-1j * dt * hamiltonian)
    psi = torch.einsum("lab,bl->al", unitary, psi)
    # psi = psi - 0.5j * dt * torch.einsum("lab,lb->la", hamiltonian, psi)
    # impose the norm
    # psi = psi / torch.linalg.norm(psi, dim=-1)[:, None]
    return psi


def exponentiation_algorithm(
    hamiltonian: torch.ComplexType, psi: torch.ComplexType, dt: float
):
    identity = torch.eye(psi.shape[0], dtype=torch.complex128)
    p_1 = -1j * dt * hamiltonian.clone()
    p_2 = (-1j * dt) * torch.einsum("lab,lbc->lac", p_1, hamiltonian)
    p_3 = (-1j * dt) * torch.einsum("lab,lbc->lac", p_2, hamiltonian)
    p_4 = (-1j * dt) * torch.einsum("lab,lbc->lac", p_3, hamiltonian)

    unitary = (
        identity[None, :, :] + p_1 + p_2 / 2 + p_3 / (2 * 3) + p_4 / (2 * 3 * 4)
    )  # torch.matrix_exp(-1j * dt * hamiltonian)

    psi = torch.einsum("lab,bl->al", unitary, psi)
    psi = psi / torch.linalg.norm(psi, dim=0)[None, :]
    return psi


def me_exponentiation_algorithm(
    hamiltonian: torch.ComplexType, psi: torch.ComplexType, dt: float
):
    identity = torch.eye(3)

    uni_dt_half = (
        identity[None, :, :] - (dt / 2) * hamiltonian
    )  # torch.matrix_exp(-dt * hamiltonian)
    uni_df_half = torch.linalg.inv(identity[None, :, :] + (dt / 2) * hamiltonian)
    unitary = torch.einsum("lab,lbc->lac", uni_df_half, uni_dt_half)

    # print(
    #     "unitary eig",
    #     torch.real(torch.linalg.eig(unitary)[0]) ** 2
    #     + torch.imag(torch.linalg.eig(unitary)[0]) ** 2,
    # )

    psi = torch.einsum("lab,bl->al", unitary, psi)
    return psi


# def time_step_backward_algorithm(
#     psi: torch.ComplexType,
#     h: torch.Tensor,
#     energy: nn.Module,
#     dt: float,
#     self_consistent_steps: int,
# ):
#     psi0 = psi.clone()
#     for i in range(self_consistent_steps):
#         hamiltonian, eng = get_the_hamiltonian(psi, h=h, energy=energy)
#         unitary = torch.matrix_exp(-1j * dt * hamiltonian)
#         psi1 = torch.einsum("lab,lb->la", unitary, psi)
#         # psi = psi0 - 1j * dt * torch.einsum("lab,lb->la", hamiltonian, psi)
#         psi1 = psi1 / torch.linalg.norm(psi1, dim=-1)[:, None]

#         hamiltonian1, eng1 = get_the_hamiltonian(psi1, h=h, energy=energy)
#         hamiltonian = 0.5 * (hamiltonian + hamiltonian1)
#         psi = torch.einsum("lab,lb->la", unitary, psi)
#         psi = psi / torch.linalg.norm(psi, dim=-1)[:, None]
#     return psi, eng


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


def heisemberg_matrix(omega_eff: torch.Tensor, h_eff: torch.Tensor):
    hm = torch.zeros(omega_eff.shape[-1], 3, 3, dtype=torch.double)
    hm[:, 0, 1] = h_eff
    hm[:, 1, 0] = -1 * h_eff
    hm[:, 1, 2] = omega_eff
    hm[:, 2, 1] = -1 * omega_eff

    return hm


def nonlinear_master_equation_step(
    psi: torch.Tensor,
    energy: torch.nn.Module,
    i: int,
    h: torch.Tensor,
    self_consistent_step: int,
    dt: float,
    eta: float,
):
    # m0 = torch.from_numpy(m_qutip_tot[q, i]).unsqueeze(0)

    omega_eff, engx = compute_the_gradient(
        m=psi.unsqueeze(0), h=h[i].unsqueeze(0), energy=energy, respect_to="x"
    )
    delta_eff, engy = compute_the_gradient(
        m=psi.unsqueeze(0), h=h[i].unsqueeze(0), energy=energy, respect_to="y"
    )
    h_eff, engz = compute_the_gradient(
        m=psi.unsqueeze(0), h=h[i].unsqueeze(0), energy=energy, respect_to="z"
    )

    hamiltonian0 = torch.zeros((psi.shape[-1], 3, 3))
    hamiltonian0[:, 0, 1] = 1 * h_eff[0]
    hamiltonian0[:, 1, 0] = -1 * h_eff[0]
    hamiltonian0[:, 0, 2] = -1 * delta_eff[0]
    hamiltonian0[:, 2, 0] = 1 * delta_eff[0]
    hamiltonian0[:, 1, 2] = 1 * omega_eff[0]
    hamiltonian0[:, 2, 1] = -1 * omega_eff[0]
    hamiltonian0 = 2 * hamiltonian0
    # hamiltonian0 = build_hamiltonian(
    #     field_x=-1 * omega_eff[0], field_z=-1 * h_eff[0]
    # )
    psi1 = me_exponentiation_algorithm(
        hamiltonian=hamiltonian0,
        psi=psi,
        dt=dt,
    )

    for step in range(self_consistent_step):
        # m1 = torch.from_numpy(m_qutip_tot[q, i + 1]).unsqueeze(0)

        # get the magnetization

        omega_eff1, eng = compute_the_gradient(
            m=psi1.unsqueeze(0), h=h[i + 1], energy=energy, respect_to="x"
        )
        h_eff1, _ = compute_the_gradient(
            m=psi1.unsqueeze(0), h=h[i + 1], energy=energy, respect_to="z"
        )
        delta_eff1, _ = compute_the_gradient(
            m=psi1.unsqueeze(0), h=h[i + 1], energy=energy, respect_to="y"
        )

        hamiltonian1 = torch.zeros((psi.shape[-1], 3, 3))
        hamiltonian1[:, 0, 1] = 1 * h_eff1[0]
        hamiltonian1[:, 1, 0] = -1 * h_eff1[0]
        hamiltonian1[:, 0, 2] = -1 * delta_eff1[0]
        hamiltonian1[:, 2, 0] = 1 * delta_eff1[0]
        hamiltonian1[:, 1, 2] = 1 * omega_eff1[0]
        hamiltonian1[:, 2, 1] = -1 * omega_eff1[0]
        hamiltonian1 = 2 * hamiltonian1

        psi1 = me_exponentiation_algorithm(
            hamiltonian=0.5 * (hamiltonian0 + hamiltonian1),
            psi=psi,
            dt=dt,
        )

    psi = me_exponentiation_algorithm(
        hamiltonian=0.5 * (hamiltonian0 + hamiltonian1),
        psi=psi,
        dt=dt,
    )

    return (
        psi,
        engx,
        engz,
        omega_eff,
        delta_eff,
        h_eff,
    )


def nonlinear_schrodinger_step(
    psi: torch.Tensor,
    energy: torch.nn.Module,
    i: int,
    h: torch.Tensor,
    self_consistent_step: int,
    dt: float,
    eta: float,
    exponent_algorithm: bool,
):
    z, x, y = compute_the_magnetization(psi=psi)
    m = torch.cat((z.view(1, -1), x.view(1, -1)), dim=0)
    m = m.unsqueeze(0)  # the batch dimension

    eng = energy(m, h[i].unsqueeze(0))[0].item()

    z_minus, x_minus, _ = compute_the_magnetization(psi=psi)
    m_minus = torch.cat((z_minus.view(1, -1), x_minus.view(1, -1)), dim=0)
    m_minus = m_minus.unsqueeze(0)  # the batch dimension

    # m0 = torch.from_numpy(m_qutip_tot[q, i]).unsqueeze(0)

    omega_eff, engx = compute_the_gradient(
        m=m_minus, h=h[i].unsqueeze(0), energy=energy, respect_to="x"
    )
    h_eff, engz = compute_the_gradient(
        m=m_minus, h=h[i].unsqueeze(0), energy=energy, respect_to="z"
    )

    hamiltonian_minus = build_hamiltonian(
        field_x=-1 * omega_eff[0], field_z=-1 * h_eff[0]
    )
    if exponent_algorithm:
        psi_minus = exponentiation_algorithm(
            hamiltonian=hamiltonian_minus, psi=psi, dt=dt
        )
    else:
        psi_minus = crank_nicolson_algorithm(
            hamiltonian=hamiltonian_minus, psi=psi, dt=dt
        )

    hamiltonian_plus = hamiltonian_minus.clone()

    for step in range(self_consistent_step):
        if exponent_algorithm:
            psi_plus = exponentiation_algorithm(
                hamiltonian=0.5 * (hamiltonian_minus + hamiltonian_plus),
                psi=psi,
                dt=dt,
            )
        else:
            psi_plus = crank_nicolson_algorithm(
                hamiltonian=0.5 * (hamiltonian_minus + hamiltonian_plus),
                psi=psi,
                dt=dt,
            )

        z_plus, x_plus, _ = compute_the_magnetization(psi=psi_plus)
        m_plus = torch.cat((z_plus.view(1, -1), x_plus.view(1, -1)), dim=0)
        m_plus = m_plus.unsqueeze(0)  # the batch dimension

        # m1 = torch.from_numpy(m_qutip_tot[q, i]).unsqueeze(0)

        omega_eff, eng = compute_the_gradient(
            m=m_plus, h=h[i + 1].unsqueeze(0), energy=energy, respect_to="x"
        )
        h_eff, _ = compute_the_gradient(
            m=m_plus, h=h[i + 1].unsqueeze(0), energy=energy, respect_to="z"
        )

        hamiltonian_plus = build_hamiltonian(
            field_x=-1 * omega_eff[0], field_z=-1 * h_eff[0]
        )

    if exponent_algorithm:
        psi = exponentiation_algorithm(
            hamiltonian=0.5 * (hamiltonian_plus + hamiltonian_minus),
            psi=psi,
            dt=dt,
        )
    else:
        psi = crank_nicolson_algorithm(
            hamiltonian=0.5 * (hamiltonian_minus + hamiltonian_plus),
            psi=psi,
            dt=dt,
        )

    # z, x, y = compute_the_magnetization(psi=psi)

    return psi, omega_eff, h_eff, eng, x, y, z


def nonlinear_ensamble_schrodinger_step(
    psis: List[torch.Tensor],
    energy: torch.nn.Module,
    i: int,
    h: torch.Tensor,
    self_consistent_step: int,
    dt: float,
    eta: float,
    exponent_algorithm: bool,
):
    ms_minus = torch.zeros((2, psis[0].shape[0]))
    xs = torch.zeros(psis[0].shape[0])
    ys = torch.zeros(psis[0].shape[0])
    zs = torch.zeros(psis[0].shape[0])

    for psi in psis:
        z, x, y = compute_the_magnetization(psi=psi)
        xs = xs + x
        ys = ys + y
        zs = zs + z
        m = torch.cat((z.view(1, -1), x.view(1, -1)), dim=0)
        m = m.unsqueeze(0)  # the batch dimension

        eng = energy(m, h[i].unsqueeze(0))[0].item()

        z_minus, x_minus, _ = compute_the_magnetization(psi=psi)
        m_minus = torch.cat((z_minus.view(1, -1), x_minus.view(1, -1)), dim=0)
        m_minus = m_minus.unsqueeze(0)  # the batch dimension

        ms_minus = ms_minus + m_minus
    ms_minus = ms_minus / len(psis)
    zs = zs / len(psis)
    xs = xs / len(psis)
    ys = ys / len(psis)

    omega_eff, engx = compute_the_gradient(
        m=ms_minus, h=h[i].unsqueeze(0), energy=energy, respect_to="x"
    )
    h_eff, engz = compute_the_gradient(
        m=ms_minus, h=h[i].unsqueeze(0), energy=energy, respect_to="z"
    )

    hamiltonian_minus = build_hamiltonian(
        field_x=-1 * omega_eff[0], field_z=-1 * h_eff[0]
    )

    hamiltonian_plus = hamiltonian_minus.clone()

    for step in range(self_consistent_step):
        ms_plus = torch.zeros((2, psis[0].shape[0]))
        for psi in psis:
            if exponent_algorithm:
                psi_plus = exponentiation_algorithm(
                    hamiltonian=0.5 * (hamiltonian_minus + hamiltonian_plus),
                    psi=psi,
                    dt=dt,
                )
            else:
                psi_plus = crank_nicolson_algorithm(
                    hamiltonian=0.5 * (hamiltonian_minus + hamiltonian_plus),
                    psi=psi,
                    dt=dt,
                )

            z_plus, x_plus, _ = compute_the_magnetization(psi=psi_plus)
            m_plus = torch.cat((z_plus.view(1, -1), x_plus.view(1, -1)), dim=0)
            m_plus = m_plus.unsqueeze(0)  # the batch dimension

            ms_plus = ms_plus + m_plus

        ms_plus = ms_plus / len(psis)
        # m1 = torch.from_numpy(m_qutip_tot[q, i]).unsqueeze(0)

        omega_eff, eng = compute_the_gradient(
            m=ms_plus, h=h[i + 1].unsqueeze(0), energy=energy, respect_to="x"
        )
        h_eff, _ = compute_the_gradient(
            m=ms_plus, h=h[i + 1].unsqueeze(0), energy=energy, respect_to="z"
        )

        hamiltonian_plus = build_hamiltonian(
            field_x=-1 * omega_eff[0], field_z=-1 * h_eff[0]
        )

    for i, psi in enumerate(psis):
        if exponent_algorithm:
            psis[i] = exponentiation_algorithm(
                hamiltonian=0.5 * (hamiltonian_plus + hamiltonian_minus),
                psi=psi,
                dt=dt,
            )
        else:
            psis[i] = crank_nicolson_algorithm(
                hamiltonian=0.5 * (hamiltonian_minus + hamiltonian_plus),
                psi=psi,
                dt=dt,
            )

    # z, x, y = compute_the_magnetization(psi=psi)

    return psis, omega_eff, h_eff, eng, xs, ys, zs


def nonlinear_schrodinger_step_zzx_model(
    psi: torch.Tensor,
    model: torch.nn.Module,
    i: int,
    h: torch.Tensor,
    self_consistent_step: int,
    dt: float,
    exponent_algorithm: bool,
):
    x, y, z = compute_the_magnetization(psi=psi)

    df_dz = compute_the_gradient_of_the_functional_ux_model(z=z, model=model)

    h_eff = df_dz + h[i]

    omega_eff = -1 * (
        torch.roll(x, shifts=-1, dims=-1) + torch.roll(x, shifts=+1, dims=-1)
    )

    hamiltonian_minus = build_hamiltonian(field_x=omega_eff, field_z=h_eff)

    hamiltonian_plus = hamiltonian_minus.clone()

    for step in range(self_consistent_step):
        if exponent_algorithm:
            psi_plus = exponentiation_algorithm(
                hamiltonian=0.5 * (hamiltonian_minus + hamiltonian_plus),
                psi=psi,
                dt=dt,
            )
        else:
            psi_plus = crank_nicolson_algorithm(
                hamiltonian=0.5 * (hamiltonian_minus + hamiltonian_plus),
                psi=psi,
                dt=dt,
            )

        x_plus, _, z_plus = compute_the_magnetization(psi=psi_plus)

        df_dz = compute_the_gradient_of_the_functional_ux_model(z=z_plus, model=model)
        h_eff = df_dz + h[i + 1]
        omega_eff = -1 * (
            torch.roll(x_plus, shifts=-1, dims=-1)
            + torch.roll(x_plus, shifts=+1, dims=-1)
        )

        hamiltonian_plus = build_hamiltonian(field_x=omega_eff, field_z=h_eff)

    if exponent_algorithm:
        psi = exponentiation_algorithm(
            hamiltonian=0.5 * (hamiltonian_plus + hamiltonian_minus),
            psi=psi,
            dt=dt,
        )
    else:
        psi = crank_nicolson_algorithm(
            hamiltonian=0.5 * (hamiltonian_minus + hamiltonian_plus),
            psi=psi,
            dt=dt,
        )
    return psi, omega_eff, df_dz, z


def nonlinear_schrodinger_step_zzx_model_full_effective_field(
    psi: torch.Tensor,
    model: torch.nn.Module,
    i: int,
    h: torch.Tensor,
    full_z: torch.Tensor,  # full z in size x time
    self_consistent_step: int,
    dt: float,
    exponent_algorithm: bool,
    #    dataset_z: torch.Tensor,
):

    # if i == 0:
    #     dataset = dataset_z[:, 0, :].unsqueeze(1)
    # else:
    #     dataset = dataset_z[:, : i + 1, :]

    # full_z_proj = z_dataset_projection(z=full_z, dataset=dataset)

    df_dz = get_effective_field(z=full_z, model=model, i=-1)
    h_eff = 0.5 * (h[i])  # + df_dz)

    omega_eff = 0.5 * torch.ones_like(h_eff)
    hamiltonian = torch.zeros((h_eff.shape[0], 2, 2), dtype=torch.complex128)
    hamiltonian = build_hamiltonian(field_x=omega_eff, field_z=h_eff)

    psi = exponentiation_algorithm(
        hamiltonian=hamiltonian,
        psi=psi,
        dt=dt,
    )

    exp_hamiltonian = torch.matrix_exp(-1j * dt * hamiltonian)
    psi = torch.einsum("lab,bl->al", exp_hamiltonian, psi)

    hamiltonian_minus = hamiltonian.clone()
    hamiltonian_plus = hamiltonian_minus.clone()
    x, y, z = compute_the_magnetization(psi=psi)

    for step in range(self_consistent_step):

        # psi_plus = exponentiation_algorithm(
        #     hamiltonian=0.5 * (hamiltonian_minus + hamiltonian_plus),
        #     psi=psi,
        #     dt=dt,
        # )

        exp_hamiltonian_plus = torch.matrix_exp(
            -1j * dt * 0.5 * (hamiltonian_minus + hamiltonian_plus)
        )
        psi_plus = torch.einsum("lab,bl->al", exp_hamiltonian_plus, psi)

        _, _, z_plus = compute_the_magnetization(psi=psi_plus)
        full_z_plus = torch.cat((full_z, z_plus.unsqueeze(0)), dim=0)
        # full_z_plus_proj = z_dataset_projection(z=full_z_plus, dataset=dataset)

        df_dz = get_effective_field(z=full_z_plus, model=model, i=-1)
        h_eff = 0.5 * (h[i + 1])  # + df_dz)

        hamiltonian_plus = build_hamiltonian(field_x=omega_eff, field_z=h_eff)

    # psi = exponentiation_algorithm(
    #     hamiltonian=0.5 * (hamiltonian_plus + hamiltonian_minus),
    #     psi=psi,
    #     dt=dt,
    # )

    full_z = torch.cat((full_z, z.unsqueeze(0)), dim=0)
    return psi, df_dz, full_z


def nonlinear_schrodinger_step_zzxz_model(
    psi: torch.Tensor,
    model: torch.nn.Module,
    i: int,
    h: torch.Tensor,
    self_consistent_step: int,
    dt: float,
    exponent_algorithm: bool,
):
    x, y, z = compute_the_magnetization(psi=psi)

    df_dz = compute_the_gradient_of_the_functional_ux_model(z=z, model=model)

    h_eff = df_dz + h[i, 0]

    omega_eff = (
        -1 * (torch.roll(x, shifts=-1, dims=-1) + torch.roll(x, shifts=+1, dims=-1))
        - h[i, 1]
    )

    hamiltonian_minus = build_hamiltonian(field_x=omega_eff, field_z=h_eff)

    hamiltonian_plus = hamiltonian_minus.clone()

    for step in range(self_consistent_step):
        if exponent_algorithm:
            psi_plus = exponentiation_algorithm(
                hamiltonian=0.5 * (hamiltonian_minus + hamiltonian_plus),
                psi=psi,
                dt=dt,
            )
        else:
            psi_plus = crank_nicolson_algorithm(
                hamiltonian=0.5 * (hamiltonian_minus + hamiltonian_plus),
                psi=psi,
                dt=dt,
            )

        x_plus, _, z_plus = compute_the_magnetization(psi=psi_plus)

        df_dz = compute_the_gradient_of_the_functional_ux_model(z=z_plus, model=model)
        h_eff = df_dz + h[i + 1, 0]
        omega_eff = (
            -1
            * (
                torch.roll(x_plus, shifts=-1, dims=-1)
                + torch.roll(x_plus, shifts=+1, dims=-1)
            )
            - h[i + 1, 1]
        )

        hamiltonian_plus = build_hamiltonian(field_x=omega_eff, field_z=h_eff)

    if exponent_algorithm:
        psi = exponentiation_algorithm(
            hamiltonian=0.5 * (hamiltonian_plus + hamiltonian_minus),
            psi=psi,
            dt=dt,
        )
    else:
        psi = crank_nicolson_algorithm(
            hamiltonian=0.5 * (hamiltonian_minus + hamiltonian_plus),
            psi=psi,
            dt=dt,
        )
    return psi, omega_eff, h_eff, z


def get_effective_field(z: torch.tensor, model: nn.Module, i: int):
    z = torch.einsum("ti->it", z)
    effective_field = model(z.unsqueeze(0).double())[0, 0, :, i].detach()
    return effective_field


def z_pca(z: torch.Tensor, dataset: torch.Tensor):

    mu = dataset.mean(0)
    sigma = dataset.std(0)
    # x = (dataset - mu[None, :, :]) / sigma[None, :, :]
    x = dataset

    cov_matrix = torch.einsum("ati,bti->ab", x, x) * (1 / (x.shape[0] - 1))
    _, eigenvectors = torch.linalg.eigh(cov_matrix)
    pca = torch.einsum("ab,bti->ati", eigenvectors, x) * (1 / (x.shape[0] - 1))

    x_value = (z - mu) / sigma
    z_a = torch.einsum("ati,ti->a", pca, z)
    z_proj = torch.einsum("a,ati->ti", z_a, x)
    # z_proj = sigma * x_proj + mu

    return z_proj


def z_dataset_projection(z: torch.Tensor, dataset: torch.Tensor):

    coeff = torch.einsum("ati,ti->ai", dataset, z) / (
        torch.einsum("ati,ati->ai", dataset, dataset)
    )

    z_projection = torch.mean(coeff[:, None, :] * dataset, dim=0)

    return z_projection
