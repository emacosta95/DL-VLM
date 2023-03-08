from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import R2Score
from tqdm import trange
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from src.training.models import Energy_unet
from src.training.utils import count_parameters

# Methods from quantum_ising_simulation


def parallel_nambu_diagonalization_ising_model(
    nbatch, l: int, j_coupling: float, hs: np.array, device: str, pbc: bool
):
    """Compute the correlation <\sigma_x \sigma_x>(ij) of the transverse quantum ising model using the Nambu Mapping

    Args:
        l (int): length of the chain
        j_coupling (float): coupling costant of the spin interaction
        hs (np.array): realizations of the magnetic field [batch,l]
        checkpoint (bool): if True, it creates a npz version of the dataset
        train (bool): if True, it labels the file as train, valid otherwise
        device(str): the device used for the computation. Can be either 'cuda' or 'cpu' (standard is 'cpu').
    Returns:
        e,f,m_z (Tuple[np.array]): a triple of energies, H-K functional values and transverse magnetizations for each hs realizations
    """

    n_dataset = hs.shape[0]

    batch = int(n_dataset / nbatch)
    # uniform means h_ave=0
    hs = torch.tensor(hs, dtype=torch.double, device=device)

    # obc
    j_vec = j_coupling * torch.ones(l, device=device)
    # the 0-th component is null in OBC
    j_vec_l = j_vec.clone()
    if not (pbc):
        j_vec_l[0] = 0
    if pbc:
        j_vec_l[0] = -1 * j_vec_l[0]

    # the l-th component is null in OBC
    j_vec_r = j_vec.clone()
    if not (pbc):
        j_vec_r[-1] = 0
    if pbc:
        j_vec_r[-1] = -1 * j_vec_r[-1]

    # create the nambu matrix

    # create the j matrix in the nearest neighbourhood case
    j_l = torch.einsum("ij,j->ij", torch.eye(l, device=device), j_vec_l)
    j_l = torch.roll(j_l, shifts=-1, dims=1)
    j_r = torch.einsum("ij,j->ij", torch.eye(l, device=device), j_vec_r)
    j_r = torch.roll(j_r, shifts=1, dims=1)
    # the coupling part for a
    j = -0.5 * (j_r + j_l)
    # the coupling part for b
    j_b = -0.5 * (j_r - j_l)
    # the b matrix of the nambu matrix
    b = j_b

    for i in trange(nbatch):
        # the external field
        h = hs[i * batch : (i + 1) * batch]
        h_matrix = torch.einsum("ij,aj->aij", torch.eye(l, device=device), h)
        # the a matrix of the nambu matrix
        a = j + h_matrix

        # create the nambu matrix
        h_nambu = torch.zeros((batch, 2 * l, 2 * l), device=device)
        h_nambu[:, :l, :l] = a
        h_nambu[:, :l, l:] = b
        h_nambu[:, l:, :l] = -1 * torch.conj(b)
        h_nambu[:, l:, l:] = -1 * torch.conj(a)

        e, w = torch.linalg.eigh(h_nambu)

        # the v coefficients
        v = w.clone()[:, l:, :l]

        u = w.clone()[:, :l, :l]
        # compute the correlation sigma_x sigma_x
        c_vv = torch.einsum("anl,aml->anm", v, torch.conj(v))
        c_uu = torch.einsum("anl,aml->anm", u, torch.conj(u))
        c_vu = torch.einsum("anl,aml->anm", v, torch.conj(u))
        c_uv = torch.einsum("anl,aml->anm", u, torch.conj(v))
        c = c_vv + c_vu - c_uu - c_uv

        s_z = 1 - 2 * torch.einsum("aik,aik->ai", v, torch.conj(v))
        s_z_different = torch.einsum("aik,aik->ai", u, torch.conj(u)) - torch.einsum(
            "aik,aik->ai", v, torch.conj(v)
        )

        density_f = c[:, np.arange(l), (np.arange(l) + 1) % l]
        density_f[:, -1] = -1 * density_f[:, -1]

        e_0 = torch.sum(e[:, 0:l], dim=-1) / l
        f = e_0 - torch.mean(h * s_z, dim=-1)

        if i == 0:
            magn_z = s_z
            magn_z_diff = s_z_different
            e_tot = e_0
            f_tot = f
            tot_density_f = density_f
        else:
            magn_z = np.append(magn_z, s_z, axis=0)
            magn_z_diff = np.append(magn_z_diff, s_z_different, axis=0)
            e_tot = np.append(e_tot, e_0)
            f_tot = np.append(f_tot, f)
            tot_density_f = np.append(tot_density_f, density_f, axis=0)

    return hs, magn_z, magn_z_diff, f_tot, tot_density_f, e_tot


def parallel_nambu_correlation_ising_model(
    nbatch,
    l: int,
    j_coupling: float,
    hs: np.array,
    device: str,
    name_file: str,
    pbc: bool,
):
    """Compute the correlation <\sigma_x \sigma_x>(ij) of the transverse quantum ising model using the Nambu Mapping

    Args:
        l (int): length of the chain
        j_coupling (float): coupling costant of the spin interaction
        hs (np.array): realizations of the magnetic field [batch,l]
        checkpoint (bool): if True, it creates a npz version of the dataset
        train (bool): if True, it labels the file as train, valid otherwise
        device(str): the device used for the computation. Can be either 'cuda' or 'cpu' (standard is 'cpu').
    Returns:
        e,f,m_z (Tuple[np.array]): a triple of energies, H-K functional values and transverse magnetizations for each hs realizations
    """

    n_dataset = hs.shape[0]

    batch = int(n_dataset / nbatch)
    # uniform means h_ave=0
    hs = torch.tensor(hs, dtype=torch.double, device=device)

    # obc
    j_vec = j_coupling * torch.ones(l, device=device)
    # the 0-th component is null in OBC
    j_vec_l = j_vec.clone()
    if not (pbc):
        j_vec_l[0] = 0
    if pbc:
        j_vec_l[0] = -1 * j_vec_l[0]

    # the l-th component is null in OBC
    j_vec_r = j_vec.clone()
    if not (pbc):
        j_vec_r[-1] = 0
    if pbc:
        j_vec_r[-1] = -1 * j_vec_r[-1]

    # create the nambu matrix

    # create the j matrix in the nearest neighbourhood case
    j_l = torch.einsum("ij,j->ij", torch.eye(l, device=device), j_vec_l)
    j_l = torch.roll(j_l, shifts=-1, dims=1)
    j_r = torch.einsum("ij,j->ij", torch.eye(l, device=device), j_vec_r)
    j_r = torch.roll(j_r, shifts=1, dims=1)
    # the coupling part for a
    j = -0.5 * (j_r + j_l)
    # the coupling part for b
    j_b = -0.5 * (j_r - j_l)
    # the b matrix of the nambu matrix
    b = j_b

    for i in trange(nbatch):
        ss_x = torch.zeros((batch, l, l))
        ss_z = torch.zeros((batch, l, l))
        ss_s_s_z = torch.zeros((batch, l, l))
        # the external field
        h = hs[i * batch : (i + 1) * batch]
        h_matrix = torch.einsum("ij,aj->aij", torch.eye(l, device=device), h)
        # the a matrix of the nambu matrix
        a = j + h_matrix

        # create the nambu matrix
        h_nambu = torch.zeros((batch, 2 * l, 2 * l), device=device)
        h_nambu[:, :l, :l] = a
        h_nambu[:, :l, l:] = b
        h_nambu[:, l:, :l] = -1 * torch.conj(b)
        h_nambu[:, l:, l:] = -1 * torch.conj(a)

        e, w = torch.linalg.eigh(h_nambu)

        # the v coefficients
        v = w.clone()[:, l:, :l]

        u = w.clone()[:, :l, :l]
        # compute the correlation sigma_x sigma_x
        c_vv = torch.einsum("anl,aml->anm", v, torch.conj(v))
        c_uu = torch.einsum("anl,aml->anm", u, torch.conj(u))
        c_vu = torch.einsum("anl,aml->anm", v, torch.conj(u))
        c_uv = torch.einsum("anl,aml->anm", u, torch.conj(v))
        c = c_vv + c_vu - c_uu - c_uv

        s_z = 1 - 2 * torch.einsum("aik,aik->ai", v, torch.conj(v))
        for k in range(l):
            for r in range(k, l):
                if r != k:
                    ss_z[:, k, r] = ss_z[:, k, r] + (
                        c[:, k, k] * c[:, r, r] - c[:, k, r] * c[:, r, k]
                    )
                    ss_z[:, r, k] = ss_z[:, r, k] + ss_z[:, k, r]
                    # print(f'k={k},j={j}')
                    ss_x[:, k, r] = ss_x[:, k, r] + torch.linalg.det(
                        c[:, k:r, k + 1 : r + 1]
                    )
                    ss_x[:, r, k] = ss_x[:, r, k] + ss_x[:, k, r]
                    # print(c[k:j,k+1:j+1])
                    # print(c,k,j)
                else:
                    ss_x[:, k, r] = 1.0
                    ss_z[:, k, r] = 1
        if i == 0:
            corr_zz = ss_z
            magn_z = s_z
            corr_xx = ss_x
        else:
            corr_zz = np.append(corr_zz, ss_z, axis=0)
            corr_xx = np.append(corr_xx, ss_x, axis=0)
            magn_z = np.append(magn_z, s_z, axis=0)

    return corr_xx, corr_zz, magn_z
