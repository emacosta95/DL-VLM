# %% Imports


import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from src.training.models_adiabatic import EnergyReductionXXZ
from src.qutip_lab.qutip_class import SpinOperator, SpinHamiltonian, SteadyStateSolver

from src.tddft_methods.kohm_sham_utils import (
    initialize_psi_from_z,
    nonlinear_schrodinger_step_zzx_model_full_effective_field,
    get_effective_field,
)
from src.gradient_descent import GradientDescentKohmSham
import qutip
from typing import List
import os

# Set the seed for generating random numbers
np.random.seed(42)
torch.manual_seed(42)


def generate_smooth_gaussian_noise(
    time: np.ndarray,
    tau: float,
    tf: float,
    mean: float,
    sigma: float,
    min_range: float,
    max_range: float,
    shift: float,
):
    a_omegas = np.random.normal(mean, sigma, size=time.shape[0])
    omegas = np.linspace(0, time.shape[0] * 2 * np.pi / tf, time.shape[0])
    driving = np.zeros(time.shape[0])

    for tr in range(time.shape[0]):
        if omegas[tr] < 2 * np.pi / tau:
            driving = driving + a_omegas[tr] * np.cos(omegas[tr] * time)

        else:
            break

    max_driving = np.max(driving)
    min_driving = np.min(driving)

    old_interval = max_driving - min_driving
    driving = (
        (driving - min_driving) * (max_range - min_range) / old_interval
        + min_range
        + shift
    )

    return driving


class Driving:
    def __init__(self, h: np.array, idx: int, dt: float) -> None:
        self.h = h
        # self.tf=tf
        self.idx: int = idx
        self.dt: float = dt

    def field(self, t: float, args):
        return self.h[int(t / self.dt), self.idx]

    def get_the_field(
        self,
    ):
        return self.h


### SET NUM THREADS
# os.environ["OMP_NUM_THREADS"] = "3"
# os.environ["NUMEXPR_NUM_THREADS"] = "3"
# os.environ["MKL_NUM_THREADS"] = "3"
# torch.set_num_threads(3)


# %% Data and Hyperparameters


l = 8

model = torch.load(
    "model_rep/kohm_sham/model_current_t_interval_500_240202_[40, 40, 40, 40, 40, 40]_hc_[3, 5]_ks_1_ps_6_nconv_1_nblock",
    map_location="cpu",
)
model.eval()
model = model.to(dtype=torch.double)

# dataset for the driving
data_file_name = "dataset_max_shift_0.6_nbatch_100_batchsize_100_steps_1000_tf_30.0.npz"

data = np.load("data/dataset_h_eff/" + data_file_name)

h_tot = data["h"]
z_exact = data["z"]
h_eff_exact = data["h_eff"]
print(h_tot[0])
print(h_tot.shape)
# initialization
exponent_algorithm = True
self_consistent_step = 0
nbatch = z_exact.shape[0]

steps = 1000
tf = 30.0
time = np.linspace(0.0, tf, steps)
dt = time[1] - time[0]
print(dt)

z_tot = np.zeros((nbatch, steps, l))
h_eff_tot = np.zeros_like(z_tot)


# %% Compute the initial ground state configuration


for q in trange(nbatch):
    # Qutip Dynamics
    # Hamiltonian
    h = h_tot[q]

    # build up the time dependent object for the qutip evolution

    input = torch.einsum("ti->it", torch.tensor(z_tot[q]))
    heff = model(input.unsqueeze(0)).detach().squeeze()
    heff = torch.einsum("it->ti", heff)
    # initial magnetization
    z_evolution = (
        torch.tensor(z_tot[q, 0]).clone().unsqueeze(0)
    )  # initialize the evolution
    #  Kohm Sham step 1) Initialize the state from an initial magnetization

    # density matrix initialization
    psi = initialize_psi_from_z(z=-1 * torch.tensor(z_tot[q, 0]))

    h_eff = torch.zeros((steps, l))
    t_bar = tqdm(enumerate(time))
    for i in trange(steps - 1):
        t = time[i]
        psi, df, z_evolution = (
            nonlinear_schrodinger_step_zzx_model_full_effective_field(
                psi=psi,
                model=model,
                i=i,
                h=torch.tensor(h_tot[q]) + heff,
                full_z=z_evolution,  # full z in size x time
                self_consistent_step=self_consistent_step,
                dt=dt,
                exponent_algorithm=exponent_algorithm,
            )
        )
        h_eff[i] = df

    h_eff_tot[q] = h_eff
    z_tot[q] = z_evolution.detach().numpy()

    if q % 100 == 0:
        np.savez(
            "data/dataset_h_eff/reconstruction_dataset/reconstruction_"
            + data_file_name,
            z_exact=z_exact,
            h=h_tot,
            h_eff_exact=h_eff_exact,
            h_eff=h_eff_tot,
        )


np.savez(
    "data/dataset_h_eff/reconstruction_dataset/reconstruction_" + data_file_name,
    z_exact=z_exact,
    h=h_tot,
    h_eff_exact=h_eff_exact,
    h_eff=h_eff_tot,
)
