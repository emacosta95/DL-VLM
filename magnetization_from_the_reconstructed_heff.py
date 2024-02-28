# %% Imports


import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from src.training.models_adiabatic import EnergyReductionXXZ
from src.qutip_lab.qutip_class import SpinOperator, SpinHamiltonian, SteadyStateSolver

from src.tddft_methods.kohm_sham_utils import (
    initialize_psi_from_z_parallel,
    nonlinear_schrodinger_step_zzx_model_full_effective_field_parallel,
    get_effective_field,
    parallelized_compute_the_magnetization,
)
from src.gradient_descent import GradientDescentKohmSham
import qutip
from typing import List
import os

# Set the seed for generating random numbers
np.random.seed(42)
torch.manual_seed(42)


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
    "model_rep/kohm_sham/cnn_density2field/model_density2field_t_interval_500_240211_quench_dataset_[80, 80, 80, 80, 80, 80]_hc_[3, 9]_ks_1_ps_6_nconv_1_nblock",
    map_location="cpu",
)
model.eval()
model = model.to(dtype=torch.double)

# dataset for the driving
data_file_name = "dataset_quench_nbatch_10_batchsize_100_steps_1000_tf_30.0_l_8.npz"

data = np.load("data/dataset_h_eff/quench/" + data_file_name)

h_tot = data["h"][:, :]
z_exact = data["z"][:, :]
h_eff_exact = data["h_eff"][:, :]
print(h_tot[0])
print(h_tot.shape)
print("z_exact_shape=", z_exact.shape)
# initialization
exponent_algorithm = True
self_consistent_step = 0
nbatch = 1
batch_size = 2

steps = 1000
tf = 30.0
time = np.linspace(0.0, tf, steps)
dt = time[1] - time[0]
print(dt)

z_tot = np.zeros((nbatch * batch_size, steps, l))
h_eff_tot = np.zeros_like(z_tot)
h_eff_reconstruction = np.zeros_like(z_tot)

# %% Compute the initial ground state configuration


for q in trange(nbatch):
    # Qutip Dynamics
    # Hamiltonian
    h = h_tot[q * batch_size : (q + 1) * (batch_size)]

    # build up the time dependent object for the qutip evolution

    input = torch.einsum(
        "rti->rit",
        torch.tensor(z_exact[q * batch_size : (q + 1) * (batch_size), :steps]),
    )
    heff = model(input).detach().squeeze()
    heff = torch.einsum("rit->rti", heff)
    # initialize the evolution

    #  Kohm Sham step 1) Initialize the state from an initial magnetization

    # initialize psi
    zi = torch.tensor(z_exact[q * batch_size : (q + 1) * (batch_size), 0])
    # print(zi.shape)
    # initial magnetization
    z_evolution = zi.clone().unsqueeze(1)

    psi = initialize_psi_from_z_parallel(z=-1 * zi)

    _, _, z_test = parallelized_compute_the_magnetization(psi)
    # print("TEST=", z_test - zi)

    h_eff = torch.zeros((h.shape[0], steps, l))
    t_bar = tqdm(enumerate(time))
    for i in trange(steps - 1):
        t = time[i]
        psi, df, z_evolution = (
            nonlinear_schrodinger_step_zzx_model_full_effective_field_parallel(
                psi=psi,
                model=model,
                i=i,
                h=torch.tensor(h[:, :steps])
                + torch.tensor(heff[q * batch_size : (q + 1) * (batch_size), :steps]),
                full_z=z_evolution,  # full z in size x time
                self_consistent_step=self_consistent_step,
                dt=dt,
                exponent_algorithm=exponent_algorithm,
            )
        )
        h_eff[:, i] = df

    h_eff_tot[q * batch_size : (q + 1) * (batch_size)] = h_eff
    z_tot[q * batch_size : (q + 1) * (batch_size)] = z_evolution.detach().numpy()
    h_eff_reconstruction[q * batch_size : (q + 1) * (batch_size)] = (
        heff.detach().numpy()
    )

    np.savez(
        "data/dataset_h_eff/reconstruction_dataset/reconstruction_" + data_file_name,
        z_exact=z_exact,
        h=h_tot,
        h_eff_exact=h_eff_exact,
        h_eff=h_eff_tot,
        h_eff_reconstruction=h_eff_reconstruction,
        z=z_tot,
    )
