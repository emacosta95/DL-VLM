# %% Imports


import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from src.training.models_adiabatic import EnergyReductionXXZ
from src.qutip_lab.qutip_class import SpinOperator, SpinHamiltonian, SteadyStateSolver

from src.tddft_methods.kohm_sham_utils import (
    initialize_psi_from_z,
    nonlinear_schrodinger_step_zzx_model,
)
from src.gradient_descent import GradientDescentKohmSham
import qutip
from typing import List
import os


### SET NUM THREADS
# os.environ["OMP_NUM_THREADS"] = "3"
# os.environ["NUMEXPR_NUM_THREADS"] = "3"
# os.environ["MKL_NUM_THREADS"] = "3"
# torch.set_num_threads(3)


# %% Qutip details
class Driving:
    def __init__(self, h_i: np.array, h_f: np.array, rate: float, idx: int) -> None:
        self.hi = h_i
        self.hf = h_f
        self.rate = rate
        self.idx: int = idx

    def field(self, t: float, args):
        return (
            self.hi[self.idx] * np.exp(-t * self.rate)
            + (1 - np.exp(-t * self.rate)) * self.hf[self.idx]
        )

    def get_the_field(self, t: np.ndarray):
        return (
            self.hi[None, :] * np.exp(-t[:, None] * self.rate)
            + (1 - np.exp(-t[:, None] * self.rate)) * self.hf[None, :]
        )


class PeriodicDriving:
    def __init__(self, h_i: np.array, delta: np.array, rate: float, idx: int) -> None:
        self.hi = h_i
        self.delta = delta
        self.rate = rate
        self.idx: int = idx

    def field(self, t: float, args):
        return self.hi[self.idx] + (self.delta[self.idx]) * np.sin(self.rate * t)

    def get_the_field(self, t: np.ndarray):
        return self.hi[None, :] + (self.delta[None, :]) * np.sin(self.rate * t)[:, None]


# %% Data


l = 8

model = torch.load(
    "model_rep/kohm_sham/disorder/zzx_model/model_zzx_dataset_fields_0.0_5.0_j_-1_1nn_n_800k_unet_l_train_8_[60, 60, 60, 60, 60, 60]_hc_5_ks_1_ps_6_nconv_0_nblock",
    map_location="cpu",
)
model.eval()
model = model.to(dtype=torch.double)
energy = EnergyReductionXXZ(model=model)
energy.eval()


# initialization
exponent_algorithm = True
self_consistent_step = 1
steps = 1000
tf = 100.0
time = torch.linspace(0.0, tf, steps)
dt = time[1] - time[0]

rates = np.array([0, 0.1, 0.3, 0.5, 0.8, 1])
ndata = rates.shape[0]


h_tot = np.zeros((ndata, steps, l))
z_qutip_tot = np.zeros((ndata, steps, l))
z_tot = np.zeros((ndata, steps, l))

eng_tot = np.zeros((ndata, steps))
eng_qutip_tot = np.zeros((ndata, steps))
gradients_tot = np.zeros((ndata, steps, 2, l))


# is the driving periodic?
periodic = False

# define the initial external field
# zz x quench style (?)
hi = torch.ones(l)
# high transverse field

# define the final external field
hf = 0.5 + torch.rand(l)


# define the delta for the periodic driving
delta = 0.8 * torch.ones((l))


# %% Compute the initial ground state configuration

gd = GradientDescentKohmSham(
    loglr=-2,
    energy=energy,
    epochs=1000,
    seed=23,
    num_threads=3,
    device="cpu",
    n_init=-0.9 * np.ones(l),
    h=hi,
)


zi = gd.run()
zi = torch.from_numpy(zi)[0]

for q, rate in enumerate(rates):
    # Qutip Dynamics
    # Hamiltonian
    ham0 = SpinHamiltonian(
        direction_couplings=[("x", "x")],
        pbc=True,
        coupling_values=[-1.0],
        size=l,
    )

    hamExtZ = SpinOperator(
        index=[("z", i) for i in range(l)], coupling=hi.detach().numpy(), size=l
    )

    eng, psi0 = np.linalg.eigh(ham0.qutip_op + hamExtZ.qutip_op)
    psi0 = qutip.Qobj(psi0[:, 0], shape=psi0.shape, dims=([[2 for i in range(l)], [1]]))

    print("real ground state energy=", eng[0])
    # to check if we have the same outcome with the Crank-Nicholson algorithm
    # psi = initialize_psi_from_z_and_x(z=-1 * zi[0], x=zi[1])
    # psi = psi.detach().numpy()
    # for i in range(l):
    #     psi_l = qutip.Qobj(psi[i], shape=psi[i].shape, dims=([[2], [1]]))
    #     if i == 0:
    #         psi0 = psi_l
    #     else:
    #         psi0 = qutip.tensor(psi0, psi_l)
    # compute and check the magnetizations
    obs: List[qutip.Qobj] = []
    for i in range(l):
        z_op = SpinOperator(index=[("z", i)], coupling=[1.0], size=l, verbose=1)

        print(z_op.expect_value(psi=psi0) - zi[i].detach().numpy())
        obs.append(z_op.qutip_op)

    print("\n INITIALIZE THE HAMILTONIAN \n")
    # build up the time dependent object for the qutip evolution
    hamiltonian = [ham0.qutip_op]

    print("periodic=", periodic, "\n")
    for i in range(l):
        if periodic:
            drive_z = PeriodicDriving(
                h_i=hi.detach().numpy(),
                delta=delta.detach().numpy(),
                rate=rate,
                idx=i,
            )
        else:
            drive_z = Driving(
                h_i=hi.detach().numpy(),
                h_f=hf.detach().numpy(),
                rate=rate,
                idx=i,
            )

        hamiltonian.append([obs[i], drive_z.field])

    h_z = drive_z.get_the_field(time.detach().numpy()).reshape(time.shape[0], -1)

    h_tot[q] = h_z
    h = torch.from_numpy(h_z)
    print(h.shape)

    # evolution

    output = qutip.sesolve(hamiltonian, psi0, time.detach().numpy(), e_ops=obs)

    # %% visualization
    for r in range(l):
        z_qutip_tot[q, :, r] = output.expect[r]

    #  Kohm Sham step 1) Initialize the state from an initial magnetization
    psi = initialize_psi_from_z(z=-1 * zi)
    # psi = initialize_psi_from_xyz(z=-1 * zi[0], x=zi[1], y=torch.zeros_like(zi[1]))

    t_bar = tqdm(enumerate(time))
    for i in trange(time.shape[0] - 1):
        t = time[i]

        psi, omega_eff, h_eff, z = nonlinear_schrodinger_step_zzx_model(
            psi=psi,
            model=model,
            i=i,
            h=h,
            self_consistent_step=self_consistent_step,
            dt=dt,
            exponent_algorithm=exponent_algorithm,
        )

        z_tot[q, i, :] = z.detach().numpy()
        gradients_tot[q, i, 1, :] = -1 * omega_eff.detach().numpy()
        gradients_tot[q, i, 0, :] = -1 * h_eff.detach().numpy()

        if periodic:
            np.savez(
                f"data/kohm_sham_approach/results/dl_functional/zzx_model/periodic/tddft_periodic_uniform_zzxxzx_model_h_0_5_omega_0_2_ti_0_tf_{tf:.0f}_hi_{hi[0].item():.4f}_delta_{delta[0].item():.4f}_steps_{steps}_self_consistent_steps_{self_consistent_step}_ndata_{ndata}_exp_{exponent_algorithm}_size_{l}",
                z_qutip=z_qutip_tot[:, :i],
                z=z_tot[:, :i],
                potential=h_tot[:, :i],
                gradient=gradients_tot[:, :i],
                rates=rates,
                time=time[:i],
            )

        else:
            np.savez(
                f"data/kohm_sham_approach/results/dl_functional/zzx_model/non_uniform/tddft_quench_model_h_0_2_omega_0_2_ti_0_tf_{tf:.0f}_hi_{hi.mean().item():.1f}_hf_{hf.mean().item():.1f}_steps_{steps}_self_consistent_steps_{self_consistent_step}_ndata_{ndata}_exp_{exponent_algorithm}_size_{l}",
                z_qutip=z_qutip_tot[:, :i],
                z=z_tot[:, :i],
                potential=h_tot[:, :i],
                gradient=gradients_tot[:, :i],
                rates=rates,
                time=time[:i],
            )
