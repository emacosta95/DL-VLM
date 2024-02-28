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
    "model_rep/kohm_sham/cnn_density2field/model_density2field_t_interval_500_240211_quench_dataset_[80, 80, 80, 80, 80, 80]_hc_[3, 9]_ks_1_ps_6_nconv_1_nblock",
    map_location="cpu",
)
model.eval()
model = model.to(dtype=torch.double)

# dataset for the driving
data = np.load(
    "data/dataset_h_eff/quench/dataset_quench_nbatch_10_batchsize_100_steps_1000_tf_30.0_l_8.npz"
)

h_tot = data["h"][300:400]
dataset_z = data["z"][300:400]


h_eff_exact = data["h_eff"][300:400]
print(h_tot[0])
print(h_tot.shape)
# initialization
exponent_algorithm = True
self_consistent_step = 0
nbatch = 1


rate = 0.2

min_range_driving = 0.01
max_range_driving = 1.0

shift = 0.5

steps = 1000
tf = 30.0
time = np.linspace(0.0, tf, steps)
dt = time[1] - time[0]
print(dt)

z_qutip_tot = np.zeros((nbatch, steps, l))
z_tot = np.zeros_like(z_qutip_tot)
h_eff_tot = np.zeros_like(z_qutip_tot)
h_eff_tot_exact=np.zeros_like(z_qutip_tot)

current_qutip_tot = np.zeros_like(z_qutip_tot)
current_derivative_tot = np.zeros_like(z_qutip_tot)

ham0 = SpinHamiltonian(
    direction_couplings=[("x", "x")],
    pbc=True,
    coupling_values=[-1.0],
    size=l,
)

obs: List[qutip.Qobj] = []
current_obs: List[qutip.Qobj] = []
for i in range(l):
    z_op = SpinOperator(index=[("z", i)], coupling=[1.0], size=l, verbose=1)
    # print(f"x[{i}]=", x.qutip_op, "\n")
    current = SpinOperator(
        index=[("x", (i - 1) % l, "y", i), ("y", i, "x", (i + 1) % l)],
        coupling=[-2, -2],
        size=l,
    )
    obs.append(z_op.qutip_op)
    current_obs.append(current.qutip_op)

time_stop = 500

# %% Compute the initial ground state configuration

print('ITS OK!')
for q in range(nbatch):
    # Qutip Dynamics
    # Hamiltonian
    h = h_tot[q]

    hamExtZ = SpinOperator(index=[("z", i) for i in range(l)], coupling=h[0], size=l)

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

    # build up the time dependent object for the qutip evolution
    hamiltonian = [ham0.qutip_op]

    for i in range(l):
        drive_z = Driving(
            h=h,
            dt=time[1] - time[0],
            idx=i,
        )

        hamiltonian.append([obs[i], drive_z.field])

    # evolution and

    output = qutip.sesolve(hamiltonian, psi0, time, e_ops=obs + current_obs)

    current_exp = np.zeros((steps, l))
    z_exp = np.zeros_like(current_exp)
    for r in range(l):
        z_exp[:, r] = output.expect[r]
        current_exp[:, r] = output.expect[l + r]

    # get the full field
    heff_1stversion = torch.zeros_like(torch.tensor(z_exp))

    z_vector = torch.tensor(z_exp[0]).unsqueeze(0)
    
    for i in trange(time_stop):

        if i == 0:

            heff_1stversion[i] = get_effective_field(z=(z_vector), model=model, i=-1)
            z_vector = torch.cat(
                (z_vector, torch.tensor(z_exp[i + 1]).unsqueeze(0)), dim=0
            )
        if i != 0:
            heff_1stversion[i] = get_effective_field(z=z_vector, model=model, i=-1)

            if i != time_stop - 1:
                z_vector = torch.cat(
                    (z_vector, torch.tensor(z_exp[i + 1]).unsqueeze(0)), dim=0
                )

    input = torch.einsum("ti->it", torch.tensor(z_exp[:time_stop]))
    heff = model(input.unsqueeze(0)).detach().squeeze()
    heff = torch.einsum("it->ti", heff)

    current_derivative = np.gradient(current_exp, time, axis=0)
    x_sp = np.sqrt(1 - z_exp**2) * np.cos(
        np.arcsin(-1 * (current_exp) / (2 * np.sqrt(1 - z_exp**2)))
    )
    h_eff_exact = (0.25 * current_derivative + z_exp) / (x_sp + 10**-4) - h
    zi = torch.tensor(z_exp[0, :])  # initial magnetization
    z_evolution = zi.clone().unsqueeze(0)  # initialize the evolution
    #  Kohm Sham step 1) Initialize the state from an initial magnetization

    # psi = initialize_psi_from_xyz(z=-1 * zi[0], x=zi[1], y=torch.zeros_like(zi[1]))
    # density matrix initialization
    psi = initialize_psi_from_z(z=-1 * zi)
    print("psi.shape", psi.shape)

    h_eff = torch.zeros((time_stop, l))
    t_bar = tqdm(enumerate(time))
    for i in trange(time_stop - 1):
        t = time[i]
        print("H SHAPE=", h.shape)
        psi, df, z_evolution = (
            nonlinear_schrodinger_step_zzx_model_full_effective_field(
                psi=psi,
                model=model,
                i=i,
                h=torch.tensor(h[:time_stop]) ,
                full_z=z_evolution,  # full z in size x time
                self_consistent_step=self_consistent_step,
                dt=dt,
                exponent_algorithm=exponent_algorithm,
                #    dataset_z=torch.tensor(dataset_z),
            )
        )
        h_eff[i] = df

        z_qutip_tot[q, i, :] = z_exp[i]
        z_tot[q, i, :] = z_evolution.detach().numpy()
        h_eff_tot[q, i, :] = h_eff
        h_tot[q, i, :] = h[i]
        h_eff_tot_exact[q, i, :] = h_eff_exact[i, :]

        np.savez(
            f"data/tddft_results/test",
            z_qutip=z_qutip_tot[:, :i],
            z=z_tot[:, :i],
            potential=h_tot[:, :i],
            h_eff=h_eff_tot[:, :i],
            h_eff_exact=h_eff_tot_exact[:, :i],
            time=time[:i],)
    # for j in range(l):
    #     plt.figure(figsize=(10, 10))
    #     plt.title(f"site={j}", fontsize=40)
    #     plt.plot(time[:time_stop], z_exp[:time_stop, j], label="exact")
    #     plt.plot(
    #         time[:time_stop],
    #         z_evolution[:time_stop, j],
    #         color="red",
    #         linestyle="--",
    #         linewidth=5,
    #         label="from reconstruction",
    #     )
    #     plt.legend(fontsize=40)
    #     plt.xlabel(r"$t[1/J]$", fontsize=40)
    #     plt.ylabel(r"$z_i(t)$", fontsize=40)
    #     plt.tick_params(
    #         which="both",
    #         left=True,
    #         right=True,
    #         labelleft=True,
    #         labelright=True,
    #         direction="inout",
    #         length=5,
    #         colors="black",
    #         labelsize=40,
    #     )
    #     plt.show()
    # for j in range(l):
    #     plt.figure(figsize=(10, 10))
    #     plt.title(f"site={j}", fontsize=40)
    #     plt.plot(time[:time_stop], heff.numpy()[:time_stop, j], label="reconstructed")
    #     plt.plot(
    #         time[:time_stop],
    #         h_eff_exact[:time_stop, j],
    #         label="exact",
    #         linewidth=5,
    #         linestyle="--",
    #     )
    #     plt.plot(time[:time_stop], h_eff[:time_stop, j], label="TDDFT", linewidth=5)
    #     plt.legend(fontsize=40)
    #     plt.xlabel(r"$t[1/J]$", fontsize=40)
    #     plt.ylabel(r"$z_i(t)$", fontsize=40)
    #     plt.tick_params(
    #         which="both",
    #         left=True,
    #         right=True,
    #         labelleft=True,
    #         labelright=True,
    #         direction="inout",
    #         length=5,
    #         width=3,
    #         colors="black",
    #         labelsize=40,
    #     )

    #     plt.show()

# %%
smooth_heff = (heff + torch.roll(heff, shifts=1, dims=0)) / 2

plt.plot(heff.numpy()[:, 0])
plt.plot(h_eff_exact[:, 0])
plt.plot(smooth_heff[:, 0])

plt.show()
# %% USING QUTIP for the first site
ham0 = SpinOperator(index=[("x", 0)], coupling=[1], size=1)

hamExtZ = SpinOperator(index=[("z", 0)], coupling=[h_eff_exact[0, 0] + h[0, 0]], size=1)


psi0 = np.zeros(2)
psi0[0] = np.sqrt((1 + z_exp[0, 0]) / 2)
psi0[1] = np.sqrt((1 - z_exp[0, 0]) / 2)
psi0 = qutip.Qobj(psi0[:], shape=psi0.shape, dims=([[2 for i in range(1)], [1]]))

print("real ground state energy=", eng[0], eng)
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
obs_x: List[qutip.Qobj] = []
# for i in range(l):
#     z_op = SpinOperator(index=[("z", i)], coupling=[1.0], size=l, verbose=1)
#     # print(f"x[{i}]=", x.qutip_op, "\n")
#     x_op = SpinOperator(index=[("x", i)], coupling=[1.0], size=l, verbose=0)
#     obs.append(z_op.qutip_op)
#     obs_x.append(x_op.qutip_op)

obs = [
    SpinOperator(index=[("z", i) for i in range(1)], coupling=[1] * 1, size=1).qutip_op
]

print(obs[0])


print("\n INITIALIZE THE HAMILTONIAN \n")
# build up the time dependent object for the qutip evolution
hamiltonian = [ham0.qutip_op]

for i in range(1):
    drive_z = Driving(
        h=h_eff_exact[:, :] + h,
        idx=i,
        dt=time[1] - time[0],
    )

    hamiltonian.append([obs[i], drive_z.field])


# evolution

output = qutip.sesolve(hamiltonian, psi0, time)

psi_t = output.states
psi_t = np.asarray(psi_t)
print(psi_t.shape)
z_eff = np.einsum(
    "ta,ab,tb->t",
    np.conj(psi_t)[:, :, 0],
    SpinOperator(index=[("z", i) for i in range(1)], coupling=[1], size=1).qutip_op,
    psi_t[:, :, 0],
)


# %%

plt.plot(z_eff)
plt.plot(z_exp[:, 0])
plt.show()

# %%
plt.plot(h_eff_exact[:, 0])
plt.plot(heff.numpy()[:, 0])
plt.show()

# %%
from scipy.fft import fft

h_eff_exact_signal = fft(h_eff_exact[:, 0])
heff_signal = fft(heff.numpy()[:, 0])
plt.plot(h_eff_exact_signal)
plt.plot(heff_signal)
plt.xlim([40, 100])
plt.ylim([-50, 50])
plt.show()
# %%
