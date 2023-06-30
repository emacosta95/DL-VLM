# %% Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from src.training.models_adiabatic import Energy_XXZX
from src.qutip_lab.qutip_class import SpinOperator, SpinHamiltonian, SteadyStateSolver

from src.tddft_methods.kohm_sham_utils import (
    compute_the_gradient,
    build_hamiltonian,
    initialize_psi_from_z_and_x,
    compute_the_magnetization,
    time_step_crank_nicolson_algorithm,
)
import qutip
from typing import List


# %% Qutip details
class Driving:
    def __init__(
        self, h_i: np.array, h_f: np.array, rate: float, idx: int, direction: int
    ) -> None:
        self.hi = h_i
        self.hf = h_f
        self.rate = rate
        self.idx: int = idx
        self.direction = direction

    def field(self, t: float, args):
        return (
            self.hi[self.direction, self.idx] * np.exp(-t * self.rate)
            + (1 - np.exp(-t * self.rate)) * self.hf[self.direction, self.idx]
        )

    def get_the_field(self, t: np.ndarray):
        return (
            self.hi[None, self.direction, :] * np.exp(-t[:, None] * self.rate)
            + (1 - np.exp(-t[:, None] * self.rate)) * self.hf[None, self.direction, :]
        )


# %% Data
data = np.load(
    "data/kohm_sham_approach/uniform/reduction_2_input_channel_dataset_h_0.0_2.0_omega_1_j_1_1nn_n_100.npz"
)

h = data["potential"]
z = data["density"]
l = h.shape[-1]

model = torch.load(
    "model_rep/kohm_sham/disorder/model_zzxz_reduction_2_input_channel_dataset_h_0.0-2.0_omega_0.0-2.0_j_1_1nn_n_500k_unet_l_train_8_[40, 40, 40, 40, 40, 40]_hc_5_ks_1_ps_6_nconv_0_nblock",
    map_location="cpu",
)
model.eval()
model = model.to(dtype=torch.double)
energy = Energy_XXZX(model=model)
# Implement the Kohm Sham LOOP
z_target = torch.from_numpy(z).double()
h = torch.from_numpy(h).double()


# initialization
idx = 0
self_consistent_step = 10
steps = 1000
time = torch.linspace(0.0, 10.0, steps)
dt = time[1] - time[0]

ndata = 30
rates = np.linspace(0.0, 3.0, ndata)

h_tot = np.zeros((ndata, steps, 2, l))
z_qutip_tot = np.zeros((ndata, steps, l))
z_tot = np.zeros((ndata, steps, l))
x_qutip_tot = np.zeros((ndata, steps, l))
x_tot = np.zeros((ndata, steps, l))
eng_tot = np.zeros((ndata, steps))

hi, idx = torch.max(h, dim=0)
idx = idx[0, 0]
hi = h[idx.item()]
zi = z_target[idx.item()]
hf, _ = torch.min(h, dim=0)
for q, rate in enumerate(rates):
    # Qutip Dynamics
    # Hamiltonian
    ham0 = SpinHamiltonian(
        direction_couplings=[("z", "z")],
        pbc=True,
        coupling_values=[1.0],
        size=l,
    )

    hamExtX = SpinOperator(
        index=[("x", i) for i in range(l)], coupling=hi[1].detach().numpy(), size=l
    )
    hamExtZ = SpinOperator(
        index=[("z", i) for i in range(l)], coupling=hi[0].detach().numpy(), size=l
    )

    eng, psi0 = np.linalg.eigh(ham0.qutip_op + hamExtZ.qutip_op + hamExtX.qutip_op)
    psi0 = qutip.Qobj(psi0[:, 0], shape=psi0.shape, dims=([[2 for i in range(l)], [1]]))

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
    for i in range(l):
        z_op = SpinOperator(index=[("z", i)], coupling=[1.0], size=l, verbose=1)
        # print(f"x[{i}]=", x.qutip_op, "\n")
        x_op = SpinOperator(index=[("x", i)], coupling=[1.0], size=l, verbose=0)
        print(z_op.expect_value(psi=psi0) - zi[0, i].detach().numpy())
        print(x_op.expect_value(psi=psi0) - zi[1, i].detach().numpy())
        obs.append(z_op.qutip_op)
        obs_x.append(x_op.qutip_op)

    print("\n INITIALIZE THE HAMILTONIAN \n")
    # build up the time dependent object for the qutip evolution
    hamiltonian = [ham0.qutip_op]

    for i in range(l):
        drive_z = Driving(
            h_i=hi.detach().numpy(),
            h_f=hf.detach().numpy(),
            rate=rate,
            idx=i,
            direction=0,
        )
        hamiltonian.append([obs[i], drive_z.field])

    h_z = drive_z.get_the_field(time.detach().numpy()).reshape(time.shape[0], 1, -1)
    for i in range(l):
        drive_x = Driving(
            h_i=hi.detach().numpy(),
            h_f=hf.detach().numpy(),
            rate=rate,
            idx=i,
            direction=1,
        )
        hamiltonian.append([obs_x[i], drive_x.field])
    h_x = drive_x.get_the_field(time.detach().numpy()).reshape(time.shape[0], 1, -1)

    h = np.append(h_z, h_x, axis=1)
    h_tot[q] = h
    h = torch.from_numpy(h)
    print(h.shape)

    # evolution

    output = qutip.sesolve(hamiltonian, psi0, time.detach().numpy(), e_ops=obs + obs_x)

    # %% visualization
    for r in range(l):
        z_qutip_tot[q, :, r] = output.expect[r]
    for r in range(l):
        x_qutip_tot[q, :, r] = output.expect[l + r]

    # plt.title("z")
    # plt.plot(time.detach().numpy(), z_qutip, color="red")
    # plt.plot(time.detach().numpy(), h[:, 0], color="black")
    # plt.show()

    # plt.title("x")
    # plt.plot(time.detach().numpy(), x_qutip, color="red")
    # plt.plot(time.detach().numpy(), h[:, 1], color="black")
    # plt.show()

    #  Kohm Sham step 1) Initialize the state from an initial magnetization
    psi = initialize_psi_from_z_and_x(z=-1 * zi[0], x=zi[1])

    t_bar = tqdm(enumerate(time))
    for i in trange(time.shape[0]):
        t = time[i]
        #  Kohm Sham step 2) Build up the fields
        if i == len(time) - 1:
            z, x = compute_the_magnetization(psi=psi)
            psi, eng = time_step_crank_nicolson_algorithm(
                psi=psi,
                h=h[i],
                h_plus=h[i],
                energy=energy,
                dt=dt,
                self_consistent_steps=self_consistent_step,
            )
        else:
            z, x = compute_the_magnetization(psi=psi)
            psi, eng = time_step_crank_nicolson_algorithm(
                psi=psi,
                h=h[i],
                h_plus=h[i + 1],
                energy=energy,
                dt=dt,
                self_consistent_steps=self_consistent_step,
            )

        eng_tot[q] = eng

        z_tot[q, i, :] = z[0].detach().numpy()
        x_tot[q, i, :] = x[0].detach().numpy()

        t_bar.refresh()

    np.savez(
        f"data/kohm_sham_approach/results/tddft_adiabatic_approximation_uniform_0.0_2.0_steps_{steps}_self_consistent_steps_{self_consistent_step}_ndata_{ndata}_rate_{3.0}",
        x_qutip=x_qutip_tot,
        z_qutip=z_qutip_tot,
        z=z_tot,
        x=x_tot,
        potential=h_tot,
        energy=eng_tot,
    )

# %% Visualize results

# for i in range(1):
#     plt.title("x")
#     plt.plot(time.detach().numpy(), x_dl[:, i])
#     plt.plot(time.detach().numpy(), x_qutip[:, i])
#     plt.show()

# for i in range(1):
#     plt.title("x")
#     plt.plot(
#         time.detach().numpy(), np.abs((x_dl[:, i] - x_qutip[:, i]) / x_qutip[:, i])
#     )
#     plt.show()


# for i in range(1):
#     plt.title("z")
#     plt.plot(time.detach().numpy(), z_dl[:, i])
#     plt.plot(time.detach().numpy(), z_qutip[:, i])
#     plt.show()

# for i in range(1):
#     plt.title("z")
#     plt.plot(
#         time.detach().numpy(), np.abs((z_dl[:, i] - z_qutip[:, i]) / z_qutip[:, i])
#     )
#     plt.show()


# plt.plot(time.detach().numpy(), engs)
# %%
