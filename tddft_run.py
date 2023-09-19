# %% Imports


import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from src.training.models_adiabatic import Energy_XXZX, Energy_reduction_XXZX
from src.qutip_lab.qutip_class import SpinOperator, SpinHamiltonian, SteadyStateSolver

from src.tddft_methods.kohm_sham_utils import (
    compute_the_gradient,
    build_hamiltonian,
    initialize_psi_from_z_and_x,
    initialize_psi_from_xyz,
    compute_the_magnetization,
    crank_nicolson_algorithm,
    exponentiation_algorithm,
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


class PeriodicDriving:
    def __init__(
        self, h_i: np.array, delta: np.array, rate: float, idx: int, direction: int
    ) -> None:
        self.hi = h_i
        self.delta = delta
        self.rate = rate
        self.idx: int = idx
        self.direction = direction

    def field(self, t: float, args):
        return self.hi[self.direction, self.idx] + (
            self.delta[self.direction, self.idx]
        ) * np.sin(self.rate * t)

    def get_the_field(self, t: np.ndarray):
        return (
            self.hi[None, self.direction, :]
            + (self.delta[None, self.direction, :]) * np.sin(self.rate * t)[:, None]
        )


# %% Data
data = np.load(
    "data/kohm_sham_approach/disorder/2_input_channel_dataset_h_mixed_0.0_5.0_h_0.0-2.0_j_1_1nn_n_6000.npz"
)


z = data["density"]

print(z.shape)

l = z.shape[-1]

model = torch.load(
    "model_rep/kohm_sham/disorder/model_xxzzxz_2_input_channel_dataset_h_mixed_0.0_5.0_h_0.0-2.0_j_1_1nn_n_500k_unet_l_train_8_[60, 60, 60, 60, 60, 60]_hc_5_ks_1_ps_6_nconv_0_nblock",
    map_location="cpu",
)
model.eval()
model = model.to(dtype=torch.double)
energy = Energy_XXZX(model=model)
energy.eval()
# Implement the Kohm Sham LOOP
z_target = torch.from_numpy(z).double()

# initialization
exponent_algorithm = True
self_consistent_step = 1
steps = 1000
tf = 10.0
time = torch.linspace(0.0, tf, steps)
dt = time[1] - time[0]

ndata = 10
rates = np.linspace(0.0, 0.2, ndata)

rates = np.array([0.0, 0.01, 0.05, 0.1, 0.5, 1, 1.5])
ndata = rates.shape[0]


h_tot = np.zeros((ndata, steps, 2, l))
z_qutip_tot = np.zeros((ndata, steps, l))
z_tot = np.zeros((ndata, steps, l))
x_qutip_tot = np.zeros((ndata, steps, l))
y_qutip_tot = np.zeros((ndata, steps, l))


x_tot = np.zeros((ndata, steps, l))
y_tot = np.zeros((ndata, steps, l))
eng_tot_z = np.zeros((ndata, steps))
eng_tot_x = np.zeros((ndata, steps))
eng_tot = np.zeros((ndata, steps))
eng_qutip_tot = np.zeros((ndata, steps))
gradients_tot = np.zeros((ndata, steps, 2, l))
m_qutip_tot = np.append(
    z_qutip_tot.reshape(ndata, steps, 1, l),
    x_qutip_tot.reshape(ndata, steps, 1, l),
    axis=-2,
)

# is the driving periodic?
periodic = False

# define the initial external field
# zz x quench style (?)
hi = torch.ones((2, l))
hi[1] = 2.0  # high transverse field
hi[0] = 0.5
# define the final external field
hf = torch.ones((2, l))
hf[1] = 1.0
hf[0] = 0.5


# define the delta for the periodic driving
delta = torch.ones((2, l))
delta[1] = 0.9
delta[0] = 0.0


# %% Compute the initial ground state configuration

gd = GradientDescentKohmSham(
    loglr=-3,
    energy=energy,
    epochs=5000,
    seed=23,
    num_threads=3,
    device="cpu",
    n_init=torch.mean(z_target, dim=0),
    h=hi,
)


zi = gd.run()
zi = torch.from_numpy(zi)[0]

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
    obs_x: List[qutip.Qobj] = []
    obs_y: List[qutip.Qobj] = []
    for i in range(l):
        z_op = SpinOperator(index=[("z", i)], coupling=[1.0], size=l, verbose=1)
        x_op = SpinOperator(index=[("x", i)], coupling=[1.0], size=l, verbose=0)
        y_op = SpinOperator(index=[("y", i)], coupling=[1.0], size=l, verbose=0)

        print(z_op.expect_value(psi=psi0) - zi[0, i].detach().numpy())
        print(x_op.expect_value(psi=psi0) - zi[1, i].detach().numpy())

        obs.append(z_op.qutip_op)
        obs_x.append(x_op.qutip_op)
        obs_y.append(y_op.qutip_op)

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
                direction=0,
            )
        else:
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
        if periodic:
            drive_x = PeriodicDriving(
                h_i=hi.detach().numpy(),
                delta=delta.detach().numpy(),
                rate=rate,
                idx=i,
                direction=1,
            )
        else:
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

    output = qutip.sesolve(
        hamiltonian, psi0, time.detach().numpy(), e_ops=obs + obs_x + obs_y
    )

    # %% visualization
    for r in range(l):
        z_qutip_tot[q, :, r] = output.expect[r]
        m_qutip_tot[q, :, 0, r] = output.expect[r]
        x_qutip_tot[q, :, r] = output.expect[l + r]
        m_qutip_tot[q, :, 1, r] = output.expect[l + r]
        y_qutip_tot[q, :, r] = output.expect[2 * l + r]

    #  Kohm Sham step 1) Initialize the state from an initial magnetization
    psi = initialize_psi_from_z_and_x(z=-1 * zi[0], x=zi[1])
    # psi = initialize_psi_from_xyz(z=-1 * zi[0], x=zi[1], y=torch.zeros_like(zi[1]))

    t_bar = tqdm(enumerate(time))
    for i in trange(time.shape[0]):
        t = time[i]
        #  Kohm Sham step 2) Build up the fields
        if i == len(time) - 1:
            z, x, y = compute_the_magnetization(psi=psi)
            # uniform condition (brute force)

            m = torch.cat((z.view(1, -1), x.view(1, -1)), dim=0)
            m = m.unsqueeze(0)  # the batch dimension

            z0, x0, _ = compute_the_magnetization(psi=psi)
            m0 = torch.cat((z0.view(1, -1), x0.view(1, -1)), dim=0)
            m0 = m0.unsqueeze(0)  # the batch dimension

            # m0 = torch.from_numpy(m_qutip_tot[q, i]).unsqueeze(0)

            omega_eff, eng = compute_the_gradient(
                m=m0, h=h[i].unsqueeze(0), energy=energy, respect_to="x"
            )
            h_eff, _ = compute_the_gradient(
                m=m0, h=h[i].unsqueeze(0), energy=energy, respect_to="z"
            )

            hamiltonian0 = build_hamiltonian(
                field_x=-1 * omega_eff[0], field_z=-1 * h_eff[0]
            )
            if exponent_algorithm:
                psi1 = exponentiation_algorithm(
                    hamiltonian=hamiltonian0, psi=psi, dt=dt
                )
            else:
                psi1 = crank_nicolson_algorithm(
                    hamiltonian=hamiltonian0, psi=psi, dt=dt
                )

            for step in range(self_consistent_step):
                z1, x1, _ = compute_the_magnetization(psi=psi1)
                m1 = torch.cat((z1.view(1, -1), x1.view(1, -1)), dim=0)
                m1 = m1.unsqueeze(0)  # the batch dimension

                # m1 = torch.from_numpy(m_qutip_tot[q, i]).unsqueeze(0)

                omega_eff, eng = compute_the_gradient(
                    m=m1, h=h[i].unsqueeze(0), energy=energy, respect_to="x"
                )
                h_eff, _ = compute_the_gradient(
                    m=m1, h=h[i].unsqueeze(0), energy=energy, respect_to="z"
                )

                hamiltonian1 = build_hamiltonian(
                    field_x=-1 * omega_eff[0], field_z=-1 * h_eff[0]
                )

                if exponent_algorithm:
                    psi1 = exponentiation_algorithm(
                        hamiltonian=0.5 * (hamiltonian0 + hamiltonian1), psi=psi, dt=dt
                    )
                else:
                    psi1 = crank_nicolson_algorithm(
                        hamiltonian=0.5 * (hamiltonian0 + hamiltonian1), psi=psi, dt=dt
                    )

            if exponent_algorithm:
                psi = exponentiation_algorithm(
                    hamiltonian=0.5 * (hamiltonian0 + hamiltonian1), psi=psi, dt=dt
                )
            else:
                psi = crank_nicolson_algorithm(
                    hamiltonian=0.5 * (hamiltonian0 + hamiltonian1), psi=psi, dt=dt
                )

        else:
            z, x, y = compute_the_magnetization(psi=psi)
            m = torch.cat((z.view(1, -1), x.view(1, -1)), dim=0)
            m = m.unsqueeze(0)  # the batch dimension

            eng_qutip = energy(
                torch.from_numpy(m_qutip_tot[q, i]).unsqueeze(0), h[i].unsqueeze(0)
            )[0].item()

            eng = energy(m, h[i].unsqueeze(0))[0].item()

            z0, x0, _ = compute_the_magnetization(psi=psi)
            m0 = torch.cat((z0.view(1, -1), x0.view(1, -1)), dim=0)
            m0 = m0.unsqueeze(0)  # the batch dimension

            # m0 = torch.from_numpy(m_qutip_tot[q, i]).unsqueeze(0)

            omega_eff, engx = compute_the_gradient(
                m=m0, h=h[i].unsqueeze(0), energy=energy, respect_to="x"
            )
            h_eff, engz = compute_the_gradient(
                m=m0, h=h[i].unsqueeze(0), energy=energy, respect_to="z"
            )

            hamiltonian0 = build_hamiltonian(
                field_x=-1 * omega_eff[0], field_z=-1 * h_eff[0]
            )
            if exponent_algorithm:
                psi1 = exponentiation_algorithm(
                    hamiltonian=hamiltonian0, psi=psi, dt=dt
                )
            else:
                psi1 = crank_nicolson_algorithm(
                    hamiltonian=hamiltonian0, psi=psi, dt=dt
                )

            for step in range(self_consistent_step):
                z1, x1, _ = compute_the_magnetization(psi=psi1)
                m1 = torch.cat((z1.view(1, -1), x1.view(1, -1)), dim=0)
                m1 = m1.unsqueeze(0)  # the batch dimension

                # m1 = torch.from_numpy(m_qutip_tot[q, i + 1]).unsqueeze(0)

                omega_eff1, eng = compute_the_gradient(
                    m=m1, h=h[i + 1], energy=energy, respect_to="x"
                )
                h_eff1, _ = compute_the_gradient(
                    m=m1, h=h[i + 1], energy=energy, respect_to="z"
                )

                hamiltonian1 = build_hamiltonian(
                    field_x=-1 * omega_eff1[0], field_z=-1 * h_eff1[0]
                )

                if exponent_algorithm:
                    psi1 = exponentiation_algorithm(
                        hamiltonian=0.5 * (hamiltonian0 + hamiltonian1), psi=psi, dt=dt
                    )
                else:
                    psi1 = crank_nicolson_algorithm(
                        hamiltonian=0.5 * (hamiltonian0 + hamiltonian1), psi=psi, dt=dt
                    )

            if exponent_algorithm:
                psi = exponentiation_algorithm(
                    hamiltonian=0.5 * (hamiltonian0 + hamiltonian1), psi=psi, dt=dt
                )
            else:
                psi = crank_nicolson_algorithm(
                    hamiltonian=0.5 * (hamiltonian0 + hamiltonian1), psi=psi, dt=dt
                )
        eng_tot_z[q, i] = engz
        eng_tot_x[q, i] = engx

        eng_tot[q, i] = eng
        eng_qutip_tot[q, i] = eng_qutip

        z_tot[q, i, :] = m[0, 0].detach().numpy()
        x_tot[q, i, :] = m[0, 1].detach().numpy()
        y_tot[q, i, :] = y[0].detach().numpy()
        gradients_tot[q, i, 1, :] = -1 * omega_eff[0].detach().numpy()
        gradients_tot[q, i, 0, :] = -1 * h_eff[0].detach().numpy()

        if periodic:
            np.savez(
                f"data/kohm_sham_approach/results/tddft_periodic_uniform_zzxxzx_model_h_0_5_omega_0_2_ti_0_tf_{tf:.0f}_hi_{hi[0,0].item():.4f}_delta_{delta[0,0].item():.4f}_omegai_{hi[1,0].item():.1f}_delta_{delta[1,0].item():.1f}_steps_{steps}_self_consistent_steps_{self_consistent_step}_ndata_{ndata}_exp_{exponent_algorithm}",
                x_qutip=x_qutip_tot,
                z_qutip=z_qutip_tot,
                y_qutip=y_qutip_tot,
                z=z_tot,
                x=x_tot,
                y=y_tot,
                potential=h_tot,
                energy_x=eng_tot_x,
                energy_z=eng_tot_z,
                energy=eng_tot,
                energy_qutip=eng_qutip_tot,
                gradient=gradients_tot,
                rates=rates,
            )

        else:
            np.savez(
                f"data/kohm_sham_approach/results/dl_functional/tddft_quench_uniform_model_h_0_2_omega_0_2_ti_0_tf_{tf:.0f}_hi_{hi[0,0].item():.4f}_hf_{hf[0,0].item():.4f}_omegai_{hi[1,0].item():.1f}_omegaf_{hf[1,0].item():.1f}_steps_{steps}_self_consistent_steps_{self_consistent_step}_ndata_{ndata}_exp_{exponent_algorithm}",
                x_qutip=x_qutip_tot,
                z_qutip=z_qutip_tot,
                z=z_tot,
                x=x_tot,
                y=y_tot,
                y_qutip=y_qutip_tot,
                potential=h_tot,
                energy_x=eng_tot_x,
                energy_z=eng_tot_z,
                energy=eng_tot,
                energy_qutip=eng_qutip_tot,
                gradient=gradients_tot,
                rates=rates,
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
