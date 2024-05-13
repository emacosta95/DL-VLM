import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from src.qutip_lab.qutip_class import SpinOperator, SpinHamiltonian, SteadyStateSolver
from scipy.fft import fft, ifft
import qutip
from qutip.metrics import fidelity
from typing import List
from qutip import propagator
import os


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


# hyperparameters

nbatch = 1

batch_size = 200000
l = 8
# rates = [0.1, 0.5, 0.8, 1.0]

rate = 1.0
# j coupling
j = -1
# omega auxiliary field
omega = 1
range_h = np.linspace(0.6, 1.0, nbatch)

# max_h = 1.0 + np.linspace(0.1, 1., nbatch)
# min_h = np.linspace(0.1, 1., nbatch)

steps = 400
tf = 40.0
time = np.linspace(0.0, tf, steps)

# z_qutip_tot = np.zeros((nbatch * nbatch * batch_size, steps, l))
z_qutip_tot = np.zeros((nbatch * batch_size, steps, l))
z_big_tot = np.zeros_like(z_qutip_tot)
h_eff_tot = np.zeros_like(z_qutip_tot)
h_eff_observables_tot = np.zeros_like(z_qutip_tot)
h_tot = np.zeros_like(z_qutip_tot)
current_qutip_tot = np.zeros_like(z_qutip_tot)
current_derivative_tot = np.zeros_like(z_qutip_tot)
q_tot = np.zeros_like(z_qutip_tot)
r_diag_tot = np.zeros_like(z_qutip_tot)
r_diag_minus_tot = np.zeros_like(z_qutip_tot)
r_diag_plus_tot = np.zeros_like(z_qutip_tot)
x_sp_tot = np.zeros_like(z_qutip_tot)

ham0 = SpinHamiltonian(
    direction_couplings=[("x", "x")],
    pbc=True,
    coupling_values=[j],
    size=l,
)

obs: List[qutip.Qobj] = []
current_obs: List[qutip.Qobj] = []
q_obs: List[qutip.Qobj] = []
r_diag_obs: List[qutip.Qobj] = []
r_diag_plus_obs: List[qutip.Qobj] = []
r_diag_minus_obs: List[qutip.Qobj] = []
for i in range(l):
    z_op = SpinOperator(index=[("z", i)], coupling=[1.0], size=l, verbose=1)
    # print(f"x[{i}]=", x.qutip_op, "\n")
    current = SpinOperator(
        index=[("x", (i - 1) % l, "y", i), ("y", i, "x", (i + 1) % l)],
        coupling=[2 * j, 2 * j],
        size=l,
    )
    q = SpinOperator(
        index=[("x", (i - 1) % l, "z", i, "x", (i + 1) % l)],
        coupling=[2 * j**2],
        size=l,
    )
    r_diag = SpinOperator(
        index=[("x", (i - 1) % l, "x", i), ("x", i, "x", (i + 1) % l)],
        coupling=[j, j],
        size=l,
    )
    r_diag_plus = SpinOperator(
        index=[("y", i, "y", (i + 1) % l)],
        coupling=[-j, -j],
        size=l,
    )
    r_diag_minus = SpinOperator(
        index=[("y", (i - 1) % l, "y", i)],
        coupling=[-j, -j],
        size=l,
    )

    obs.append(z_op.qutip_op)
    current_obs.append(current.qutip_op)
    q_obs.append(q.qutip_op)
    r_diag_obs.append(r_diag.qutip_op)
    r_diag_minus_obs.append(r_diag_minus.qutip_op)
    r_diag_plus_obs.append(r_diag_plus.qutip_op)

for idx_batch in trange(0, nbatch):
    # for jdx_batch in trange(0, nbatch):
    # hi = np.random.uniform(min_h[idx_batch], max_h[idx_batch], size=(batch_size, l))
    # hf = np.random.uniform(min_h[jdx_batch], max_h[jdx_batch], size=(batch_size, l))

    # random configurations
    # hi = np.random.uniform(
    #    range_h[idx_batch], range_h[idx_batch] + 1.0, size=(batch_size, l)
    # )
    hi = np.ones(l)  # we fix the initial field to be 1J
    delta = np.random.uniform(0.1, 0.8, size=(batch_size, l))

    for idx in trange(0, batch_size):

        rate = np.random.uniform(0.3, 1.0)

        # random_int = np.random.randint(len(rates))
        # rate = rates[random_int]
        h = delta[idx, None, :] * np.sin(time * rate)[:, None] + hi[None, :]
        # h = (
        #    np.exp(-1 * rate * time)[:, None] * hi[idx, None, :]
        #    + (1 - np.exp(-1 * rate * time)[:, None]) * hf[idx, None, :]
        # )

        hamExtZ = SpinOperator(
            index=[("z", i) for i in range(l)], coupling=h[0], size=l
        )

        eng, psi0 = np.linalg.eigh(ham0.qutip_op + hamExtZ.qutip_op)
        psi0 = qutip.Qobj(
            psi0[:, 0], shape=psi0.shape, dims=([[2 for i in range(l)], [1]])
        )

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

        output = qutip.sesolve(
            hamiltonian,
            psi0,
            time,
            e_ops=obs
            + current_obs
            + q_obs
            + r_diag_obs
            + r_diag_minus_obs
            + r_diag_plus_obs,
        )

        current_exp = np.zeros((steps, l))
        z_exp = np.zeros_like(current_exp)
        r_diag_exp = np.zeros_like(current_exp)
        r_diag_minus_exp = np.zeros_like(current_exp)
        r_diag_plus_exp = np.zeros_like(current_exp)
        for r in range(l):
            z_exp[:, r] = output.expect[r]
            current_exp[:, r] = output.expect[l + r]
            r_diag_exp[:, r] = output.expect[3 * l + r]
            r_diag_minus_exp[:, r] = output.expect[4 * l + r]
            r_diag_plus_exp[:, r] = output.expect[5 * l + r]

        # Current derivative
        current_derivative = np.gradient(current_exp, time, axis=0)

        # R matrix
        r_matrix_1 = np.einsum("ij,tj->tij", np.eye(r_diag_exp.shape[-1]), r_diag_exp)
        # +1 off diagonal term
        r_matrix_2 = np.einsum(
            "ij,tj->tij", np.eye(r_diag_exp.shape[-1]), r_diag_plus_exp
        )
        r_matrix_2 = np.roll(r_matrix_2, axis=-1, shift=1)
        # -1 off diagonal term
        r_matrix_3 = np.einsum(
            "ij,tj->tij", np.eye(r_diag_exp.shape[-1]), r_diag_minus_exp
        )
        r_matrix_3 = np.roll(r_matrix_3, axis=-1, shift=-1)

        # sum of all this term
        r_matrix = r_matrix_3 + r_matrix_2 + r_matrix_1

        # compute the effective field
        x_sp = np.sqrt(1 - z_exp**2) * np.cos(
            np.arcsin(-1 * (current_exp) / (2 * np.sqrt(1 - z_exp**2)))
        )

        current_derivative = np.gradient(current_exp, time, axis=0)
        h_eff = (0.25 * current_derivative + z_exp) / (x_sp + 10**-4)

        # build the operators that defines the derivative of the current
        r_h = np.einsum(
            "tji,ti->tj", r_matrix, h
        )  # +np.einsum('tji,ti->ti',r_matrix,h))

        q_effective = (
            r_h - current_derivative / 4
        )  # factor 4 due to the commutation relations

        # definition of the heff from observables
        h_eff_from_observables = (z_exp + (r_h - q_effective)) / (x_sp + 10**-4)

        # update the database
        h_eff_tot[(batch_size * (idx_batch) + idx)] = h_eff
        h_eff_observables_tot[(batch_size * (idx_batch) + idx)] = h_eff_from_observables
        h_tot[(batch_size * (idx_batch) + idx)] = h
        z_qutip_tot[(batch_size * (idx_batch) + idx)] = z_exp
        current_qutip_tot[(batch_size * (idx_batch) + idx)] = current_exp
        x_sp_tot[(batch_size * (idx_batch) + idx)] = x_sp
        q_tot[(batch_size * (idx_batch) + idx)] = q_effective
        r_diag_tot[(batch_size * (idx_batch) + idx)] = r_diag_exp
        r_diag_plus_tot[(batch_size * (idx_batch) + idx)] = r_diag_plus_exp
        r_diag_minus_tot[(batch_size * (idx_batch) + idx)] = r_diag_minus_exp
        current_derivative_tot[(batch_size * (idx_batch) + idx)] = current_derivative

    np.savez(
        f"data/dataset_h_eff/periodic/dataset_periodic_random_rate_03-1_random_amplitude_01-08_fixed_initial_state_nbatch_{nbatch}_batchsize_{batch_size}_steps_{steps}_tf_{tf}_l_{l}_240513",
        current=current_qutip_tot,
        z=z_qutip_tot,
        h_eff=h_eff_tot,
        current_derivative=current_derivative_tot,
        h=h_tot,
        x_sp=x_sp_tot,
        q=q_tot,
        r_diag=r_diag_tot,
        r_diag_minus=r_diag_minus_tot,
        r_diag_plus=r_diag_plus_tot,
        h_eff_from_observables=h_eff_observables_tot,
        time=time,
    )

np.savez(
    f"data/dataset_h_eff/periodic/dataset_periodic_random_rate_03-1_random_amplitude_01-08_fixed_initial_state_nbatch_{nbatch}_batchsize_{batch_size}_steps_{steps}_tf_{tf}_l_{l}_240513",
    current=current_qutip_tot,
    z=z_qutip_tot,
    h_eff=h_eff_tot,
    current_derivative=current_derivative_tot,
    h=h_tot,
    x_sp=x_sp_tot,
    q=q_tot,
    r_diag=r_diag_tot,
    r_diag_minus=r_diag_minus_tot,
    r_diag_plus=r_diag_plus_tot,
    h_eff_from_observables=h_eff_observables_tot,
    time=time,
)
