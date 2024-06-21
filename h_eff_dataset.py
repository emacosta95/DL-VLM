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

batch_size = 20000
l = 8
# rates = [0.1, 0.5, 0.8, 1.0]

rate = 1.0
# j coupling
j = 1
# omega auxiliary field
omega = 1


steps = 200
tf = 20.0
time = np.linspace(0.0, tf, steps)

# z_qutip_tot = np.zeros((nbatch * nbatch * batch_size, steps, l))
z_qutip_tot = []
h_eff_tot = []
h_tot = []
current_qutip_tot = []
current_derivative_tot = []
x_sp_tot = []

ham0 = SpinHamiltonian(
    direction_couplings=[("x", "x")],
    pbc=True,
    coupling_values=[j],
    size=l,
)

hamExtX = SpinOperator(index=[("x", i) for i in range(l)], coupling=[omega] * l, size=l)

obs: List[qutip.Qobj] = []
current_obs: List[qutip.Qobj] = []

for i in range(l):
    z_op = SpinOperator(index=[("z", i)], coupling=[1.0], size=l, verbose=1)
    # print(f"x[{i}]=", x.qutip_op, "\n")
    current = SpinOperator(
        index=[("x", (i - 1) % l, "y", i), ("y", i, "x", (i + 1) % l)],
        coupling=[2 * j, 2 * j],
        size=l,
    )

    obs.append(z_op.qutip_op)
    current_obs.append(current.qutip_op)

hi = np.ones((time.shape[0], l))  # we fix the initial field to be 1J

for idx in trange(0, batch_size):

    # rate = np.random.uniform(0.3, 1.0)

    omega = np.linspace(0, np.pi, time.shape[0] // 4)
    omega_cutoff = 16
    delta = np.random.uniform(size=(omega_cutoff, l))

    h = (
        delta[:, None, :]
        * np.sin(time[None, :, None] * omega[:omega_cutoff, None, None])
        + hi
    )

    h = np.average(h, axis=0)

    hamExtZ = SpinOperator(index=[("z", i) for i in range(l)], coupling=h[0], size=l)

    eng, psi0 = np.linalg.eigh(ham0.qutip_op + hamExtZ.qutip_op + hamExtX.qutip_op)
    psi0 = qutip.Qobj(psi0[:, 0], shape=psi0.shape, dims=([[2 for i in range(l)], [1]]))

    # print("real ground state energy=", eng[0])

    hamiltonian = [ham0.qutip_op + hamExtX.qutip_op]

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

    # Current derivative
    current_derivative = np.gradient(current_exp, time, axis=0)

    # compute the effective field
    x_sp = np.sqrt(1 - z_exp**2) * np.cos(
        np.arcsin(-1 * (current_exp) / (2 * np.sqrt(1 - z_exp**2)))
    )

    current_derivative = np.gradient(current_exp, time, axis=0)
    h_eff = (0.25 * current_derivative + z_exp) / (x_sp + 10**-4)

    # update the database
    h_eff_tot.append(h_eff)
    h_tot.append(h)
    z_qutip_tot.append(z_exp)
    current_qutip_tot.append(current_exp)
    x_sp_tot.append(x_sp)
    current_derivative_tot.append(current_derivative)

    if idx % 1000 == 0:
        np.savez(
            f"data/dataset_h_eff/periodic/xxzx_model/dataset_random_rate_random_amplitude_01-08_fixed_initial_state_nbatch_{nbatch}_batchsize_{batch_size}_steps_{steps}_tf_{tf}_l_{l}_240620",
            current=np.asarray(current_qutip_tot),
            z=np.asarray(z_qutip_tot),
            h_eff=np.asarray(h_eff_tot),
            current_derivative=np.asarray(current_derivative_tot),
            h=np.asarray(h_tot),
            x_sp=np.asarray(x_sp_tot),
            time=time,
        )

np.savez(
    f"data/dataset_h_eff/periodic/xxzx_model/dataset_random_rate_random_amplitude_01-08_fixed_initial_state_nbatch_{nbatch}_batchsize_{batch_size}_steps_{steps}_tf_{tf}_l_{l}_240620",
    current=current_qutip_tot,
    z=z_qutip_tot,
    h_eff=h_eff_tot,
    current_derivative=current_derivative_tot,
    h=h_tot,
    x_sp=x_sp_tot,
    time=time,
)
