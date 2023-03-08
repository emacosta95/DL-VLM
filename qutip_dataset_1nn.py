from src.qutip_lab.qutip_class import SpinOperator, SpinHamiltonian, SteadyStateSolver
import numpy as np
import qutip
from tqdm import tqdm, trange
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
from qutip.solver import Options
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--file_name",
    type=str,
    help="name of the path where the simulation is saved",
    default="data/gaussian_driving/simulation",
)


parser.add_argument(
    "--seed",
    type=int,
    help="seed for pytorch and numpy (default=42)",
    default=42,
)


parser.add_argument(
    "--c",
    type=float,
    help="maximum amplitude of the random gaussian driving",
    default=4.0,
)


parser.add_argument(
    "--sigma",
    type=float,
    help="maximum value of the sigma in the gaussian noise (default=9)",
    default=9.0,
)


parser.add_argument(
    "--ji",
    type=float,
    help="initial j value (default=1.)",
    default=1.0,
)

parser.add_argument(
    "--jf",
    type=float,
    help="final j value (default=1.)",
    default=1.0,
)

parser.add_argument(
    "--ti",
    type=float,
    help="initial time (default=0.)",
    default=0.0,
)


parser.add_argument(
    "--tf",
    type=float,
    help="final time (default=10.)",
    default=10.0,
)

parser.add_argument(
    "--dt",
    type=float,
    help="time step (default=0.1)",
    default=0.1,
)


parser.add_argument(
    "--size",
    type=int,
    help="size of the system (default=5)",
    default=5,
)


parser.add_argument(
    "--n_dataset",
    type=int,
    help="number of samples of the dataset",
    default=15000,
)


parser.add_argument(
    "--different_gaussians",
    type=int,
    help="number of samples of the dataset",
    default=100,
)

parser.add_argument(
    "--checkpoint",
    type=int,
    help="number of time step before a checkpoint (default=100)",
    default=100,
)

args = parser.parse_args()


class Driving:
    def __init__(
        self,
        hs: np.ndarray,
        dt: float,
        i: int,
    ) -> None:
        self.hs = hs
        self.dt = dt
        self.i = i
        self.h = None

    def get_instance(self, sample: int):
        self.h = hs[sample]

    def field(self, t: float, args) -> Union[np.ndarray, float]:
        return self.h[int(t / self.dt) - 1, self.i]


# size of the system
size: int = args.size

# periodic boundary conditions
pbc: bool = True

# coupling term
j: float = args.ji

# time interval
t_resolution: int = int(args.tf / args.dt)
t: np.ndarray = np.linspace(0, args.tf, t_resolution)


# define the driving once for all
c = np.random.uniform(0, args.c, size=args.different_gaussians)
sigma = np.random.uniform(1, args.sigma, size=args.different_gaussians)

n_dataset = args.n_dataset


file_name = (
    args.file_name
    + f"_size_{size}_tf_{args.tf}_dt_{args.dt}_sigma_1_{args.sigma}_c_0_{args.c}_noise_{args.different_gaussians}_n_dataset_{n_dataset}"
)

# return
z = np.zeros((n_dataset, t_resolution, size))


corr = c[:, None, None] * np.exp(
    -(0.5 * ((t[None, :, None] - t[None, None, :]) / sigma[:, None, None]) ** 2)
)
lambd, q = np.linalg.eigh(corr)
x = np.random.normal(
    size=(int(n_dataset / args.different_gaussians), t_resolution, size)
)
x = np.einsum("st,ati->sati", np.sqrt(np.abs(lambd)), x)
hs = np.einsum("sty,sayi->sati", q, x)
hs = hs.reshape(-1, t_resolution, size)

driving_fields = []
for i in range(size):
    gaussian_driving = Driving(hs=hs, dt=args.dt, i=i)
    driving_fields.append(gaussian_driving)


# define the time independent hamiltonian
ham0 = SpinHamiltonian(
    direction_couplings=[("z", "z")], pbc=True, coupling_values=[1.0], size=size
)
print(ham0)

# define the initial exp value
_, psi0 = np.linalg.eigh(ham0.qutip_op)
psi0 = psi0[:, 0]
psi0 = qutip.Qobj(psi0, shape=psi0.shape, dims=([[2 for i in range(size)], [1]]))

obs: List[qutip.Qobj] = []
for i in range(size):
    x = SpinOperator(index=[("x", i)], coupling=[1.0], size=size, verbose=1)
    obs.append(x.qutip_op)
    z[:, 0, i] = x.expect_value(psi0)


for sample in trange(n_dataset):
    # define the instance
    ham = [ham0.qutip_op]

    # define the time dependent part
    for i in range(size):
        driving_fields[i].get_instance(sample)
        ham.append([obs[i], driving_fields[i].field])

    output = qutip.mesolve(ham, psi0, t[1:], e_ops=obs)
    z[sample, 1:, :] = np.asarray(output.expect).reshape(-1, size)

    if sample % args.checkpoint == 0:
        np.savez(file_name, density=z, potential=hs, time=t)


for r in np.isnan(psi0):
    if r == True:
        print(r)

x_gs = []


plt.plot(x_gs)


# options = Options(num_cpus=4, atol=1e-20)
