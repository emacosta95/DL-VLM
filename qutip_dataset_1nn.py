from src.qutip_lab.qutip_class import SpinOperator, SpinHamiltonian, SteadyStateSolver
import numpy as np
import qutip
from tqdm import tqdm, trange
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
from qutip.solver import Options
import argparse
from scipy.fft import fft, ifft
from src.qutip_lab.utils import counting_multiplicity

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
    type=int,
    help="maximum value of the sigma in the gaussian noise (default=40)",
    default=40,
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
    default=0.05,
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

parser.add_argument(
    "--noise_type",
    type=str,
    help="type of noise that can be either 'gaussian' or 'uniform' (default=gaussian)",
    default="gaussian",
)


parser.add_argument(
    "--hmax",
    type=float,
    help="magnitude of the disorder (default=1.0)",
    default=1.0,
)


args = parser.parse_args()


class Driving:
    def __init__(
        self,
        t_resolution: int,
        dt: float,
        sigma: int,
        c: float,
    ) -> None:
        self.t_resolution = t_resolution
        self.dt = dt
        self.i = i

        self.h = None

        noise = np.random.normal(loc=0, scale=c, size=t_resolution)
        noise_f = fft(noise)
        noise_f[sigma:] = 0.0
        noise = ifft(np.sqrt(noise_f.shape[0] / sigma) * noise_f)
        self.h = noise

    def field(self, t: float, args) -> Union[np.ndarray, float]:
        return self.h[int(t / self.dt) - 1]


class DrivingUniform:
    def __init__(
        self,
        t_resolution: int,
        dt: float,
        hmax: float,
    ) -> None:
        self.t_resolution = t_resolution
        self.dt = dt
        self.i = i

        self.h = None
        noise = np.random.uniform(0.0, hmax, size=t_resolution)
        self.h = noise

    def field(self, t: float, args) -> Union[np.ndarray, float]:
        return self.h[int(t / self.dt) - 1]


# size of the system
size: int = args.size

# periodic boundary conditions
pbc: bool = True

np.random.seed(args.seed)

# coupling term
j: float = args.ji

# time interval
t_resolution: int = int(args.tf / args.dt)
t: np.ndarray = np.linspace(0, args.tf, t_resolution)


# define the driving once for all
if args.noise_type == "gaussian":
    c = np.random.uniform(0, args.c, size=args.different_gaussians)
    sigma = np.random.randint(10, args.sigma, size=args.different_gaussians)
else:
    hmax = args.hmax

n_dataset = args.n_dataset


file_name = (
    args.file_name
    + f"_size_{size}_tf_{args.tf}_dt_{args.dt}_sigma_10_{args.sigma}_c_0_{args.c}_noise_{args.different_gaussians}_n_dataset_{n_dataset}"
)

# return
z = np.zeros((n_dataset, t_resolution, size))
hs = np.zeros((n_dataset, t_resolution, size))


# for k in range(size):
#     print(driving_fields[k].i)
#     driving_fields[k].get_instance(0)
#     plt.plot(driving_fields[k].h)


# define the time independent hamiltonian
ham0 = SpinHamiltonian(
    direction_couplings=[("z", "z")], pbc=True, coupling_values=[1], size=size
)

# define the initial exp value
eng, psi = np.linalg.eigh(ham0.qutip_op)
psi0 = counting_multiplicity(psi, eng)
print(psi0)
print(np.dot(np.conj(psi0), psi0))
print(eng)
psi0 = qutip.Qobj(psi0, shape=psi0.shape, dims=([[2 for i in range(size)], [1]]))
print()
obs: List[qutip.Qobj] = []
for i in range(size):
    x = SpinOperator(index=[("x", i)], coupling=[1.0], size=size, verbose=1)
    # print(f"x[{i}]=", x.qutip_op, "\n")
    obs.append(x.qutip_op)
    z[:, 0, i] = x.expect_value(psi0)


print("check the value=", z[0, 0, :])


for sample in trange(n_dataset):
    # define the instance
    ham = [ham0.qutip_op]

    # define the time dependent part
    for i in range(size):
        if args.noise_type == "gaussian":
            driving = Driving(
                t_resolution=t_resolution,
                dt=args.dt,
                sigma=sigma[sample % args.different_gaussians],
                c=c[sample % args.different_gaussians],
            )
        else:
            driving = DrivingUniform(
                t_resolution=t_resolution,
                dt=args.dt,
                hmax=hmax,
            )

        hs[sample, :, i] = driving.h
        ham.append([obs[i], driving.field])
    # print("obs=", obs)
    output = qutip.sesolve(ham, psi0, t, e_ops=obs)
    # this is a shame
    for i in range(size):
        z[sample, :, i] = output.expect[i]

    if sample % args.checkpoint == 0:
        np.savez(file_name, density=z, potential=hs, time=t)

np.savez(file_name, density=z, potential=hs, time=t)

# options = Options(num_cpus=4, atol=1e-20)
