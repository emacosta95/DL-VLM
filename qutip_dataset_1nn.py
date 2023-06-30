from src.qutip_lab.qutip_class import SpinOperator, SpinHamiltonian, SteadyStateSolver
import numpy as np
import qutip
from tqdm import tqdm, trange
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
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
    "--n_initial_field",
    type=int,
    help="number of diffenent constant initial field",
    default=100,
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
    help="type of noise that can be either 'periodic' ,'uniform' or 'gaussian' (default=periodic)",
    default="periodic",
)


parser.add_argument(
    "--hmax",
    type=float,
    help="magnitude of the disorder (default=1.0)",
    default=1.0,
)

parser.add_argument(
    "--h0",
    type=float,
    help="external field at the initial time (default=5.0)",
    default=5.0,
)

parser.add_argument(
    "--hf",
    type=float,
    help="external field at final time (default=0.0)",
    default=0.0,
)


parser.add_argument(
    "--a",
    type=float,
    help="magnitude of the disorder (default=1.0)",
    default=1.0,
)


parser.add_argument(
    "--omega",
    type=float,
    help="magnitude of the disorder (default=1.0)",
    default=1.0,
)

parser.add_argument(
    "--rate",
    type=float,
    help="rate of the evolution of the disorder annealing (default=1.0)",
    default=1.0,
)

args = parser.parse_args()


class DrivingUniformAnnealing:
    def __init__(
        self,
        t_final: float,
        hf: float,
        h0: float,
        rate: float,
    ) -> None:
        self.tf: float = t_final
        self.hf: float = hf
        self.rate = rate
        self.h0 = h0

    def field(self, t: float, args) -> Union[np.ndarray, float]:
        f_t = np.exp(
            -1 * self.rate * t
        )  # (np.exp(1 - self.rate * t) - 1) / (np.exp(1) - 1)
        return self.hf + (self.h0 - self.hf) * f_t


class DrivingPeriodic:
    def __init__(
        self,
        t_resolution: int,
        dt: float,
        a: float,
        omega: float,
    ) -> None:
        self.t_resolution = t_resolution
        self.dt = dt
        self.i = i

        self.h = None
        a_i = np.random.uniform(0.1, a, size=4)
        omega_i = np.random.uniform(0.1, omega, size=4)
        t = np.linspace(0, dt * t_resolution, t_resolution)
        h_i = a_i[:, None] * np.sin(omega_i[:, None] * t[None, :])
        self.h = np.average(h_i, axis=0)

    def field(self, t: float, args) -> Union[np.ndarray, float]:
        return self.h[int(t / self.dt) - 1]


class DrivingDisorder:
    def __init__(
        self,
        t_resolution: int,
        dt: float,
        h_init: np.ndarray,
        h_final: np.ndarray,
        rate: float,
        i: int,
    ) -> None:
        self.t_resolution = t_resolution
        self.dt = dt

        time = np.linspace(
            -self.dt * self.t_resolution / 2,
            self.dt * self.t_resolution / 2,
            self.t_resolution,
        )

        self.h = (
            h_init[None, :] * (1 - np.exp(time * rate)[:, None]) / 2
            + h_final[None, :] * (1 + np.exp(time * rate)[:, None]) / 2
        )  # initial field

        self.i = i

    def field(self, t: float, args) -> Union[np.ndarray, float]:
        return self.h[int(t / self.dt), self.i]


# size of the system
size: int = args.size

# periodic boundary conditions
pbc: bool = True

# fix the random seed
np.random.seed(args.seed)

# coupling term
j: float = args.ji

# time interval
t_resolution: int = int(args.tf / args.dt)
t: np.ndarray = np.linspace(0, args.tf, t_resolution)

# number of instances
n_dataset = args.n_dataset

# define the driving once for all
if args.noise_type == "gaussian":
    c = np.random.uniform(0, args.c, size=args.different_gaussians)
    sigma = np.random.randint(1, args.sigma, size=args.different_gaussians)
elif args.noise_type == "disorder":
    hmax = args.hmax
elif args.noise_type == "uniform":
    rate = np.linspace(0, args.rate, n_dataset)


if args.noise_type == "periodic":
    file_name = (
        args.file_name
        + f"_size_{size}_tf_{args.tf}_dt_{args.dt}_a_{args.a}_omega_{args.omega}_initial_field_{args.n_initial_field}_n_dataset_{n_dataset}"
    )
elif args.noise_type == "disorder":
    file_name = (
        args.file_name
        + f"_disorder_size_{size}_tf_{args.tf}_dt_{args.dt}_rate_{args.rate}_h_{np.e:.1f}_n_dataset_{n_dataset}"
    )
elif args.noise_type == "uniform":
    file_name = (
        args.file_name
        + f"_uniform_size_{size}_tf_{args.tf}_dt_{args.dt}_rate_{args.rate}_h0_{args.h0}_hf_{args.hf}_n_dataset_{n_dataset}"
    )

# initialization of the return value
z: np.ndarray = np.zeros((n_dataset, t_resolution, size))
hs: np.ndarray = np.zeros((n_dataset, t_resolution, size))
x: np.ndarray = np.zeros((n_dataset, t_resolution, size))


# define the initial exp value
obs: List[qutip.Qobj] = []
obs_x: List[qutip.Qobj] = []
for i in range(size):
    z_op = SpinOperator(index=[("z", i)], coupling=[1.0], size=size, verbose=1)
    # print(f"x[{i}]=", x.qutip_op, "\n")
    x_op = SpinOperator(index=[("x", i)], coupling=[1.0], size=size, verbose=1)
    obs.append(z_op.qutip_op)
    obs_x.append(x_op.qutip_op)


for sample in trange(n_dataset):
    # define the initial time independent Hamiltonian

    # if args.noise_type == "gaussian" and sample % args.n_initial_field == 0:
    #     ham0 = SpinHamiltonian(
    #         direction_couplings=[("z", "z")],
    #         field_directions=[("x")],
    #         pbc=True,
    #         coupling_values=[1.0],
    #         field_values=[1.0],
    #         size=size,
    #     )
    #     # define the term \sum_i h_i z_i
    #     ham_ext = SpinOperator(
    #         index=[("z", i) for i in range(size)], coupling=h0, size=size
    #     )

    # define the time dependent part
    # initialize the driving
    print(f"noise={args.noise_type}")
    if args.noise_type == "uniform":
        h0 = args.h0  # np.random.uniform(-2, 2, size=1)
        hf = args.hf
        ham0 = SpinHamiltonian(
            direction_couplings=[("z", "z")],
            field_directions=[("x"), ("z")],
            pbc=True,
            coupling_values=[1.0],
            field_values=[1.0, 0.0],
            size=size,
        )

        hamExt = SpinHamiltonian(
            field_directions=[("z")],
            pbc=True,
            field_values=[h0],
            size=size,
        )
        # define the term \sum_i h_i z_i

        # compute the initial state as groundstate of H0
        eng, psi0 = np.linalg.eigh(ham0.qutip_op + hamExt.qutip_op)
        # psi0 = counting_multiplicity(psi0, eng)
        # we take into account symmetries even if there shouldn't be
        # degeneracies
        psi0 = qutip.Qobj(
            psi0[:, 0], shape=psi0.shape, dims=([[2 for i in range(size)], [1]])
        )

    # elif args.noise_type == "disorder":
    #     # initial constant field
    #     h0 = np.random.uniform(0, np.e, size=size)
    #     hf = np.random.uniform(0, np.e, size=size)

    #     ham0 = SpinHamiltonian(
    #         direction_couplings=[("z", "z")],
    #         field_directions=[("x")],
    #         pbc=True,
    #         coupling_values=[1.0],
    #         field_values=[1.0],
    #         size=size,
    #     )
    #     # define the term \sum_i h_i z_i
    #     ham_ext = SpinOperator(
    #         index=[("z", i) for i in range(size)], coupling=h0, size=size
    #     )

    # define the instance
    ham = [ham0.qutip_op]

    # for an inhomogeneous field a loop runs
    # for each site
    for i in range(size):
        # gaussian type
        if args.noise_type == "uniform":
            driving = DrivingUniformAnnealing(
                t_final=args.tf,
                hf=args.hf,
                h0=args.h0,
                rate=rate[sample],
            )

            f_t = np.exp(
                -1 * rate[sample] * t
            )  # (np.exp(1 - rate[sample] * t) - 1) / (np.exp(1) - 1)

            hs[sample, :, i] = (args.h0 - args.hf) * f_t + args.hf
            # we add the time dependent term in the Hamiltonian
            # following the Qutip solver requests
            ham.append([obs[i], driving.field])

        # periodic type
        # elif args.noise_type == "periodic":
        #     driving = DrivingPeriodic(
        #         t_resolution=t_resolution, dt=args.dt, a=args.a, omega=args.omega
        #     )

        #     hs[sample, :, i] = driving.h + h0[sample % args.n_initial_field]
        #     # we add the time dependent term in the Hamiltonian
        #     # following the Qutip solver requests
        #     ham.append([obs[i], driving.field])

        elif args.noise_type == "disorder":
            driving = DrivingDisorder(
                t_resolution=t_resolution,
                dt=args.dt,
                h_init=h0,
                h_final=hf,
                rate=args.rate,
                i=i,
            )

            hs[sample, :, :] = driving.h
            # we add the time dependent term in the Hamiltonian
            # following the Qutip solver requests
            ham.append([obs[i], driving.field])

    # solver
    output = qutip.sesolve(ham, psi0, t, e_ops=obs + obs_x)
    # upload the outcomes in z (density) and x
    for i in range(size):
        z[sample, :, i] = output.expect[i]
    for i in range(size):
        x[sample, :, i] = output.expect[size + i]

    # save it every args.checkpoint times
    if sample % args.checkpoint == 0:
        np.savez(file_name, density=z, potential=hs, time=t, transverse_magnetization=x)

# global save at the end
np.savez(file_name, density=z, potential=hs, time=t, transverse_magnetization=x)
