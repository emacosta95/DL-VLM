from src.qutip_lab.qutip_class import SpinOperator, SpinHamiltonian, SteadyStateSolver
import numpy as np
import qutip
from tqdm import tqdm,trange
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
from qutip.solver import Options
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--file_name",
    type=str,
    help="name of the path where the simulation is saved",
    default="data/simulation",
)

parser.add_argument(
    "--num_threads",
    type=int,
    help="the number of threads for pytorch (default=3)",
    default=3,
)

parser.add_argument(
    "--seed",
    type=int,
    help="seed for pytorch and numpy (default=42)",
    default=42,
)


parser.add_argument(
    "--c_max",
    type=float,
    help="maximum amplitude of the random gaussian driving",
    default=4.0,
)


parser.add_argument(
    "--sigma_max",
    type=float,
    help="maximum value of the ",
    default=3.0,
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
    "--batch",
    type=int,
    help="batch size",
    default=100,
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
    default=10000,
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
        
    def __get_instance(self,sample:int):
        self.h=hs[sample]

    def field(self, t: float, args) -> Union[np.ndarray, float]:
        return self.h[int(t / self.dt), self.i]


# size of the system
size: int = args.size

# periodic boundary conditions
pbc: bool = True

# coupling term
j: float = args.j0

# time interval
t_resolution: int = int(args.tf / args.dt)
t: np.ndarray = np.linspace(0, args.tf, t_resolution)


# define the driving once for all
c = np.random.uniform(0, args.c_max, size=args.different_gaussians)
sigma = np.random.uniform(0.1, args.sigma_max, size=args.different_gaussians)

n_dataset = args.n_dataset
batch = args.batch

corr = c[:, None, None] * np.exp(
    -(0.5 * ( (t[None, :, None] - t[None, None, :]) / sigma[:, None, None]) ** 2)
)
lambd, q = np.linalg.eigh(corr)
x = np.random.normal(size=(int(n_dataset / args.different_gaussians), t_resolution, size))
x = np.einsum("st,ati->sati", np.sqrt(np.abs(lambd)), x)
hs = np.einsum("sty,sayi->sati", q, x)
hs = hs.reshape(-1, t_resolution, size)

driving_fields=[]
for i in range(size):
    gaussian_driving=Driving(hs=hs,dt=args.dt,i)
    driving_fields.append(gaussian_driving.field)


# define the time independent hamiltonian
ham0 = SpinHamiltonian(
    direction_couplings=[("z", "z")], pbc=True, coupling_values=[1.0], size=size
)
print(ham0)

# define the initial exp value
_, psi0 = np.linalg.eigh(ham0.qutip_op)
psi0 = qutip.Qobj(psi0, shape=psi0.shape, dims=([[2 for i in range(size)], [1]]))


for sample in trange(n_dataset):
    #define the instance
    gaussian_driving.__get_instance(sample)
    ham = [ham0.qutip_op]
    obs:List[qutip.Qobj]=[]
    
    
    
    for i in range(size):
        x = SpinOperator(index=[("x", i)], coupling=[1.0], size=size, verbose=0)
        print(x)
        obs.append(x)
        ham.append([x.qutip_op, driving_fields[i]])
        
    # initialize the external field
    

    


# compute the gaussian random driving


hs = hs.reshape(-1, t_resolution, l)


# observables
obs: List[qutip.Qobj] = []
# zz=SpinOperator(index=[('z',int(size/2),'z',int(size/2)+1)],coupling=[1.],size=size)
# obs.append(zz.qutip_op)











psi0 = psi0[:, 0]



for r in np.isnan(psi0):
    if r == True:
        print(r)

x_gs = []

for i in range(size):
    x = SpinOperator(index=[("x", i)], coupling=[1.0], size=size, verbose=1)
    print(x)
    obs.append(x.qutip_op)
    x_gs.append(x.expect_value(psi))

plt.plot(x_gs)


# options = Options(num_cpus=4, atol=1e-20)

output = qutip.mcsolve(ham, psi, t, e_ops=obs)

z = output.expect[i]
