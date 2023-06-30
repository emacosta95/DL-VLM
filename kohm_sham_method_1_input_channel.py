# %% Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import trange
from src.training.models_adiabatic import Energy_XXZX_1input
from src.tddft_methods.kohm_sham_utils import (
    compute_the_gradient,
    compute_the_inverse_jacobian,
    build_hamiltonian,
    initialize_psi_from_z_and_x,
    compute_the_magnetization,
)

from typing import List

# %% Data
data = np.load(
    "data/kohm_sham_approach/disorder/reduction_1_input_channel_dataset_h_2.7_costant_omega_1.0_j_1_1nn_n_150000.npz"
)

h = data["potential"]
z = data["density"]
x = data["density_F"][:, 1, :]
l = h.shape[-1]

model = torch.load(
    "model_rep/kohm_sham/disorder/model_zzxz_reduction_1_input_channel_f_h_2.7_constant_omega_1.0_j_1_1nn_150k_unet_l_train_8_[40, 40, 40, 40, 40, 40]_hc_5_ks_1_ps_6_nconv_0_nblock",
    map_location="cpu",
)
model.eval()
model = model.to(dtype=torch.double)
energy = Energy_XXZX_1input(model=model)
# Implement the Kohm Sham LOOP
z_target = torch.from_numpy(z).double()
h = torch.from_numpy(h).double()
x_target = torch.from_numpy(x).double()
idx = 0

# %% hyperparemeters
lr = 0.0001
z_measure: List = []
x_measure: List = []


# %% Kohm Sham step 1) Initialize the state from an initial magnetization

psi = initialize_psi_from_z_and_x(z=z_target.mean(0), x=x_target.mean(0))

print(psi)
print(psi.shape)

# %%
for t in trange(1000):
    #  Kohm Sham step 2) Build up the fields

    z, x = compute_the_magnetization(psi=psi)

    dz = torch.abs(z - z_target[idx]).mean(0).item()
    dx = torch.abs(x - x_target[idx]).mean(0).item()

    z_measure.append(dz)
    x_measure.append(dx)

    z = z.view(1, -1)
    x = x.view(1, -1)

    #  Kohm Sham step 3) Compute the gradients
    h_eff = compute_the_gradient(z=z, h=h[idx], energy=energy)

    #  Kohm Sham step 3bis) Compute the Jacobian and the x field

    # the first index is the x index, the second one is the z index
    inverse_jacobian_matrix = compute_the_inverse_jacobian(
        z=z, energy=energy, tol=0.001
    )
    omega_eff = torch.einsum("ij,j->i", inverse_jacobian_matrix, h_eff)

    #  Visualization
    # plt.plot(omega_eff.detach().numpy())
    # plt.plot(h_eff.detach().numpy())
    # plt.show()

    #  Kohm Sham step 4) compute the Hamiltonian

    hamiltonian = build_hamiltonian(field_x=omega_eff, field_z=h_eff)

    #  Update the field

    psi_tilde = torch.einsum("lij,lj->li", hamiltonian, psi)

    psi = (1 - lr) * psi + lr * psi_tilde

    psi = psi / torch.linalg.norm(psi, dim=-1)[:, None]


# %% Visualize results

plt.plot(z_measure)
# plt.loglog()
plt.show()

plt.plot(x_measure)
# plt.loglog()
plt.show()

# %%
