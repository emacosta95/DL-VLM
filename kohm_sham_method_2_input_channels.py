# %% Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from src.training.models_adiabatic import Energy_XXZX
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


# %% hyperparemeters
lr = 0.001
intermediate_step = 10
ndata = 50
iteration = 2000

z_measure: np.ndarray = np.zeros((ndata, iteration))
x_measure: np.ndarray = np.zeros((ndata, iteration))
engs: np.ndarray = np.zeros((ndata, iteration))

psi0 = initialize_psi_from_z_and_x(
    z=-1 * z_target[:, 0].mean(dim=0), x=z_target[:, 1].mean(dim=0)
)

for idx in range(ndata):
    # Kohm Sham step 1) Initialize the state from an initial magnetization

    psi = psi0.clone()

    for t in trange(iteration):
        #  Kohm Sham step 2) Build up the fields

        for i in range(intermediate_step):
            z, x = compute_the_magnetization(psi=psi)
            z = torch.cat((z.view(1, -1), x.view(1, -1)), dim=0)
            z = z.unsqueeze(0)  # the batch dimension

            #  Kohm Sham step 3) Compute the gradients
            h_eff, eng = compute_the_gradient(
                m=z, h=h[idx], energy=energy, respect_to="z"
            )

            omega_eff, _ = compute_the_gradient(
                m=z, h=h[idx], energy=energy, respect_to="x"
            )

            #  Kohm Sham step 3bis) Compute the Jacobian and the x field

            #  Visualization
            # plt.plot(omega_eff.detach().numpy())
            # plt.plot(h_eff.detach().numpy())
            # plt.show()

            #  Kohm Sham step 4) compute the Hamiltonian

            hamiltonian = build_hamiltonian(
                field_x=-1 * omega_eff[0], field_z=-1 * h_eff[0]
            )

            if i == 0:
                tot_hamiltonian = hamiltonian
            else:
                tot_hamiltonian = 0.5 * (hamiltonian + tot_hamiltonian)

            #  Update the field

            psi_tilde = torch.einsum("lij,lj->li", tot_hamiltonian, psi)
            psi = (1 - lr) * psi + lr * psi_tilde
            psi = psi / torch.linalg.norm(psi, dim=-1)[:, None]

            if i == 0:
                engs[idx, t] = eng

        z_measure[idx, t] = torch.abs(z[0, 0, :] - z_target[idx, 0]).mean(0).item()
        x_measure[idx, t] = torch.abs(z[0, 1, :] - z_target[idx, 1]).mean(0).item()

    np.savez(
        f"data/kohm_sham_approach/results/kohm_sham_uniform_0.0_2.0_ndata_{ndata}_iteration_{iteration}_intermediate_step_{intermediate_step}",
        energy=engs,
        dx=x_measure,
        dz=z_measure,
    )

# %% Visualize results
# plt.figure(figsize=(10, 10))
# plt.plot(z_measure, color="black", linewidth=5)
# plt.ylabel(r"$\Delta z$", fontsize=30)
# plt.xlabel(r"$N_t$", fontsize=30)
# plt.loglog()
# plt.tick_params(
#     top=True,
#     right=True,
#     labeltop=False,
#     labelright=False,
#     direction="in",
#     labelsize=30,
#     width=2,
# )
# plt.show()
# plt.figure(figsize=(10, 10))
# plt.plot(x_measure, color="black", linewidth=5)
# plt.ylabel(r"$\Delta x$", fontsize=30)
# plt.xlabel(r"$N_t$", fontsize=30)
# plt.loglog()
# plt.tick_params(
#     top=True,
#     right=True,
#     labeltop=False,
#     labelright=False,
#     direction="in",
#     labelsize=30,
#     width=2,
# )
# plt.show()

# plt.figure(figsize=(10, 10))
# plt.plot(engs, color="black", linewidth=5)
# plt.ylabel(r"$e$", fontsize=30)
# plt.xlabel(r"$N_t$", fontsize=30)
# plt.tick_params(
#     top=True,
#     right=True,
#     labeltop=False,
#     labelright=False,
#     direction="in",
#     labelsize=30,
#     width=2,
# )
# plt.show()

# # %%

# plt.plot(z_target[idx, 0].detach().numpy())
# plt.plot(z[0, 0].detach().numpy())
# plt.show()

# plt.plot(z_target[idx, 1].detach().numpy())
# plt.plot(z[0, 1].detach().numpy())
# plt.show()
# %%
