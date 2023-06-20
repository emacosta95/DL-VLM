# %% Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import trange

# %% Data
data = np.load(
    "data/kohm_sham_approach/disorder/reduction_f_h_mixed_delta_2_h_ave_5_j_1_1nn_n_150000.npz"
)

h = data["potential"]

# zz = data["zz"]
density = data["density"]
df = data["density_F"]
l = h.shape[-1]

model = torch.load(
    "model_rep/kohm_sham/model_zzxz_reduction_150k_unet_l_train_8_[40, 40, 40, 40, 40, 40]_hc_5_ks_1_ps_6_nconv_0_nblock",
    map_location="cpu",
)
model.eval()
model = model.to(dtype=torch.double)
# Implement the Kohm Sham LOOP

density_target = torch.from_numpy(density)
h = torch.from_numpy(h).double()

# %% Kohm Sham Hamiltonian
x_op: torch.Tensor = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
z_op: torch.Tensor = torch.tensor([[1.0, 0.0], [0, -1.0]], dtype=torch.complex128)

idx = 0
epochs = 10000
psi_old = 0.0
cost = np.zeros((epochs))
cost_psi = np.zeros((epochs))

# initialize psi
a = torch.sqrt((1 + density_target[idx, 0]) / 2)
b = torch.sqrt((1 - density_target[idx, 0]) / 2)
print(density_target[idx, 1] / torch.sqrt(1 - density_target[idx, 0] ** 2))
tetha = torch.acos(density_target[idx, 1] / torch.sqrt(1 - density_target[idx, 0] ** 2))
psi = torch.zeros(size=(l, 2), dtype=torch.complex128)
psi[:, 0] = torch.exp(-1j * tetha) * a
psi[:, 1] = b

print(torch.linalg.norm(psi, dim=-1))

psi = psi / torch.linalg.norm(psi, dim=-1)[:, None]
psi = psi.to(dtype=torch.complex128)
density = torch.zeros(size=(2, l)).double()

df_values = []

for i in trange(epochs):
    density = density.detach()
    density[0, :] = torch.einsum("la,ab,lb->l", psi.conj(), z_op, psi).double()
    density[1, :] = torch.einsum("la,ab,lb->l", psi.conj(), x_op, psi).double()

    classical_grad = torch.roll(density[0], shifts=-1) + torch.roll(
        density[0], shifts=1
    )

    density.requires_grad_(True)
    dfml = model(density.unsqueeze(0)).view(-1)
    dfml = dfml[0].sum(-1)
    df_values.append(dfml.clone().detach().numpy())
    dfml.backward()
    with torch.no_grad():
        grad = density.grad.clone()
        density.grad.zero_()

    field_z = h[idx] + grad[0, :].clone() + classical_grad
    field_x = 1 + grad[1, :].clone()

    # field[torch.abs(field) < 10**-1] = 10**-1

    hamiltonian = (
        field_x[:, None, None] * x_op[None, :, :]
        + (field_z)[:, None, None] * z_op[None, :, :]
    )
    psi_tilde = torch.einsum("lab,lb->la", hamiltonian.double(), psi.double())
    psi = psi * 0.999 + 0.001 * psi_tilde
    psi = psi / torch.linalg.norm(psi, dim=-1)[:, None]
    # z_t = (1 / psi.shape[2]) * torch.einsum("ira,ab,irb->i", (psi), z_op, psi).detach()

    dpsi = torch.mean(torch.abs(psi - psi_old))
    psi_old = psi
    cost_psi[i] = dpsi

    dz = torch.mean(torch.abs(density_target[idx, 0].abs() - density[0].abs()), dim=-1)
    cost[i] = dz.item()

# %% Visualization
print(dz)
# %%
plt.plot(cost)
plt.show()

plt.plot(df_values)
plt.show()

plt.plot(cost_psi)
plt.show()
# %%
plt.plot(density[0].detach().numpy())
plt.plot(density_target[idx, 0].detach().numpy())
plt.show()

# %%
print(hamiltonian[0])
e, psi_pop = torch.linalg.eigh(hamiltonian[0])
print(psi_pop)
print(psi[0])
# %%
