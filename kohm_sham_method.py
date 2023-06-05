# %% Imports
import torch
import numpy as np
import matplotlib.pyplot as plt


# %% Data
data = np.load(
    "data/kohm_sham_approach/uniform/dataset_2channels_h_5.0_j_1_1nn_n_10.npz"
)
# zz = data["zz"]
z = data["density"]
df = data["density_F"]
h = data["potential"]

model = torch.load(
    "model_rep/kohm_sham/uniform/model_zzxz_reduction_150k_unet_l_train_8_[40, 40, 40, 40, 40, 40]_hc_5_ks_1_ps_6_nconv_0_nblock",
    map_location="cpu",
)
model.eval()
model = model.to(dtype=torch.double)
# Implement the Kohm Sham LOOP

z_target = torch.from_numpy(z)
h = torch.from_numpy(h).double()

# %% Kohm Sham Hamiltonian
x_op: torch.Tensor = torch.tensor([[0.0, 1.0], [1.0, 0.0]]).double()
z_op: torch.Tensor = torch.tensor([[-1.0, 0.0], [0, 1.0]]).double()

idx = 0
epochs = 3
psi_old = 0.0
z_t: torch.Tensor = 0.5 * torch.ones(8).double()
# z_t: torch.Tensor = z_target[idx]
print(z_t)
cost = np.zeros((epochs))
cost_psi = np.zeros((epochs))
for i in range(epochs):
    z_t.requires_grad_(True)
    dfml = model(z_t.unsqueeze(0)).view(2, -1)
    dfml = dfml[0].sum(-1)
    dfml.backward()
    with torch.no_grad():
        grad = z_t.grad.clone()
        z_t.grad.zero_()

    field = h[idx] + grad.mean(-1)
    # field[torch.abs(field) < 10**-1] = 10**-1

    hamiltonian = x_op[None, :, :] + (field)[:, None, None] * z_op[None, :, :]
    print(hamiltonian)
    plt.plot((field).detach().numpy())
    plt.show()
    e, psi = torch.linalg.eig(hamiltonian)
    psi = psi / torch.linalg.norm(psi, dim=-1)[:, None]
    # z_t = (1 / psi.shape[2]) * torch.einsum("ira,ab,irb->i", (psi), z_op, psi).detach()
    z_t = (
        ((psi[:, 0, 0].conj() * psi[:, 0, 0]) - (psi[:, 0, 1].conj() * psi[:, 0, 1]))
        .detach()
        .double()
    )

    plt.plot(z_t.detach().numpy())
    plt.plot(z_target[idx])
    plt.show()

    dpsi = torch.mean(torch.abs(psi - psi_old))
    psi_old = psi
    cost_psi[i] = dpsi

    dz = torch.mean(torch.abs(z_target[idx].abs() - z_t.abs()), dim=-1)
    cost[i] = dz.item()

# %% Visualization
print(dz)
plt.plot(z_t.detach().numpy())
plt.plot(z_target[idx].detach().numpy())
plt.show()

plt.plot(grad)
plt.show()
# %%
plt.plot(cost)
plt.show()

plt.plot(cost_psi)
plt.show()
# %%
