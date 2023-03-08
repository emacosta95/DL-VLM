#%%
import torch
import numpy as np
import matplotlib.pyplot as plt

#%%
# data = np.load(
#     "data/gaussian_driving/l_5_tf_10.0_dt_0.1_gaussians_1000_n_dataset_15000.npz"
# )

data = np.load(
    "data/gaussian_driving/l_5_tf_10.0_dt_0.1_sigma_max_1.0_gaussians_100_n_dataset_15000.npz"
)

z = torch.tensor(data["density"], dtype=torch.double)
h = torch.tensor(data["potential"], dtype=torch.double)

t = np.linspace(0, 9.8, 98)

#%%

# model = torch.load(
#     "model_rep/tddft_relu_gaussian_driving_sigma_01_1_dt_01_l_5_nt_30_30k_pbc_[40, 40, 40, 40]_hc_[11, 3]_ks_1_ps_4_nconv_0_nblock",
#     map_location="cpu",
# )

# model = torch.load(
#     "model_rep/tddft_relu_gaussian_driving_sigma_01_1_dt_01_l_5_nt_30_60k_pbc_[40, 40, 40, 40]_hc_[11, 3]_ks_1_ps_4_nconv_0_nblock",
#     map_location="cpu",
# )

model = torch.load(
    "model_rep/tddft_relu_gaussian_driving_sigma_01_1_dt_01_l_5_nt_30_120k_pbc_[40, 40, 40, 40, 40, 40, 40, 40, 40, 40]_hc_[11, 3]_ks_1_ps_10_nconv_0_nblock",
    map_location="cpu",
)
model.to(dtype=torch.double)
model.eval()

#%% test the model and compare
n_samples = 2
print(z[0, 0, :])


x = torch.cat(
    (h[:n_samples, :, :].unsqueeze(1), z[:n_samples, :, :].unsqueeze(1)), dim=1
)

z_new = model(x)
print(x.shape)
f = torch.zeros_like(x, dtype=torch.double)
f[:, 0, :, :] = h[0:n_samples]
f[:, 1, 0, :] = z[:n_samples, 0, :]


for t in range(1, 98):
    for j in range(5):
        f[:, 1, t, j] = z_new[0:n_samples, 0, t, j]
        z_new = model(f.double())

    plt.plot(f[0, 1, :, 0].detach().numpy())
    plt.plot(z[0, :, 0].detach().numpy())
    plt.show()
#%%
print(z_new.shape)
#%%

for k in range(2):
    for i in range(5):
        plt.plot(t, z_new.detach().numpy()[k, :, i])
        plt.plot(t, z.detach().numpy()[k, :, i])
        plt.plot(t, h.detach().numpy()[k, :, i], color="black", linestyle="--")
        plt.show()
# %%
for i in range(5):
    plt.plot(z.detach().numpy()[44, :, :])
plt.show()

plt.figure(figsize=(2, 30))
plt.imshow(z.detach().numpy()[44, :, :])
plt.colorbar()
plt.show()

# %%
print(z_new.shape, z.shape)
mse = np.average(
    np.abs(z_new[:].detach().numpy() - z[0:n_samples].detach().numpy()), axis=-1
)
plt.figure(figsize=(10, 10))
for t in [0, 30, 60, -1]:
    plt.hist(mse[:, t], label=t, bins=100, range=(0, 1.0))
plt.legend()
plt.show()
mse = np.average(mse, axis=0)

plt.plot(mse)
plt.axhline(y=0.1, color="red", linestyle="--")
plt.show()
# %% LOSSES
loss_train_8_layers = torch.load(
    "losses_dft_pytorch/tddft_gaussian_driving_sigma_01_1_dt_01_l_5_nt_50_15k_pbc_[40, 40, 40, 40, 40, 40, 40, 40, 40, 40]_hc_[11, 3]_ks_1_ps_10_nconv_0_nblock_loss_train"
)

loss_valid_8_layers = torch.load(
    "losses_dft_pytorch/tddft_gaussian_driving_sigma_01_1_dt_01_l_5_nt_50_15k_pbc_[40, 40, 40, 40, 40, 40, 40, 40, 40, 40]_hc_[11, 3]_ks_1_ps_10_nconv_0_nblock_loss_train"
)


loss_train_4_layers = torch.load(
    "losses_dft_pytorch/tddft_gaussian_driving_sigma_01_1_dt_01_l_5_nt_50_15k_pbc_[40, 40, 40, 40]_hc_[11, 3]_ks_1_ps_4_nconv_0_nblock_loss_train"
)

loss_valid_4_layers = torch.load(
    "losses_dft_pytorch/tddft_gaussian_driving_sigma_01_1_dt_01_l_5_nt_50_15k_pbc_[40, 40, 40, 40]_hc_[11, 3]_ks_1_ps_4_nconv_0_nblock_loss_train"
)

plt.plot(loss_train_4_layers)
plt.plot(loss_valid_8_layers)
plt.loglog()
plt.show()
# %%
import torch
from src.training.nn_blocks import MaskedConv2d

in_channels = 2
hc = [10]
kernel_size = [11, 3]
padding_mode = "zeros"

model = MaskedConv2d(
    in_channels=in_channels,
    out_channels=hc[0],
    kernel_size=kernel_size,
    padding=[
        ((kernel_size[0] - 1) // 2),
        ((kernel_size[1] - 1) // 2),
    ],
    padding_mode=padding_mode,
    mask_type="A",
)


model_2 = torch.load(
    "model_rep/tddft_relu_gaussian_driving_sigma_01_1_dt_01_l_5_nt_30_30k_pbc_[40, 40, 40, 40]_hc_[11, 3]_ks_1_ps_4_nconv_0_nblock",
    map_location="cpu",
)

model_2.eval()

print("shape", model_2.CNNBlock.block[0].weight.shape)
print("weigth=", model_2.CNNBlock.block[0].weight[0, 0])

# %%


#%%
import torch
import numpy as np
import matplotlib.pyplot as plt

#%%
# data = np.load(
#     "data/gaussian_driving/l_5_tf_10.0_dt_0.1_gaussians_1000_n_dataset_15000.npz"
# )

data = np.load(
    "data/gaussian_driving/l_5_tf_10_dt_01_sigma_01_1_gaussians_100_n_dataset_15000.npz"
)

z = torch.tensor(data["density"], dtype=torch.double)
h = torch.tensor(data["potential"], dtype=torch.double)

t = np.linspace(0, 10, 50)

model = torch.load(
    "model_rep/tddft_relu_no_memory_gaussian_driving_sigma_01_1_dt_01_l_5_nt_25_60k_pbc_[40, 40, 40, 40, 40, 40, 40, 40]_hc_[11, 3]_ks_1_ps_8_nconv_0_nblock",
    map_location="cpu",
)
model.to(dtype=torch.double)
model.eval()

n_samples = 30
z_new = model(h[0:n_samples])
z_new = z_new.squeeze()

for k in range(30):
    for i in range(5):
        plt.plot(t, z_new.detach().numpy()[k, :, i])
        plt.plot(t, z.detach().numpy()[k, :, i])
        plt.plot(t, h.detach().numpy()[k, :, i], color="black", linestyle="--")
        plt.show()
# %%
for i in range(5):
    plt.plot(z.detach().numpy()[44, :, :])
plt.show()

plt.figure(figsize=(2, 30))
plt.imshow(z.detach().numpy()[44, :, :])
plt.colorbar()
plt.show()

# %%
print(z_new.shape, z.shape)
mse = np.average(
    np.abs(z_new[:].detach().numpy() - z[0:n_samples].detach().numpy()), axis=-1
)
plt.figure(figsize=(10, 10))
for t in [0, 30, -1]:
    plt.hist(mse[:, t], label=t, bins=100, range=(0, 1.0))
plt.legend()
plt.show()
mse = np.average(mse, axis=0)

plt.plot(mse)
plt.axhline(y=0.1, color="red", linestyle="--")
plt.show()

# %%
