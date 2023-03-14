# %%


#%%
import torch
import numpy as np
import matplotlib.pyplot as plt


# data = np.load(
#     "data/gaussian_driving/l_5_tf_10.0_dt_0.1_gaussians_1000_n_dataset_15000.npz"
# )

data = np.load(
    "data/gaussian_driving/simulation_size_6_tf_10.0_dt_0.05_sigma_10_40_c_0_4.0_noise_100_n_dataset_15000.npz"
)

z = torch.tensor(data["density"][:, :128], dtype=torch.double)
h = torch.tensor(data["potential"][:, :128], dtype=torch.double)
t = data["time"][:128]


model = torch.load(
    "model_rep/unet_gaussian_driving_sigma_10_20_t_005_l_6_nt_32_15k_pbc_[40, 80, 160]_hc_[5, 3]_ks_1_ps_3_nconv_0_nblock",
    map_location="cpu",
)
model.to(dtype=torch.double)
model.eval()

n_samples = 30
z_new = model(h[0:n_samples])
z_new = z_new.squeeze()

for k in [-5, -4, -3, -2, -1]:
    for i in range(5):
        plt.plot(t, z_new.detach().numpy()[k, :, i])
        plt.plot(t, z.detach().numpy()[k, :, i])
        # plt.plot(t, h.detach().numpy()[k, :, i], color="black", linestyle="--")
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
