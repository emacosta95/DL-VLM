#%%
import torch
import numpy as np
import matplotlib.pyplot as plt


# data = np.load(
#     "data/gaussian_driving/train_size_6_tf_10.0_dt_0.05_sigma_10_20_c_0_2.0_noise_100_n_dataset_15000.npz"
# )

data = np.load(
    "data/periodic/test_size_6_tf_20.0_dt_0.1_sigma_10_40_c_0_4.0_noise_100_n_dataset_200.npz"
)

z = torch.tensor(data["density"][:, :192], dtype=torch.double)
h = torch.tensor(data["potential"][:, :192], dtype=torch.double)
t = data["time"][:192]

#%% load the models

time_intervals = [96]


z_ml = {}
z_target = {}
for ts in time_intervals:
    model = torch.load(
        f"model_rep/causalcnn_periodic_omega_4_amplitude_1_t_01_l_6_mixed_nt_{ts}_150k_pbc_[40, 40, 40, 40, 40, 40, 40]_hc_[11, 3]_ks_1_ps_7_nconv_0_nblock",
        map_location="cpu",
    )
    model.to(dtype=torch.double)
    model.eval()

    n_samples = 200
    z_new = model(h[0:n_samples])
    z_new = z_new.squeeze()
    z_ml[ts] = z_new.detach().numpy()
    z_target[ts] = z.detach().numpy()

for k in [0, 1, 2, 3, 4]:
    for i in range(5):
        plt.title(f"instance={k}")
        plt.plot(t, z_new.detach().numpy()[k, :, i])
        plt.plot(t, z.detach().numpy()[k, :, i])
        plt.plot(t, h.detach().numpy()[k, :, i], color="black", linestyle="--")
        plt.show()

        mse_instances = np.sqrt(
            np.abs(z_new[k, :, i].detach().numpy() - z[k, :, i].detach().numpy()) ** 2
        )
        plt.plot(t, mse_instances)
        plt.show()

# %%
for ts in time_intervals:
    mse = np.sqrt(
        np.average(
            np.abs(z_ml[ts][:] - z[0:n_samples].detach().numpy()) ** 2,
            axis=-1,
        )
    )

    mse_total = np.abs(
        np.average(z_ml[ts][:], axis=-1)
        - np.average(z[0:n_samples].detach().numpy(), axis=-1)
    )

    plt.hist(mse, label=f"{ts}")
    plt.axhline(y=0.1, color="red", linestyle="--")
plt.legend()
plt.show()

plt.plot(t, np.average(mse, axis=0))
plt.semilogy()
plt.show()

plt.plot(t, np.average(mse_total, axis=0))
plt.semilogy()
plt.show()


# %%
