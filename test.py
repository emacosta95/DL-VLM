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

# load the models


z_ml = {}
z_target = {}
model = torch.load(
    f"model_rep/seq2seq/seq2seq_periodic_omega_4_amplitude_1_t_01_l_6_mixed_nt_96_15k_pbc_[40, 40, 40, 40]_hc_[5, 3]_ks_1_ps_4_nconv_0_nblock",
    map_location="cpu",
)
model.to(dtype=torch.double)
model.eval()

time_step_initial = 50
print(h.shape)
z_ml = model.get_sample(x=h[:10], y=z[0:10])
z_complete = z.clone()
z_complete[:, time_step_initial:, :] = 0.0
z_complete = z_complete
# z_ml = model.prediction_step(
#    time_step_initial=time_step_initial, x=h[:10], y=z_complete[:10]
# )

print(h.shape)
print(z_ml.shape)
z_ml = z_ml.squeeze(1)
# %%
for i in range(1):
    for k in range(1):
        plt.plot(z_ml[i, :, :].detach().numpy(), color="red")
        plt.plot(z[i, :, :].detach().numpy(), color="black")
        plt.show()

for i in range(1):
    for k in range(1):
        plt.plot(
            np.abs(z[i, 0:, :].detach().numpy() - z_ml[i, 0:, :].detach().numpy()),
            color="black",
        )
        plt.semilogy()
        plt.show()
# %%

time_step_initial = 20
idx = 0
k = 0

h_sample = h[:1]
z_sample = torch.zeros_like(h_sample)
z_sample[:, :time_step_initial] = z[:1, :time_step_initial]

z_ml = model.prediction_step(
    x=h_sample[:1, :], y=z_sample, time_step_initial=time_step_initial
)
#%%
plt.plot(z_ml.detach().numpy()[:, 0])
plt.plot(z[0, :, 0].detach().numpy())
plt.show()


# %%
print(model.decoder.causal_embedding.weight.data[:, 0, -1, :])
# %% We check the gradient to understand if the output does not depend on the forward values

input = h[:10].unsqueeze(1)
output = z[:10].unsqueeze(1).clone()
output.requires_grad_(True)
# z_ml = model.decoder.conv_part[0](model.decoder.causal_embedding(output))
# z_ml = model.decoder.causal_embedding(output)
# for block in model.decoder.conv_part:
#    z_ml = block(z_ml)

# z_ml = model.decoder.causal_embedding(output)
z_ml = model(x=input, y=output)
t_point = 10
z_ml[0, 0, t_point, 0].backward(torch.ones_like(z_ml[0, 0, t_point, 0]))

grad = output.grad

print(grad[0, :, t_point - 10 : t_point + 10, :])
# %%
