#%%
import torch
import numpy as np
import matplotlib.pyplot as plt


# data = np.load(
#     "data/gaussian_driving/train_size_6_tf_10.0_dt_0.05_sigma_10_20_c_0_2.0_noise_100_n_dataset_15000.npz"
# )

data = np.load(
    "data/uniform/train_size_6_tf_10.0_dt_0.1_sigma_1_9_c_0_4_set_of_gaussians_100_n_dataset_60000.npz"
)

n_samples = 200
z = torch.tensor(data["density"][-n_samples:, :96], dtype=torch.double)
h = torch.tensor(data["potential"][-n_samples:, :96], dtype=torch.double)
t = data["time"][:96]

#%% load the models

time_intervals = [128]


z_ml = {}
z_target = {}
for ts in time_intervals:
    model = torch.load(
        f"model_rep/cnnlstm/uniform_gaussian_noise/cnnlstm_sigma_1_9_c_0_4_set_of_gaussians_100_t_01_l_6_nt_48_150k_pbc_[40, 40, 40, 40, 40, 40, 40, 40, 40]_hc_[1]_ks_1_ps_9_nconv_0_nblock",
        map_location="cpu",
    )
    # model = torch.load(
    #     f"model_rep/causalunet/uniform_gaussian_noise/causalunet_sigma_1_9_c_0_4_set_of_gaussians_100_t_01_l_6_nt_64_15k_pbc_[80, 80, 80, 80]_hc_[11, 1]_ks_1_ps_4_nconv_0_nblock",
    #     map_location="cpu",
    # )
    model.to(dtype=torch.double)
    model.eval()

    print(h.shape)
    z_new = model(h[0:n_samples])

    z_new = z_new.squeeze()
    z_ml[ts] = z_new.detach().numpy()
    z_target[ts] = z.detach().numpy()

#%%
print(z.shape)
print(z_new.shape)

#%%
for k in [0, 1, 2, 3, 4]:
    plt.title(f"instance={k}")
    plt.plot(t, z_new.detach().numpy()[k, :])
    plt.plot(t, z.detach().numpy()[k, :])
    # plt.plot(t, h.detach().numpy()[k, :], color="black", linestyle="--")
    plt.show()

    mse_instances = np.abs(z_new[k, :].detach().numpy() - z[k, :, 0].detach().numpy())
    plt.plot(t, mse_instances)
    plt.show()

# %%

mse = np.average(
    np.abs(z_new[:, :].detach().numpy() - z[0:n_samples, :, 0].detach().numpy()),
    axis=0,
)

plt.plot(t, mse)
plt.show()


# %%
import torch
import numpy as np
import matplotlib.pyplot as plt


# data = np.load(
#     "data/gaussian_driving/train_size_6_tf_10.0_dt_0.05_sigma_10_20_c_0_2.0_noise_100_n_dataset_15000.npz"
# )

data = np.load(
    "data/uniform/train_size_6_tf_10.0_dt_0.1_sigma_1_9_c_0_4_set_of_gaussians_100_n_dataset_15000.npz"
)

z = torch.tensor(data["density"][:, :96], dtype=torch.double)
h = torch.tensor(data["potential"][:, :96], dtype=torch.double)
t = data["time"][:96]

# load the models


z_ml = {}
z_target = {}
model = torch.load(
    f"model_rep/seq2seq/uniform_gaussian_noise/regularization_01_sigma_1_9_c_0_4_set_of_gaussians_100_t_01_l_6_nt_48_150k_pbc_[40, 40, 40, 40, 40, 40, 40, 40, 40]_hc_[5, 1]_ks_1_ps_9_nconv_0_nblock",
    map_location="cpu",
)
model.to(dtype=torch.double)
model.eval()

time_step_initial = 0
print(h.shape)
z_ml = model(x=h[:10].unsqueeze(1), y=z[0:10].unsqueeze(1))
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

time_step_initial = 0
n_samples = 20
idx = 0
k = 0

h_sample = h[:n_samples]
z_sample = torch.zeros_like(h_sample)
z_sample[:, :time_step_initial] = z[:n_samples, :time_step_initial]

z_ml = model.prediction_step(
    x=h_sample[:n_samples, :], y=z_sample, time_step_initial=time_step_initial
)

print(z_ml.shape)
print(z.shape)
#%%
for s in range(n_samples):
    for i in range(1):
        plt.plot(z_ml.detach().numpy()[s, :])
        plt.plot(z[s, :, i].detach().numpy())
        plt.show()

#%%

mae = np.average(
    np.abs(z_ml.detach().numpy() - z[:, :, 0].detach().numpy()[0:n_samples]), axis=0
)

for s in range(2):
    plt.plot(mae[:])
    # plt.semilogy()
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
