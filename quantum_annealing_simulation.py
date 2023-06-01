# %% Imports
import torch
from src.tddft_methods.adiabatic_tddft import QuantumAnnealing
import numpy as np
from src.training.models_adiabatic import Energy_XXZX
import matplotlib.pyplot as plt

# %% Data
data = np.load(
    "data/kohm_sham_approach/dataset_2channels_annealing_time_1000_test_h_2.7_j_1_1nn_n_10.npz"
)
z = data["density"]
h = data["potential"]
e = data["energy"]
ndata = 100
t_steps = 1000


print(h.shape)
for i in range(h.shape[-1]):
    plt.plot(h[0, :, i])
plt.show()


# %%
z_torch = torch.from_numpy(z)[:ndata]
h_torch = torch.from_numpy(h)[:ndata]
e_target = torch.from_numpy(e)[:ndata]
model = torch.load(
    "model_rep/kohm_sham/model_zzxz_reduction_150k_unet_l_train_8_[40, 40, 40, 40, 40, 40]_hc_5_ks_1_ps_6_nconv_0_nblock",
    map_location="cpu",
)
model.eval()
energy = Energy_XXZX(model=model)

idx = 1
time = torch.linspace(0, 1, t_steps)
alpha = time  # torch.sigmoid(time)
h_sample = h_torch
print(h_sample.shape)
z_target = z_torch
# %% RUN THE QUANTUM ANNEALING
qa = QuantumAnnealing(
    z_init=z_torch[:, 0, :],
    h=h_sample,
    energy=energy,
    lr=0.1,
    annealing_steps=40,
    e_target=e_target,
)

qa.run()


# %%
print(qa.z.shape)
# %%
plt.plot(time.numpy(), qa.eng.numpy()[idx])
plt.plot(time.numpy(), e[idx] / 8)
plt.show()

plt.plot(time.numpy(), np.abs(qa.eng.numpy()[idx] - e[idx] / 8))
plt.show()

# %%
print(qa.z.shape)
print(z_target.shape)
dz = torch.abs(qa.z - z_target).mean(-1).numpy() / torch.abs(z_target).mean(-1)

plt.plot(time.numpy(), dz)
plt.show()
# %%
for idx in range(0, 1000, 100):
    plt.plot(qa.z[idx].numpy())
    plt.plot(z_target[idx].numpy())
    plt.show()
plt.plot(qa.z[-1].numpy())
plt.plot(z_target[-1].numpy())
plt.show()
# %%
