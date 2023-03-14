#%%
import torch
from src.tddft_methods.adiabatic_tddft import AdiabaticTDDFT
import numpy as np
import matplotlib.pyplot as plt

# check the model manifold

data = np.load(
    "data/data_for_the_adiabatic_approximation/unet_periodic_8_l_2.7_h_6000_n.npz"
)

z = data["density"]
h = data["potential"]

mu = np.average(h, axis=0)
s = np.std(h, axis=0)

plt.plot(mu)
plt.plot(mu + s, color="red")
plt.plot(mu - s, color="red")
plt.show()


# n_init = torch.tensor((1 - z[:1]) / 2, dtype=torch.complex128)
n_init = torch.tensor((1 - np.average(z, axis=0)) / 2, dtype=torch.complex128).view(
    1, -1
)
print(n_init.shape)
#%%

# data
l = 8
psi = torch.sqrt(n_init)


psi = psi.to(torch.complex128)

tf = 100.0
dt = 0.01
t_linspace = torch.linspace(0, tf, int(tf / dt))
omega = 1
hs = (
    1.1
    * (
        0.5
        * (1 + torch.cos(t_linspace * omega)[None, :, None])
        * torch.ones(size=(1, l))[:, None, :]
    )
    + 1.0
)

# hs=torch.ones((1,t_linspace,l))

plt.plot(hs.detach().numpy()[0, :, 0])
#
model = torch.load(
    "model_rep/tddft_adiabatic/h_2.7_150k_unet_no_aug_l_train_8_[40, 40, 40, 40, 40, 40]_hc_5_ks_1_ps_6_nconv_0_nblock_b",
    map_location="cpu",
)

model.eval()
model.to(torch.double)

#%%

atddft = AdiabaticTDDFT(model=model, h=hs, psi0=psi)

z = torch.zeros(size=(1, int(tf / dt), l))
z[:, 0, :] = atddft.compute_magnetization()
print(z[:, 0, :])
for t in t_linspace[:-3]:
    atddft.time_step(dt=dt, t=t)
    # print(atddft.psi)
    # plt.plot(atddft.psi[0, 0, :].detach().numpy())
    # plt.plot(np.real(atddft.psi.detach().numpy())[0], color="red")
    # plt.plot(np.imag(atddft.psi.detach().numpy())[0], color="red")
    # plt.show()
    z[:, int(t / dt) + 1, :] = atddft.compute_magnetization()

# plt.show()

# %%


plt.plot(z[0, :60, 0].detach().numpy())
plt.show()
# %%
