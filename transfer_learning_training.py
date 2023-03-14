#%%
import torch
import torch.nn as nn
import argparse
import os
import random
from tqdm.notebook import tqdm, trange
import numpy as np
import torch as pt
import torch.nn as nn
import matplotlib.pyplot as plt

from src.training.models import TDDFTCNN, TDDFTCNNNoMemory, Causal_REDENT2D
from src.training.train_module import fit
from src.training.utils import (
    count_parameters,
    get_optimizer,
    make_data_loader_unet,
)
from src.tddft_methods.adiabatic_tddft import AdiabaticTDDFTNN

file_name = "data/gaussian_driving/train_size_6_tf_10.0_dt_0.05_sigma_10_40_c_0_4.0_noise_100_n_dataset_15000.npz"
time_interval = 30
device = "cuda"
lr = 0.1

data = np.load(file_name)

z = data["density"]
v = data["potential"]

z = torch.tensor(z[0:1], dtype=torch.double, device=device)
v = torch.tensor(v[0:1], dtype=torch.double, device=device)

train_dl, valid_dl = make_data_loader_unet(
    file_name=file_name,
    split=0.8,
    bs=100,
    keys=["potential", "density"],
    time_interval=time_interval,
    preprocessing=False,
)


#%% define the model
model = AdiabaticTDDFTNN(
    n_conv=4,
    activation=nn.ReLU(),
    hc=[40, 40, 40, 40],
    in_channels=1,
    kernel_size=3,
    padding_mode="circular",
    out_channels=1,
    j=1.0,
    tf=10.0,
    dt=0.05,
    device=device,
    time_interval=time_interval,
)

model = model.to(torch.double).to(device)


epochs = 100
t_bar = tqdm(train_dl)

dt = model.time[1] - model.time[0]

loss = nn.MSELoss()


z_synt = torch.zeros((1, model.time.shape[0], 6),device=device,dtype=torch.double)
for epoch in trange(epochs):
    _, psi = model.initialization(h=v)
    for p in model.parameters():
        p.requires_grad_(True)

    for t in range(time_interval):
        psi = model.time_step(dt=dt, t=model.time[t], psi=psi, h=v)
        z_ml = model.get_magnetization(psi=psi)
        
        loss_value = +loss(z_ml, z[:, t])
        z_synt[:, t, :] = z_ml[:, :].clone().detach().cpu().numpy()
    loss_value.backward()

    with torch.no_grad():
        for p in model.v_adiabatic.parameters():
            # print("grad=", p.grad)
            p -= p.grad * lr
            # print(p.grad)
        model.zero_grad()
    print(loss_value.item())

# %%
plt.plot(z.detach().cpu().numpy()[0, :, 0])
plt.plot(z_synt[0, :, 0])
plt.show()

# %%
