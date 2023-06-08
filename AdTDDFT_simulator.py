# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.tddft_methods.adiabatic_tddft import AdiabaticTDDFT
from tqdm import tqdm

# size of the system
size: int = 8
# periodic boundary conditions
pbc: bool = True
# fix the random seed
np.random.seed(12)
# coupling term
j: float = 1.0
# time
tf = 10.0
dt = 0.001

data = np.load(
    "data/uniform/test_for_adiabatic_uniform_size_8_tf_10.0_dt_0.001_rate_3.0_h0_5.0_hf_0.0_n_dataset_50.npz"
)


z_qutip = data["density"]
h_qutip = data["potential"]
time = data["time"]

h_torch = torch.from_numpy(h_qutip)
z_torch = torch.from_numpy(z_qutip)
print(z_torch.shape)
print(h_torch.shape)
# %% initialize psi
psi = torch.zeros(size=(1, size, 2), dtype=torch.complex128)
psi[:, :, 1] = (1 - z_torch[0, 0, :]) / 2  # it's the same for everybody
psi[:, :, 0] = (1 + z_torch[0, 0, :]) / 2

print(psi[:, :, 0] * psi[:, :, 0].conj() - psi[:, :, 1] * psi[:, :, 1].conj())


# %% load the model
model = torch.load(
    "model_rep/kohm_sham/uniform/model_zzxz_reduction_150k_unet_l_train_8_[40, 40, 40, 40, 40, 40]_hc_5_ks_1_ps_6_nconv_0_nblock",
    map_location="cpu",
)
model.eval()
print(model)

# %%
run1 = AdiabaticTDDFT(
    model=model, h=h_torch, omega=1.0, device="cpu", with_grad=True, uniform_option=True
)

# %% Run
z_adiabatic = torch.zeros_like(h_torch)
z_adiabatic[:, 0, :] = 0.0
grad = torch.zeros_like(z_adiabatic)
f = torch.zeros((h_torch.shape[0], h_torch.shape[1]))
t_bar = tqdm(time[:-1])
for t in t_bar:
    # plt.plot(np.real(psi[:,0]))
    psi = run1.time_step(dt=dt, t=t, psi=psi)
    grad[:, int(t / dt) + 1, :] = run1.grad
    f[:, int(t / dt) + 1] = run1.f_values
    z_adiabatic[:, int(t / dt) + 1, :] = run1.compute_magnetization(psi=psi)
    t_bar.refresh()


np.savez(
    "data/AdTDDFT_results/AdTDDFT_uniform_080623",
    density=z_adiabatic.detach().numpy(),
    density_target=z_qutip,
    potential=h_qutip,
    gradient=grad.detach().numpy(),
    F=f.detach().numpy(),
)
