from src.training.models_adiabatic import Energy_XXZX, Energy_reduction_XXZX
from src.gradient_descent import GradientDescent
import torch
import numpy as np

data = np.load(
    "data/kohm_sham_approach/disorder/zzxyz_model/test_dataset_zzxyz_range_0.0_3.0_j_1_1nn_n_8157_l_8.npz"
)

ndata = 10

z = data["density"]
f = data["density_F"]
h = data["potential"]
e = data["energy"]

print(h.shape)
z_torch = torch.from_numpy(z[:ndata])
f_torch = torch.from_numpy(f[:ndata])
h_torch = torch.from_numpy(h[:ndata])
e_torch = torch.from_numpy(e[:ndata])

# z_torch = -0.2 * torch.ones_like(z_torch)

model = torch.load(
    "model_rep/kohm_sham/disorder/zzxyz_model/model_zzxyz_dataset_fields_0.0_3.0_j_1_1nn_n_100k_unet_l_train_8_[60, 60, 60, 60, 60, 60]_hc_5_ks_1_ps_6_nconv_0_nblock",
    map_location="cpu",
)
model.eval()

energy = Energy_XXZX(model=model)


gd = gd = GradientDescent(
    n_instances=10,
    run_name="231002_disorder_zzxyz",
    loglr=-3,
    n_init=z_torch,
    cut=2,
    n_ensambles=1,
    energy=energy,
    target_path="data/kohm_sham_approach/disorder/zzxyz_model/test_dataset_zzxyz_range_0.0_3.0_j_1_1nn_n_10000_l_8.npz",
    epochs=6000,
    variable_lr=False,
    early_stopping=False,
    L=8,
    resolution=1,
    final_lr=10,
    num_threads=3,
    device="cpu",
    seed=235,
    logdiffsoglia=10,
    save=True,
)

gd.run()
