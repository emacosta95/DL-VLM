from src.training.models_adiabatic import Energy_XXZX, Energy_reduction_XXZX
from src.gradient_descent import GradientDescent
import torch
import numpy as np

data = np.load(
    "data/kohm_sham_approach/uniform/reduction_2_input_channel_dataset_h_0.0_2.0_omega_1_j_1_1nn_n_100.npz"
)

ndata = 10

z = data["density"]
f = data["density_F"]
h = data["potential"]
e = data["energy"]

z_torch = torch.from_numpy(z[:ndata])
f_torch = torch.from_numpy(f[:ndata])
h_torch = torch.from_numpy(h[:ndata])
e_torch = torch.from_numpy(e[:ndata])


model = torch.load(
    "model_rep/kohm_sham/disorder/model_zzxz_2_input_channel_dataset_h_mixed_0.0_5.0_h_0.0-2.0_j_1_1nn_n_500k_unet_l_train_8_[40, 40, 40, 40, 40, 40]_hc_5_ks_1_ps_6_nconv_0_nblock",
    map_location="cpu",
)
model.eval()

energy = Energy_XXZX(model=model)


gd = gd = GradientDescent(
    n_instances=10,
    run_name="280723_uniform_0_2_new_model",
    loglr=-1,
    n_init=z_torch,
    cut=2,
    n_ensambles=1,
    energy=energy,
    target_path="data/kohm_sham_approach/uniform/reduction_2_input_channel_dataset_h_0.0_2.0_omega_1_j_1_1nn_n_100.npz",
    epochs=3000,
    variable_lr=False,
    early_stopping=False,
    L=8,
    resolution=1,
    final_lr=10,
    num_threads=10,
    device="cpu",
    seed=235,
    logdiffsoglia=10,
    save=True,
)

gd.run()
