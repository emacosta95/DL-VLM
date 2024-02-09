# %%
import matplotlib.pyplot as plt
import numpy as np


data = np.load(
    "data/dataset_h_eff/quench/dataset_quench_nbatch_3_batchsize_3_steps_1000_tf_30.0_l_8.npz"
)


z = data["z"]
h_eff = data["h_eff"]
h = data["h"]

print(h.shape)
# %%
for i in range(20, 27):
    plt.plot(h[i, :, :])
    plt.show()

# %%
for i in range(20, 27):
    plt.plot(z[i, :, :])
    plt.show()

# %%
for i in range(20, 27):
    plt.plot(h_eff[i, :, :])
    plt.show()

# %%


import matplotlib.pyplot as plt
import numpy as np


data = np.load(
    "data/dataset_h_eff/reconstruction_dataset/reconstruction_dataset_max_shift_0.6_nbatch_100_batchsize_100_steps_1000_tf_30.0.npz"
)


z = data["z"]
z_exact = data["z_exact"]
h_eff = data["h_eff"]
h_eff_exact = data["h_eff_exact"]
h = data["h"]

print(h.shape)

# %%
