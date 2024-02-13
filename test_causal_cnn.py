# %%
import torch
import matplotlib.pyplot as plt
from src.training.model_utils.cnn_causal_blocks import CausalConv2d, MaskedConv2d

# %%
# Implement the causal conv
conv = CausalConv2d(in_channels=1, out_channels=1, kernel_size=[1, 2], bias=False)
# kernel size with space,time
image = torch.zeros(size=(1, 1, 5, 10))

rand_pixels = torch.rand(size=(1, 1, 5, 2))


image[:, :, :, :2] = rand_pixels

plt.imshow(image.detach().numpy()[0, 0])
plt.colorbar()

# %%
print(image)
output = conv(image)
print(output.shape)
print(output)
plt.imshow(output.detach().numpy()[0, 0])
plt.colorbar()

# %% Results of the h_eff reconstruction
import torch
import matplotlib.pyplot as plt
import numpy as np

data = np.load(
    "data/dataset_h_eff/reconstruction_dataset/reconstruction_dataset_max_shift_0.6_nbatch_100_batchsize_100_steps_1000_tf_30.0.npz"
)

z = data["z_exact"]
z_reconstruction = data["z"]
h_eff_exact = data["h_eff_exact"]
h_eff = data["h_eff"]
h_eff_reconstruction = data["h_eff_reconstruction"]

print(h_eff.shape)
# %%
for i in range(1):
    for j in range(h_eff.shape[-1]):
        plt.plot(z[i, : z_reconstruction.shape[1], j], label="z")
        plt.plot(z_reconstruction[i, :, j], label="z recon")
        plt.show()

for i in range(1):
    for j in range(h_eff.shape[-1]):
        plt.plot(h_eff_exact[i, :, j], label="h eff exact")
        plt.plot(h_eff[i, :, j], label="h eff")
        plt.plot(
            h_eff_reconstruction[i, :, j],
            label="h eff reconstruction",
            linestyle="--",
            linewidth=3,
            color="red",
        )
        plt.legend()
        plt.show()


# %%
