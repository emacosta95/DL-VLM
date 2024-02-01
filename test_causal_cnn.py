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

# %%
