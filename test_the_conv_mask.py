# %%
import torch
import matplotlib.pyplot as plt
from src.training.model_utils.cnn_causal_blocks import MaskedTimeConv2d

# %%
conv = MaskedTimeConv2d(
    in_channels=1, out_channels=1, kernel_size=[1, 3], bias=False, mask_type="A"
)

print(conv.weight)
# kernel size with space,time
image = torch.zeros(size=(1, 1, 5, 10))

rand_pixels = torch.rand(size=(1, 1, 5, 2))

image[:, :, :, :2] = rand_pixels
plt.imshow(image.detach().numpy()[0, 0])
plt.colorbar()

# %%

output = conv(image)
print(output)
print(image)

plt.imshow(output.detach().numpy()[0, 0])
plt.colorbar()

print(output.shape)

# %%
print(conv.weight)
# %% Testing the dependence by using autograd
image = torch.rand(size=(1, 1, 5, 10), requires_grad=True)

conv = MaskedTimeConv2d(
    in_channels=1, out_channels=1, kernel_size=[1, 3], bias=False, mask_type="A"
)

output = conv(image)
# we fix the index
idx = 3
loss = output[0, 0, idx, idx]
loss.backward()
input_gradients = image.grad

print(input_gradients[0, 0, idx, idx], "it should be zero \n")
print(input_gradients[0, 0, idx, idx - 1], "it should be not zero \n")
print(input_gradients[0, 0, idx, idx + 1], "it should be zero \n")

# %%
