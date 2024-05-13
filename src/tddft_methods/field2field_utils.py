import torch
import torch.nn as nn
import numpy as np
from numpy.fft import fft, ifft
from typing import List
from tqdm import trange
import scipy
from scipy.sparse import linalg


def field2field_mapping(model: nn.Module, h_input: np.ndarray) -> np.ndarray:

    # nbatch x time x space -> nbatch x omega x space
    h_input_fft = fft(
        h_input,
        norm="forward",
        axis=1,
    )

    h_fft_real = np.real(h_input_fft).reshape(
        h_input_fft.shape[0], 1, h_input_fft.shape[1], -1
    )
    h_fft_imag = np.imag(h_input_fft).reshape(
        h_input_fft.shape[0], 1, h_input_fft.shape[1], -1
    )

    h_fft = np.append(h_fft_real, h_fft_imag, axis=1)
    print(h_fft.shape)
    model_input = torch.tensor(h_fft[:, :, : h_input_fft.shape[-2] // 2 + 1])
    print(model_input.shape)
    model_output = model(model_input).detach().numpy()

    # reconstruct the whole fourier transform
    symmetric_part = np.flip(model_output[:, :, 1:-1], axis=-2)
    h_eff_fft_reconstruction = np.append(model_output, symmetric_part, axis=-2)
    h_eff_fft_reconstruction[:, 1, h_input_fft.shape[-2] // 2 :] = (
        -1 * h_eff_fft_reconstruction[:, 1, h_input_fft.shape[-2] // 2 :]
    )

    h_eff = ifft(
        h_eff_fft_reconstruction[:, 0] + 1j * h_eff_fft_reconstruction[:, 1],
        norm="forward",
        axis=1,
    )

    return h_eff


def fourier2time(fourier: np.ndarray) -> np.ndarray:
    # fourier variable batch x q //2 +1 x space
    steps = fourier.shape[1] - 1
    print(steps)

    symmetric_part = np.flip(fourier[:, :, 1:-1], axis=-2)
    h_eff_fft_reconstruction = np.append(fourier, symmetric_part, axis=-2)
    # we should fix this issue, we should make this restriction UNBIASED
    h_eff_fft_reconstruction[:, 1, 100:] = -1 * h_eff_fft_reconstruction[:, 1, 100:]

    h_eff = ifft(
        h_eff_fft_reconstruction[:, 0] + 1j * h_eff_fft_reconstruction[:, 1],
        norm="forward",
        axis=1,
    )
    return h_eff


def get_the_pca(x: np.ndarray, k_max: int, batch: int):

    x = x - np.average(x, axis=0)[None, :, :, :]
    average_x = np.average(x, axis=0)

    # we need to divide in minibatch
    nbatch = x.shape[0] // batch

    total_cov = 0.0
    for i in trange(nbatch):
        x_batch = x[i * nbatch : (i + 1) * nbatch]
        cov = np.einsum("bcki,bcqi->ciqk", x_batch, x_batch)
        total_cov = total_cov + cov

    mean_cov = total_cov / x.shape[0]
    print(mean_cov.shape)
    pc = np.zeros((mean_cov.shape[0], mean_cov.shape[1], mean_cov.shape[-1], k_max))
    for i in range(mean_cov.shape[0]):
        for j in trange(mean_cov.shape[1]):
            _, eigvectors = linalg.eigsh(mean_cov[i, j], k=k_max, which="LA")
            pc[i, j] = eigvectors

    return pc, average_x
