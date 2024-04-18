import torch
import torch.nn as nn
import numpy as np
from numpy.fft import fft, ifft
from typing import List


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
    h_eff_fft_reconstruction[:, 1, 100:] = -1 * h_eff_fft_reconstruction[:, 1, 100:]

    h_eff = ifft(
        h_eff_fft_reconstruction[:, 0] + 1j * h_eff_fft_reconstruction[:, 1],
        norm="forward",
        axis=1,
    )

    return h_eff
