import torch
import numpy as np
from typing import List, Tuple


def loss_upload(
    models_name: List,
):

    loss_valid: List = []
    loss_train: List = []
    for name in models_name:
        loss_valid.append(torch.load("losses_dft_pytorch/" + name + "_loss_valid"))
        loss_train.append(torch.load("losses_dft_pytorch/" + name + "_loss_train"))

    return loss_valid, loss_train


def mse_analysis(
    models_name: List, h: torch.Tensor, z_target: np.ndarray, model_type: str
):
    mse: List = []

    for name in models_name:

        model = torch.load("model_rep/" + name, map_location="cpu")
        model.to(dtype=torch.double)
        model.eval()

        z_ml = model.prediction_step(x=h, y=z_target, time_step_initial=0)

        mse.append(
            np.average(
                np.abs(z_ml.detach().numpy() - z_target[:, :, 0].detach().numpy()),
                axis=0,
            )
        )

    return mse
