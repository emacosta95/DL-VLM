# Libraries
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch as pt
import torch.nn as nn
from torchmetrics import R2Score
from tqdm import trange
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from src.training.models import Energy_unet
from src.training.utils import count_parameters


def dataloader(
    type: str,
    session_name: str,
    n_instances: int,
    lr: int,
    diff_soglia: int,
    n_ensambles: int,
    epochs: int,
    early_stopping: bool,
    variable_lr: bool,
):

    session_name = session_name

    name_istances = f"number_istances_{n_instances}"
    session_name = session_name + "_" + name_istances

    n_initial_name = f"n_ensamble_{n_ensambles}_different_initial"
    session_name = session_name + "_" + n_initial_name

    epochs_name = f"epochs_{epochs}"
    session_name = session_name + "_" + epochs_name

    lr_name = f"lr_{lr}"
    session_name = session_name + "_" + lr_name

    if variable_lr:
        variable_name = "variable_lr"
        session_name = session_name + "_" + variable_name

    if early_stopping:
        diff_name = f"diff_soglia_{diff_soglia}"
        session_name = session_name + "_" + diff_name

    if type == "density":

        data = np.load(
            "data/gd_data/density_" + session_name + ".npz",
            allow_pickle=True,
        )

        min_n = data["min_density"]
        gs_n = data["gs_density"]
        return min_n, gs_n

    elif type == "energy":

        data = np.load(
            "data/gd_data/eng_" + session_name + ".npz",
            allow_pickle=True,
        )

        min_eng = data["min_energy"]
        gs_eng = data["gs_energy"]
        return min_eng, gs_eng

    elif type == "history":

        data = np.load(
            "data/gd_data/history/history_" + session_name + ".npz",
            allow_pickle=True,
        )

        history = data["history"]

        history_n = data["history_n"]

        return history, history_n


def test_models_unet(
    models_name: List, data_path: List, limit_data: int, batch_size: int
):
    accuracy_prediction_energy_average = []
    accuracy_prediction_energy_std = []
    fs = []
    fmls = []
    for i, model_name in enumerate(models_name):

        data = np.load(data_path[i])
        n_std = data["density"][:limit_data]
        f_dens = data["density_F"][:limit_data]
        # e_std = data["energy"][:limit_data]
        f_std = np.average(f_dens, axis=-1)
        ds = TensorDataset(pt.tensor(n_std), pt.tensor(f_std))  # , pt.tensor(e_std))
        dl = DataLoader(ds, batch_size=batch_size)

        model = pt.load("model_rep/" + model_name, map_location="cpu")
        model.eval()
        model = model.to(dtype=pt.double)

        de = 0.0
        devde = 0
        for n, f in dl:

            n = n.to(dtype=pt.double)
            f = f.to(dtype=pt.double)
            model.eval()

            output = model(n)

            # print(f[0],pt.mean(output,dim=-1)[0])
            de = de + np.average(
                np.abs(
                    output.mean(dim=-1).detach().numpy().reshape(-1)
                    - f.detach().numpy()
                )
                / np.abs(f.detach().numpy())
            )
            print("de=", de)
            devde = devde + (1 / np.sqrt(f.shape[0])) * np.std(
                np.abs(
                    output.mean(dim=-1).detach().numpy().reshape(-1)
                    - f.detach().numpy()
                )
                / np.abs(f.detach().numpy())
            )

        print(model)
        print(f"# parameters={count_parameters(model)}")
        print(f"de={de} std(de)={devde}")
        accuracy_prediction_energy_average.append(np.average(de))
        accuracy_prediction_energy_std.append(np.average(devde))
        fs.append(f_dens)
        fmls.append(output.detach().numpy())
    return (
        (accuracy_prediction_energy_average),
        (accuracy_prediction_energy_std),
        fmls,
        fs,
    )


def nuv_representability_check(
    model: nn.Module, z: np.ndarray, v: np.ndarray, plot: bool, gs_z: np.ndarray
):

    l = z.shape[-1]
    x = pt.tensor(z, dtype=pt.double)
    x.requires_grad_(True)
    f = pt.mean(model(x), dim=-1)  # the total f value
    f.backward(pt.ones_like(f))
    with pt.no_grad():
        grad = x.grad
        grad = -l * grad.detach().numpy()
        pseudo_pot = grad
    # print(grad.shape)
    g_acc = np.average(np.abs(v - pseudo_pot), axis=-1) / np.average(np.abs(v), axis=-1)

    if plot:
        for i in range(0, 10):
            plt.plot(pseudo_pot[i], label="ml grad")
            plt.plot(v[i], label="pot")
            plt.legend()
            plt.show()

    nbatch = 10
    j_coupling = 1
    _, m, _, f, fm, e = parallel_nambu_diagonalization_ising_model(
        nbatch=nbatch,
        l=l,
        j_coupling=j_coupling,
        hs=pseudo_pot,
        device="cpu",
        pbc=True,
    )
    z_acc = np.average(np.abs(m - gs_z), axis=-1) / np.average(np.abs(z), axis=-1)

    if plot:
        for i in range(0, 10):
            plt.plot(m[i], label="pseudo pot --> z")
            plt.plot(gs_z[i], label="pot --> z")
            plt.legend()
            plt.show()

    return g_acc, z_acc


def mean_field_functional_1nn(z: np.ndarray, hs: np.ndarray):
    """The mean field functional in the 1nn Ising model

    Args:
        z (np.ndarray): transverse magnetization
        hs (np.ndarray): the external field
    """

    return -1 * np.sum(1 - z ** 2, axis=-1) + np.einsum("ai,ai->a", hs, z)


def gradient_prediction_analysis(
    models_name: List, data_path: List, ls: List, ndata: int
) -> List[List]:

    g_accs = []
    dev_g = []
    for j in range(len(models_name)):
        g_acc = []
        for i in range(len(ls)):
            data = np.load(data_path[i])
            m = data["density"]
            v = data["potential"]
            model = pt.load(
                "model_rep/" + models_name[j],
                map_location="cpu",
            )
            x = m[:ndata]
            x = pt.tensor(x, dtype=pt.double)
            x.requires_grad_(True)
            f = pt.mean(model(x), dim=-1)
            # print(f.shape)
            f.backward(pt.ones_like(f))
            with pt.no_grad():
                grad = x.grad
                grad = -ls[i] * grad.detach().numpy()
                pseudo_pot = grad
            # print(grad.shape)
            g_acc.append(
                np.sqrt(np.average((grad - v[:ndata]) ** 2, axis=-1))
                / np.sqrt(np.average((v[:ndata]) ** 2, axis=-1))
            )

        dev_g.append([np.std(g) / np.sqrt(g.shape[0]) for g in g_acc])
        g_acc = [np.average(g) for g in g_acc]
        g_accs.append(g_acc)

    return (g_accs, dev_g)


# functions
def correlation_through_the_machine(
    z: pt.tensor,
    model: nn.Module,
):
    x = pt.unsqueeze(z, dim=1)
    outputs = []
    for i, block in enumerate(model.conv_downsample):
        x = block(x)
        outputs.append(x)
        if i == 0:
            outputs_in = x.unsqueeze(0).detach().numpy()
        else:
            outputs_in = np.append(x.unsqueeze(0).detach().numpy(), outputs_in, axis=0)

    for i, block in enumerate(model.conv_upsample):
        if i == 0:
            x = block(x)
            outputs_out = x.unsqueeze(0).detach().numpy()
        else:
            x = x + outputs[model.n_conv_layers - 1 - i]
            x = block(x)
            if i <= len(model.conv_upsample) - 2:
                outputs_out = np.append(
                    x.unsqueeze(0).detach().numpy(), outputs_out, axis=0
                )
            if i == len(model.conv_upsample) - 1:
                last_output = x.detach().numpy()
    f_dens = pt.squeeze(x)
    outputs = np.append(outputs_out, outputs_in, axis=0)
    return outputs, f_dens


def makes_the_fluctuations(outputs: np.ndarray, z: pt.tensor, channels: Tuple):
    x_0 = np.append(
        z.detach().numpy().reshape(1, z.shape[0], z.shape[1]),
        outputs[:, :, channels[0], :],
        axis=0,
    )
    x_1 = np.append(
        z.detach().numpy().reshape(1, z.shape[0], z.shape[1]),
        outputs[:, :, channels[1], :],
        axis=0,
    )
    mean_x = np.average(x_0)
    dx0 = x_0 - np.average(x_0, axis=1)[:, None, :]
    mean_x = np.average(x_1)
    dx1 = x_1 - np.average(x_1, axis=1)[:, None, :]
    return (dx0, dx1)


def covtt(dx0, dx1):
    return np.average(dx0[:, None, :, None, :] * dx1[None, :, :, :, None], axis=2) / (
        np.std(dx0, axis=1)[:, None, None, :] * np.std(dx1, axis=1)[None, :, :, None]
    )
