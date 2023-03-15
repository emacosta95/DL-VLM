# %%
# Libraries
from typing import Dict, List, Tuple
import numpy as np
import torch as pt
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.training.dataset import ScalableCorrelationDataset

# %%


def count_parameters(model: pt.nn.Module) -> int:
    """Counts the number of trainable parameters of a module
    Arguments:
    param model: model that contains the parameters to count
    returns: the number of parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# %%


def make_data_loader_unet(
    file_name: str,
    split: float,
    bs: int,
    keys: Tuple,
    time_interval: int,
    preprocessing: bool,
) -> tuple:
    """
    This function create a data loader from a .npz file

    Arguments

    file_name: name of the npz data_file (numpy format)
    pbc: if True the input data is extended in a periodic fashion with 128 components both on the top and bottom (128+256+128)
    split: the ratio valid_data/train_data
    bs: batch size of the data loader
    img: if True reshape the x data into a one dimensional image        (N_dataset,1,dimension)
    """

    data = np.load(file_name)
    # in this way we generalize this vector-vector function
    k1 = pt.tensor(data[keys[0]])
    k2 = pt.tensor(data[keys[1]])

    p = np.random.permutation(np.arange(k1.shape[0]))
    k1 = k1[p]
    k2 = k2[p]
    n_train = int(k1.shape[0] * split)
    if preprocessing:
        x = k1.view(k1.shape[0], k1.shape[-1], k1.shape[1])
        y = k2.view(k2.shape[0], k2.shape[-1], k2.shape[1])

        train_ds = TensorDataset(
            x[0:n_train, :, :time_interval],
            y[0:n_train, :, :time_interval],
        )
        train_dl = DataLoader(train_ds, bs, shuffle=True)
        valid_ds = TensorDataset(
            x[n_train:,:, :time_interval],
            y[n_train:, :, :time_interval],
        )
        valid_dl = DataLoader(valid_ds, 2 * bs, shuffle=True)
    else:
        x = k1
        y = k2
        train_ds = TensorDataset(
            x[0:n_train, :time_interval],
            y[0:n_train, :time_interval],
        )
        train_dl = DataLoader(train_ds, bs, shuffle=True)
        valid_ds = TensorDataset(
            x[n_train:, :time_interval],
            y[n_train:, :time_interval],
        )
        valid_dl = DataLoader(valid_ds, 2 * bs, shuffle=True)

    return train_dl, valid_dl


def make_data_loader_correlation_scale(
    file_names: list,
    split: float,
    bs: int,
) -> tuple:
    """
    This function create a data loader from a .npz file

    Arguments

    file_name: name of the npz data_file (numpy format)
    pbc: if True the input data is extended in a periodic fashion with 128 components both on the top and bottom (128+256+128)
    split: the ratio valid_data/train_data
    bs: batch size of the data loader
    img: if True reshape the x data into a one dimensional image        (N_dataset,1,dimension)
    """
    ns_train = []
    corrs_train = []
    ns_valid = []
    corrs_valid = []
    for file_name in file_names:
        data = np.load(file_name)
        n = data["density"]
        corr = data["correlation"]
        N_train = int(n.shape[0] * split)
        print(N_train)
        ns_train.append(n[0:N_train])
        corrs_train.append(corr[0:N_train])
        ns_valid.append(n[N_train:])
        corrs_valid.append(corr[N_train:])

    print(f"bs={bs}")
    train_ds = ScalableCorrelationDataset(ns_train, corrs_train)
    train_dl = DataLoader(train_ds, bs, shuffle=True)
    valid_ds = ScalableCorrelationDataset(ns_valid, corrs_valid)
    valid_dl = DataLoader(valid_ds, 2 * bs, shuffle=True)

    return train_dl, valid_dl


def make_data_loader_unet_scale(
    file_names: list,
    split: float,
    bs: int,
) -> tuple:
    """
    This function create a data loader from a .npz file

    Arguments

    file_name: name of the npz data_file (numpy format)
    pbc: if True the input data is extended in a periodic fashion with 128 components both on the top and bottom (128+256+128)
    split: the ratio valid_data/train_data
    bs: batch size of the data loader
    img: if True reshape the x data into a one dimensional image        (N_dataset,1,dimension)
    """
    ns_train = []
    f_dens_train = []
    ns_valid = []
    f_dens_valid = []
    for file_name in file_names:
        data = np.load(file_name)
        n = data["density"]
        f_dens = data["density_F"]
        N_train = int(n.shape[0] * split)
        print(N_train)
        ns_train.append(n[0:N_train])
        f_dens_train.append(f_dens[0:N_train])
        ns_valid.append(n[N_train:])
        f_dens_valid.append(f_dens[N_train:])

    print(f"bs={bs}")
    train_ds = ScalableCorrelationDataset(ns_train, f_dens_train)
    train_dl = DataLoader(train_ds, bs, shuffle=True)
    valid_ds = ScalableCorrelationDataset(ns_valid, f_dens_valid)
    valid_dl = DataLoader(valid_ds, 2 * bs, shuffle=True)

    return train_dl, valid_dl


def data_loader_response(file_name: str, split: float, bs: int) -> tuple:

    data = np.load(file_name)
    n = data["density"]
    v = data["potential"]
    n = n.reshape(n.shape[0], 1, -1)
    v = v.reshape(v.shape[0], 1, -1)
    N_train = int(n.shape[0] * split)
    train_ds = TensorDataset(pt.tensor(v[0:N_train]), pt.tensor(n[0:N_train]))
    train_dl = DataLoader(train_ds, bs)
    valid_ds = TensorDataset(pt.tensor(v[N_train:]), pt.tensor(n[N_train:]))
    valid_dl = DataLoader(valid_ds, 2 * bs)
    return train_dl, valid_dl


def plotting_test(output_model: pt.tensor, target: pt.tensor):
    """This function plot the test output_model vs target

    Argument:

    output_model: The values of the trained model related to the targets

    target: the actual values of the target

    """

    output_model = output_model.detach().numpy()

    fig, ax = plt.subplots()

    ax.scatter(target, output_model)

    plt.show()


def get_optimizer(model: pt.nn.Module, lr: int) -> pt.optim.Optimizer:
    """This function fixies the optimizer

    Argument:

    model: the model which should be trained, related to the Optimizer
    lr: learning rate of the optimization process
    """

    opt = pt.optim.Adam(model.parameters(), lr=lr)

    return opt


def transfer_learning_condition(model: nn.Module) -> nn.Module:
    """Freeze the convolutional layer in order to adopt the transfer learning

    Args:

    model[nn.Module]: the cnn architecture with a sub-module called final layer


    Returns:

    model[nn.Module]: the same model with parameters partially freezed
    """

    for parameter in model.parameters():

        parameter.requires_grad_(False)

    for parameter in model.final_layer.parameters():

        parameter = parameter.requires_grad_(True)

    return model


def initial_ensamble_random(n_instances: int) -> pt.tensor:
    """This function creates the ensamble of the initial density profiles from a dataset.
    Those functions are average values of a number of subsets of the dataset.

    Argument:

    n: dataset in np.array
    n_istances: number of subsets of the dataset.

    Returns:

    the tensor of the inital values as [n_istances,resolution] in np.array


    """

    np.random.seed(42)
    pt.manual_seed(42)

    L = 14

    resolution = 256

    dx = L / resolution

    n_ensambles = pt.tensor([])

    for j in range(int(n_instances)):

        x = pt.linspace(0, L, resolution)

        # the minimum value of sigma
        # in order to avoid deltas
        min_sigma = (L / 2) * 0.001

        # generates 1, 2 or 3
        # gaussians
        if j < int(n_instances / 2):

            # localized profiles
            n_gauss = 1

        else:

            # delocalized profiles
            n_gauss = 3

        # generates n_gauss parameters
        params = pt.rand(n_gauss)

        # we need to create the rand
        # before the cycle because
        # the seed is fixed
        sigma_rand = pt.rand(n_gauss)
        shift_rand = pt.rand(n_gauss)

        # initialize the sample
        sample = pt.zeros(resolution)

        for i in range(n_gauss):

            # we define the sigma
            sigma = (L / 2) * sigma_rand[i] + min_sigma

            # the gaussian distribution
            # centered in L/2
            gauss = pt.exp((((-1 / sigma) * ((L / 2) - x) ** 2)))

            # define a random shift of
            # the average value
            shift = int(resolution * shift_rand[i])
            gauss = pt.roll(gauss, shift)

            # sum the gauss to the
            # sample
            sample = sample + params[i] * gauss

        # normalize the sample
        norm = pt.trapezoid(sample, dx=dx)
        sample = sample / norm

        # reshape to append
        sample = sample.view(1, -1)

        if j == 0:
            # initial value
            n_ensambles = sample

        else:
            # append the other
            # values
            n_ensambles = pt.cat((n_ensambles, sample), dim=0)

    return n_ensambles


def dataloader(
    type: str,
    model_name: str,
    cut: int,
    n_instances: int,
    lr: int,
    diff_soglia: int,
    n_ensambles: int,
    epochs: int,
    early_stopping: bool,
    variable_lr: bool,
):

    session_name = model_name

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
            "gradient_descent_ensamble_numpy/min_density_" + session_name + ".npz",
            allow_pickle=True,
        )

        min_n = data["min_density"]
        gs_n = data["gs_density"]
        return min_n, gs_n

    elif type == "energy":

        data = np.load(
            "gradient_descent_ensamble_numpy/min_vs_gs_gradient_descent_"
            + session_name
            + ".npz",
            allow_pickle=True,
        )

        min_eng = data["min_energy"]
        gs_eng = data["gs_energy"]
        return min_eng, gs_eng

    elif type == "history":

        data = np.load(
            "gradient_descent_ensamble_numpy/history_" + session_name + ".npz",
            allow_pickle=True,
        )

        history = data["history"]

        history_n = data["history_n"]

        return history, history_n


def from_txt_to_bool(status: str):

    if status == "True" or status == "true":
        return True
    elif status == "False" or status == "false":
        return False
    else:
        return print("boolean symbol not recognized")


def trapez(f: pt.tensor, dx: float):

    f_roll = pt.roll(f, shifts=1, dims=-1)
    return pt.sum(f_roll * f, dim=-1) * dx


# %%
