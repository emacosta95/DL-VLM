import torch
import torch.nn as nn
from typing import Tuple, List
from tqdm import tqdm, trange


def decreasing(val_losses, best_loss, min_delta=0.001):
    """for early stopping"""
    try:
        is_decreasing = val_losses[-1] < best_loss * (1 - min_delta)
    except:
        is_decreasing = True
    return is_decreasing


def fit(
    epochs: int,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    train_dls: List[torch.utils.data.DataLoader],
    valid_dls: List[torch.utils.data.DataLoader],
    loss_func: nn.Module,
    supervised: bool,
    checkpoint: bool,
    name_checkpoint: str,
    history_train: List,
    history_valid: List,
    history_best: List,
    patiance: int,
    early_stopping: float,
    device: str,
) -> Tuple:
    """This function fits the model using the selected optimizer.
        It will return a list with the loss values and the accuracy as a tuple (loss,accuracy).

    Argument:

    epochs: number of epochs
    model: the selected model choosen for the train
    opt: the optimization class
    train_dl: the Dataloader for the training set
    valid_dl: the DataLoader for the validation set
    loss_func: The loss function used for the training
    checkpoint: if true a model is saved every 5 epochs
    name_checkpoint: if checkpoint is true, the name of the checkpoint model
    history_train: the record of train losses over the past epochs
    history_valid: the record of valid losses over the past epochs

    return: the evolution of the train and valid losses

    """

    loss_func = loss_func

    wait = 0
    if supervised:
        r_max = -100000
    best_loss = 10 ** 9

    for epoch in trange(epochs, desc="train epoch"):

        model.train()
        loss_ave_train = 0
        loss_ave_valid = 0
        kldiv_train = 0
        kldiv_valid = 0

        for i in range(len(train_dls)):
            train_dl = train_dls[i]
            valid_dl = valid_dls[i]
            tqdm_iterator = tqdm(
                (train_dl),
                total=len(train_dl),
                desc=f"batch [loss_ave: None]",
                leave=False,
            )

            for batch in tqdm_iterator:

                batch = batch
                loss = model.train_step(batch, device)
                loss.backward()

                opt.step()
                opt.zero_grad()

                tqdm_iterator.set_description(
                    f"train batch subset-{i} [avg loss: {loss.item():.9f}]"
                )
                tqdm_iterator.refresh()

            model.eval()
            # if supervised:
            #     r2 = R2Score()

            for batch in valid_dl:
                if supervised:
                    loss = model.train_step(batch, device)
                    loss_ave_valid += loss.item()
                else:
                    loss, kldiv = model.train_generative_step(batch, device)
                    loss_ave_valid += loss.item()
                    kldiv_valid += kldiv

            for batch in train_dl:
                if supervised:
                    loss = model.train_step(batch, device)
                    loss_ave_train += loss.item()
                else:
                    loss, kldiv = model.train_generative_step(batch, device)
                    loss_ave_train += loss.item()
                    kldiv_train += kldiv
        if supervised:

            loss_ave_train = loss_ave_train / (len(train_dl) * len(train_dls))
            history_train.append(loss_ave_train)
            loss_ave_valid = loss_ave_valid / (len(valid_dl) * len(train_dls))
            history_valid.append(loss_ave_valid)
            print(loss_ave_valid)

        else:
            kldiv_train = kldiv_train / len(train_dl)
            kldiv_valid = kldiv_valid / len(valid_dl)

            loss_ave_train = loss_ave_train / len(train_dl)
            history_train.append(loss_ave_train)
            loss_ave_valid = loss_ave_valid / len(valid_dl)
            history_valid.append(loss_ave_valid)

        wait = wait + 1
        if supervised:
            metric = best_loss
        else:
            metric = best_loss
        if decreasing(history_valid, metric, early_stopping):
            wait = 0
        if wait >= patiance:
            print(f"EARLY STOPPING AT {early_stopping}")

        if checkpoint:
            if best_loss >= loss_ave_valid:
                print("Decreasing!")
                torch.save(
                    model,
                    f"model_rep/{name_checkpoint}",
                )
                best_loss = loss_ave_valid

            history_best.append(best_loss)

            if supervised:
                text = ""
            else:
                text = "_generative"

            torch.save(
                history_train,
                f"losses_dft_pytorch/{name_checkpoint}_loss_train" + text,
            )
            torch.save(
                history_valid,
                f"losses_dft_pytorch/{name_checkpoint}_loss_valid" + text,
            )
            torch.save(
                history_best,
                f"losses_dft_pytorch/{name_checkpoint}_loss_best" + text,
            )

        if supervised:
            print(
                f"loss_ave_train={loss_ave_train} \n"
                f"loss_ave_valid={loss_ave_valid} \n"
                f"best loss={best_loss} \n"
                f"epochs={epoch}\n"
            )
        else:
            print(
                f"kldiv_valid={kldiv_valid} \n"
                f"kldiv_train={kldiv_train} \n"
                f"loss_ave_train={loss_ave_train} \n"
                f"loss_ave_valid={loss_ave_valid} \n"
                f"epochs={epoch}\n"
            )

    return history_train, history_valid
