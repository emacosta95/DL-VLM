import time
from ast import increment_lineno
from numpy.lib.mixins import _inplace_binary_method
import torch as pt
import torch.nn as nn
import numpy as np
from src.training.utils import initial_ensamble_random
from src.tddft_methods.kohm_sham_utils import compute_the_gradient
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import random
from typing import Optional

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")


# %% THE GRADIENT DESCENT CLASS


class GradientDescent:
    def __init__(
        self,
        n_instances: int,
        loglr: int,
        cut: int,
        logdiffsoglia: int,
        n_ensambles: int,
        target_path: str,
        energy: nn.Module,
        run_name: str,
        epochs: int,
        variable_lr: bool,
        final_lr: float,
        early_stopping: bool,
        L: int,
        resolution: int,
        seed: int,
        num_threads: int,
        device: str,
        n_init: np.array,
        save: bool,
    ):
        self.save = save

        self.device = device
        self.num_threads = num_threads
        self.seed = seed

        self.early_stopping = early_stopping
        self.variable_lr = variable_lr

        # two version for different operations
        self.dx_torch = pt.tensor(L / resolution, dtype=pt.double, device=self.device)
        self.dx = L / resolution

        self.n_instances = n_instances

        self.loglr = loglr
        if self.early_stopping:
            self.lr = (10**loglr) * pt.ones(n_ensambles, device=self.device)
        else:
            self.lr = pt.tensor(10**loglr, device=self.device)
        self.cut = cut

        self.epochs = epochs
        self.logdiffsoglia = logdiffsoglia
        self.diffsoglia = 10**logdiffsoglia
        self.n_ensambles = n_ensambles

        self.n_target = np.load(target_path)["density"]
        self.v_target = np.load(target_path)["potential"]
        self.e_target = np.load(target_path)["energy"]

        self.n_init = n_init

        self.energy = energy
        self.run_name = run_name

        if self.variable_lr:
            self.ratio = pt.exp(
                (1 / epochs) * pt.log(pt.tensor(final_lr) / (10**loglr))
            )

        # the set of the loaded data
        self.min_engs = np.array([])
        self.min_ns = np.array([])
        self.min_hist = []
        self.min_exct_hist = []
        self.eng_model_ref = np.array([])
        self.grads = np.array([])

        self.min_engs = {}
        self.min_ns = {}
        self.min_hist = {}
        self.min_exct_hist = {}
        self.eng_model_ref = {}
        self.grads = {}

        self.epochs = epochs

    def run(self) -> None:
        """This function runs the entire process of gradient descent for each instance."""

        # select number of threads
        pt.set_num_threads(self.num_threads)

        # fix the seed
        # Initialize the seed
        pt.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # loading the model
        print("loading the model...")
        self.energy = self.energy.to(device=self.device)
        self.energy.eval()
        # starting the cycle for each instance
        print("starting the cycle...")
        for idx in trange(0, self.n_instances):
            # initialize phi
            phi = self.initialize_phi()
            print(f"is leaf={phi.is_leaf}")

            # compute the gradient descent
            # for a single target sample
            self._single_gradient_descent(phi=phi, idx=idx)

    def initialize_phi(self) -> pt.tensor:
        """This routine initialize the phis using the average decomposition of the dataset (up to now, the best initialization ever found)

        Returns:
            phi[pt.tensor]: [the initialized phis with non zero gradient]
        """

        # sqrt of the initial configuration
        if self.n_ensambles != 1:
            # initialize with the random angles
            if self.logdiffsoglia == -10:
                print("train init!")
                idxs = pt.randint(0, 10000, (self.n_ensambles,))
                m_init = [self.n_init[idx] for idx in idxs]
                m_init = pt.tensor(m_init, dtype=pt.double)
                phi = pt.acos(m_init)
            else:
                m_init = 1 - 2 * pt.rand((self.n_ensambles, 2, self.n_target.shape[-1]))
                phi = pt.acos(m_init)
        elif self.n_ensambles == 1:
            # initialize with the flat density profile

            if len(self.n_init.shape) == 2:
                m_init = pt.mean(pt.tensor(self.n_init, dtype=pt.double), dim=0).view(
                    1, self.n_init.shape[-1]
                )
            else:
                m_init = pt.mean(pt.tensor(self.n_init, dtype=pt.double), dim=0).view(
                    1, 3, self.n_init.shape[-1]
                )

            phi = pt.acos(m_init)
            print(phi.shape)
        # initialize in double and device
        phi = phi.to(dtype=pt.double)
        phi = phi.to(device=self.device)
        # make it a leaft
        phi.requires_grad_(True)

        return phi

    def _single_gradient_descent(self, phi: pt.tensor, idx: int) -> tuple:
        """This routine compute the gradient descent for an energy functional
        with external potential given by the idx-th instance and kinetic energy functional determined by model.

        Args:
            phi (pt.tensor): [the sqrt of the density profile]
            idx (int): [index of the instance]
            model (nn.Module): [model which describes the kinetic energy functional]

        Returns:
           eng[np.array] : [the energy values for different initial configurations]
           exact_eng[np.array] : [an estimation of the Von Weiszacker functional]
           phi[pt.tensor] : [the minimum configuration of the run for different initial states]
           history[np.array] : [the histories of the different gradient descents]
           exact_history[np.array] : [the histories of the estimation of the different gradient descents]
        """

        # initialize the single gradient descent

        n_ref = self.n_target[idx]
        pot = pt.tensor(self.v_target[idx], device=self.device)

        history = pt.tensor([], device=self.device)

        # exact_history = np.array([])

        eng_old = pt.tensor(0, device=self.device)

        # refresh the lr every time
        if self.early_stopping:
            self.lr = (10**self.loglr) * pt.ones(self.n_ensambles, device=self.device)
        else:
            self.lr = pt.tensor(10**self.loglr, device=self.device)

        # start the gradient descent
        t_iterator = tqdm(range(self.epochs))
        for epoch in t_iterator:
            eng, phi, grad = self.gradient_descent_step(phi=phi, pot=pot)
            diff_eng = pt.abs(eng.detach() - eng_old)

            if self.early_stopping:
                self.lr[diff_eng < self.diffsoglia] = 0

            if self.variable_lr:
                self.lr = self.lr * self.ratio  # ONLY WITH FIXED EPOCHS

            if epoch == 0:
                history = eng.detach().view(1, eng.shape[0])
            elif epoch % 10000 == 0:
                history = pt.cat((history, eng.detach().view(1, eng.shape[0])), dim=0)

                density_img = np.cos(phi.clone().detach().numpy())

                if not (self.save):
                    plt.plot(density_img[0, 0, :])
                    plt.plot(self.n_target[idx, 0, :])
                    plt.title(r"$\mathbf{z}$ configuration", fontsize=20)
                    plt.show()

                    plt.plot(density_img[0, 1, :])
                    plt.plot(self.n_target[idx, 1, :])
                    plt.title(r"$\mathbf{x}$ configuration", fontsize=20)
                    plt.show()

            eng_old = eng.detach()

            if self.n_ensambles == 1:
                t_iterator.set_description(
                    f"eng={(eng[0].detach().cpu().numpy()-self.e_target[idx]/8)/(self.e_target[idx]/8):.6f} idx={idx}, "
                )
            t_iterator.refresh()

        self.checkpoints(
            eng=eng, phi=phi, idx=idx, history=history, epoch=epoch, grad=grad
        )

    def gradient_descent_step(self, phi: pt.tensor, pot: pt.Tensor) -> tuple:
        """This routine computes the step of the gradient using both the positivity and the nomralization constrain

        Arguments:
        energy[nn.Module]: [the energy functional]
        phi[pt.tensor]: [the sqrt of the density profile]

        Returns:
            eng[pt.tensor]: [the energy value computed before the step]
            phi[pt.tensor]: [the wavefunction evaluated after the step]
        """

        w = pt.cos(phi)
        eng = self.energy(w, pot)

        eng.backward(pt.ones_like(eng))

        with pt.no_grad():
            grad = phi.grad

            phi -= self.lr * grad
            phi.grad.zero_()

        return eng.clone().detach(), phi, grad.detach().cpu().numpy()

    def gradient_descent_step_ZZXZ(self, phi: pt.tensor, pot: pt.Tensor) -> tuple:
        """This routine computes the step of the gradient using both the positivity and the nomralization constrain

        Arguments:
        energy[nn.Module]: [the energy functional]
        phi[pt.tensor]: [the sqrt of the density profile]

        Returns:
            eng[pt.tensor]: [the energy value computed before the step]
            phi[pt.tensor]: [the wavefunction evaluated after the step]
        """

        psi = phi.clone().detach()

        psi_z = psi[:, 0, :].unsqueeze(1)
        psi_x = psi[:, 1, :].unsqueeze(1)
        psi_z.requires_grad_(True)

        input = pt.cat((psi_z, psi_x), dim=1)

        w = pt.cos(input)

        eng = self.energy(w, pot)
        eng.backward(pt.ones_like(eng))

        with pt.no_grad():
            grad_z = psi_z.grad.clone()
            # psi -= self.lr * grad_z
            psi_z.grad.zero_()

        psi = phi.clone().detach()

        psi_z = psi[:, 0, :].unsqueeze(1)
        psi_x = psi[:, 1, :].unsqueeze(1)
        psi_x.requires_grad_(True)

        input = pt.cat((psi_z, psi_x), dim=1)

        w = pt.cos(input)
        eng2 = self.energy(w, pot)
        eng2.backward(pt.ones_like(eng2))

        with pt.no_grad():
            grad_x = psi_x.grad.clone()
            # psi -= self.lr * grad_z
            psi_x.grad.zero_()

            phi[:, 0] -= self.lr * (grad_z[:, 0])
            phi[:, 1] -= self.lr * (grad_x[:, 0])
        return eng.clone().detach(), phi, grad_z.detach().cpu().numpy()

    def checkpoints(
        self,
        eng: np.array,
        phi: pt.tensor,
        grad: np.array,
        idx: int,
        history: np.array,
        epoch: int,
    ) -> None:
        """This function is a checkpoint save.

        Args:
        eng[np.array]: the set of energies for each initial configuration obtained after the gradient descent
        phi[pt.tensor]: the set of sqrt density profiles for each initial configuration obtained after the gradient descent
        idx[int]: the index of the instance
        history[np.array]: the history of the computed energies for each initial configuration
        epoch[int]: the current epoch in which the data are saved
        """

        # initialize the filename
        session_name = self.run_name

        name_istances = f"number_istances_{self.n_instances}"
        session_name = session_name + "_" + name_istances

        n_initial_name = f"n_ensamble_{self.n_ensambles}_different_initial"
        session_name = session_name + "_" + n_initial_name

        epochs_name = f"epochs_{epoch}"
        session_name = session_name + "_" + epochs_name

        lr_name = f"lr_{np.abs(self.loglr)}"
        session_name = session_name + "_" + lr_name

        if self.variable_lr:
            variable_name = "variable_lr"
            session_name = session_name + "_" + variable_name

        if self.early_stopping:
            diff_name = f"diff_soglia_{int(np.abs(np.log10(self.diffsoglia)))}"
            session_name = session_name + "_" + diff_name

        # considering the minimum value
        eng_min = pt.min(eng, axis=0)[0].cpu().numpy()
        idx_min = pt.argmin(eng, axis=0)

        # exact_eng_min = exact_eng.clone()[idx_min].cpu()

        phi_min = phi[idx_min]
        grad_min = grad[idx_min]
        history_min = history[:, idx_min]

        # exact_history_min = exact_history[idx_min]
        # append to the values
        if idx == 0:
            self.min_engs[epoch] = eng_min
            self.min_hist[epoch] = history_min.cpu().numpy().reshape(1, -1)

        else:
            self.min_engs[epoch] = np.append(self.min_engs[epoch], eng_min)
            self.min_hist[epoch] = np.append(
                self.min_hist[epoch], history_min.cpu().numpy().reshape(1, -1)
            )

        # self.min_exct_hist.append(exact_history_min)

        if idx == 0:
            self.min_ns[epoch] = np.cos(phi_min.cpu().detach().numpy())
            self.grads[epoch] = grad_min
        else:
            self.min_ns[epoch] = np.vstack(
                (self.min_ns[epoch], np.cos(phi_min.cpu().detach().numpy()))
            )
            self.grads[epoch] = np.vstack((self.grads[epoch], grad_min))

        # save the numpy values
        if self.save:
            if idx != 0:
                np.savez(
                    "data/gd_data/eng_" + session_name,
                    min_energy=self.min_engs[epoch],
                    gs_energy=self.e_target[0 : (self.min_engs[epoch].shape[0])],
                )
                np.savez(
                    "data/gd_data/density_" + session_name,
                    min_density=self.min_ns[epoch],
                    gs_density=self.n_target[0 : self.min_ns[epoch].shape[0]],
                    gradient=self.grads[epoch],
                )
                np.savez(
                    "data/gd_data/history_" + session_name,
                    history=self.min_hist[epoch],
                )


class GradientDescentKohmSham:
    def __init__(
        self,
        loglr: int,
        energy: nn.Module,
        epochs: int,
        seed: int,
        num_threads: int,
        device: str,
        n_init: np.array,
        h: np.array,
    ):
        self.device = device
        self.num_threads = num_threads
        self.seed = seed

        self.lr = pt.tensor(10**loglr, device=self.device)

        self.epochs = epochs

        self.n_init = n_init

        self.energy = energy

        self.h = h

        self.epochs = epochs

    def run(self) -> None:
        """This function runs the entire process of gradient descent for each instance."""

        # select number of threads
        pt.set_num_threads(self.num_threads)

        # fix the seed
        # Initialize the seed
        pt.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # loading the model
        print("loading the model...")
        self.energy = self.energy.to(device=self.device)
        self.energy.eval()
        # starting the cycle for each instance
        print("starting the cycle...")

        # initialize phi
        phi = self.initialize_phi()
        print(f"is leaf={phi.is_leaf}")

        # compute the gradient descent
        # for a single target sample
        z = self._single_gradient_descent(phi=phi)

        return z

    def initialize_phi(self) -> pt.tensor:
        """This routine initialize the phis using the average decomposition of the dataset (up to now, the best initialization ever found)

        Returns:
            phi[pt.tensor]: [the initialized phis with non zero gradient]
        """

        # sqrt of the initial configuration

        m_init = pt.tensor(self.n_init, dtype=pt.double).unsqueeze(0)

        phi = pt.acos(m_init)
        print(phi.shape)
        # initialize in double and device
        phi = phi.to(dtype=pt.double)
        phi = phi.to(device=self.device)
        # make it a leaft
        phi.requires_grad_(True)

        return phi

    def _single_gradient_descent(self, phi: pt.tensor) -> tuple:
        """This routine compute the gradient descent for an energy functional
        with external potential given by the idx-th instance and kinetic energy functional determined by model.

        Args:
            phi (pt.tensor): [the sqrt of the density profile]
            idx (int): [index of the instance]
            model (nn.Module): [model which describes the kinetic energy functional]

        Returns:
           eng[np.array] : [the energy values for different initial configurations]
           exact_eng[np.array] : [an estimation of the Von Weiszacker functional]
           phi[pt.tensor] : [the minimum configuration of the run for different initial states]
           history[np.array] : [the histories of the different gradient descents]
           exact_history[np.array] : [the histories of the estimation of the different gradient descents]
        """

        # initialize the single gradient descent

        pot = pt.tensor(self.h, device=self.device).unsqueeze(0)

        # exact_history = np.array([])

        eng_old = pt.tensor(0, device=self.device)

        # refresh the lr every time

        # start the gradient descent
        t_iterator = tqdm(range(self.epochs))
        for epoch in t_iterator:
            eng, phi, grad = self.gradient_descent_step(phi=phi, pot=pot)
            diff_eng = pt.abs(eng.detach() - eng_old)

            density_img = np.cos(phi.clone().detach().numpy())
            eng_old = eng.detach()

            t_iterator.set_description(f"eng={(eng[0].detach().cpu().numpy()):.6f} , ")
            t_iterator.refresh()

        return density_img

    def gradient_descent_step(self, phi: pt.tensor, pot: pt.Tensor) -> tuple:
        """This routine computes the step of the gradient using both the positivity and the nomralization constrain

        Arguments:
        energy[nn.Module]: [the energy functional]
        phi[pt.tensor]: [the sqrt of the density profile]

        Returns:
            eng[pt.tensor]: [the energy value computed before the step]
            phi[pt.tensor]: [the wavefunction evaluated after the step]
        """

        w = pt.cos(phi)
        eng = self.energy(w, pot)

        eng.backward(pt.ones_like(eng))

        with pt.no_grad():
            grad = phi.grad

            phi -= self.lr * grad
            phi.grad.zero_()

        return eng.clone().detach(), phi, grad.detach().cpu().numpy()
