import math
import os
import time
from multiprocessing import cpu_count, Pool  # Just to set the worker number

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW, Adagrad  # Supported Optimizers
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split

# Project Dependencies
from utils import Categorical, Sequence_logo_all, fasta_read, gen_data_lowT, gen_data_zeroT
import rbm_configs



class RBM(LightningModule):
    def __init__(self, config, precision="double", debug=False):
        super().__init__()
        self.h_num = config['h_num']  # Number of hidden nodes, can be variable
        self.v_num = config['v_num']   # Number of visible nodes,
        self.q = config['q']  # Number of categories the input sequence has (ex. DNA:4 bases + 1 gap)
        self.mc_moves = config['mc_moves']  # Number of MC samples to take to update hidden and visible configurations
        self.batch_size = config['batch_size']  # Pretty self explanatory

        self.epsilon = 1e-6  # Number added to denominators for numerical stability
        self.epochs = config['epochs'] # number of training iterations, needed for our weight decay function

        # Data Input
        self.fasta_file = config['fasta_file']
        self.molecule = config['molecule'] # can be protein, rna or dna currently
        assert self.molecule in ["dna", "rna", "protein"]

        # Sets worker number for both dataloaders
        if debug:
            self.worker_num = 0
        else:
            try:
                self.worker_num = config["data_worker_num"]
            except KeyError:
                self.worker_num = cpu_count()

        if hasattr(self, "trainer"): # Sets Pim Memory when GPU is being used
            if hasattr(self.trainer, "on_gpu"):
                self.pin_mem = self.trainer.on_gpu
            else:
                self.pin_mem = False
        else:
            self.pin_mem = False

        # Sequence Weighting Weights
        # Not pretty but necessary to either provide the weights or to import from the fasta file
        # To import from the provided fasta file weights="fasta" in intialization of RBM
        weights = config['sequence_weights']
        loss_type = config['loss_type']
        sample_type = config['sample_type']
        optimizer = config['optimizer']
        self.lr = config['lr']
        lr_final = config['lr_final']
        self.wd = config['weight_decay'] # Put into weight decay option in configure_optimizer, l2 regularizer
        self.decay_after = config['decay_after'] # hyperparameter for when the lr decay should occur
        self.l1_2 = config['l1_2']  # regularization on weights, ex. 0.25
        self.lf = config['lf']  # regularization on fields, ex. 0.001
        self.seed = config['seed']

        # All weights are scaled by this muliplier
        try:
            self.weight_multiplier = config["weight_multiplier"]
        except KeyError:
            self.weight_multiplier = 1.

        if weights is None:
            self.weights = None
        elif type(weights) == str:
            if weights == "fasta": # Assumes weights are in fasta file
                self.weights = "fasta"
        elif type(weights) == torch.tensor:
            self.weights = weights.numpy()
        elif type(weights) == np.ndarray:
            self.weights = weights
        else:
            print(f"Provided Weights of type {type(weights)} Not Supported, Must be None, a numpy array, torch tensor, or 'fasta'")
            exit(1)


        # loss types are 'energy' and 'free_energy' for now, controls the loss function primarily
        # sample types control whether gibbs sampling from the data points or parallel tempering from random configs are used
        # Switches How the training of the RBM is performed

        if loss_type not in ['energy', 'free_energy']:
            print(f"Loss Type {loss_type} not supported")
            exit(1)
        if sample_type not in ['gibbs', 'pt', 'pcd']:
            print(f"Sample Type {sample_type} not supported")
            exit(1)

        self.loss_type = loss_type
        self.sample_type = sample_type

        # optimizer options
        if optimizer == "SGD":
            self.optimizer = SGD
        elif optimizer == "AdamW":
            self.optimizer = AdamW
        elif optimizer == "Adagrad":
            self.optimizer = Adagrad
        else:
            print(f"Optimizer {optimizer} is not supported")
            exit(1)

        if lr_final is None:
            self.lrf = self.lr * 1e-2
        else:
            self.lrf = lr_final

        # Normal dist. times this value sets initial weight values
        self.weight_intial_amplitude = np.sqrt(0.1 / self.v_num)

        ## Only Used as a test so far, not super effective
        # self.reconstruction_loss = nn.MSELoss(reduction="mean")

        ## Might Need if grad values blow up
        # self.grad_norm_clip_value = 1000 # i have no context for setting this value at all lol, it isn't in use currently but may be later

        # Pytorch Basic Options
        torch.manual_seed(self.seed)  # For reproducibility
        supported_precisions = {"double": torch.float64, "single": torch.float32}
        try:
            torch.set_default_dtype(supported_precisions[precision])
        except:
            print(f"Precision {precision} not supported.")
            exit(-1)

        self.params = nn.ParameterDict({
            # weights
            'W_raw': nn.Parameter(self.weight_intial_amplitude * torch.randn((self.h_num, self.v_num, self.q), device=self.device)),
            # hidden layer parameters
            'theta+': nn.Parameter(torch.zeros(self.h_num, device=self.device)),
            'theta-': nn.Parameter(torch.zeros(self.h_num, device=self.device)),
            'gamma+': nn.Parameter(torch.ones(self.h_num, device=self.device)),
            'gamma-': nn.Parameter(torch.ones(self.h_num, device=self.device)),

            # Used in PT Sampling / AIS
            '0theta+': nn.Parameter(torch.zeros(self.h_num, device=self.device)),
            '0theta-': nn.Parameter(torch.zeros(self.h_num, device=self.device)),
            '0gamma+': nn.Parameter(torch.ones(self.h_num, device=self.device)),
            '0gamma-': nn.Parameter(torch.ones(self.h_num, device=self.device)),
            # visible layer parameters
            'fields': nn.Parameter(torch.zeros((self.v_num, self.q), device=self.device)),
            'fields0': nn.Parameter(torch.zeros((self.v_num, self.q), device=self.device), requires_grad=False),
        })

        self.W = torch.zeros((self.h_num, self.v_num, self.q), device=self.device, requires_grad=False)

        # Saves Our hyperparameter options into the checkpoint file generated for Each Run of the Model
        # i. e. Simplifies loading a model that has already been run
        self.save_hyperparameters()

        # Constants for faster math
        self.logsqrtpiover2 = torch.tensor(0.2257913526, device=self.device, requires_grad=False)
        self.pbis = torch.tensor(0.332672, device=self.device, requires_grad=False)
        self.a1 = torch.tensor(0.3480242, device=self.device, requires_grad=False)
        self.a2 = torch.tensor(-0.0958798, device=self.device, requires_grad=False)
        self.a3 = torch.tensor(0.7478556, device=self.device, requires_grad=False)
        self.invsqrt2 = torch.tensor(0.7071067812, device=self.device, requires_grad=False)
        self.sqrt2 = torch.tensor(1.4142135624, device=self.device, requires_grad=False)

        # Initialize PT members, might b
        self.initialize_PT(5, n_chains=None, record_acceptance=True, record_swaps=True)

    def initialize_PT(self, N_PT, n_chains=None, record_acceptance=False, record_swaps=False):
        self.N_PT = N_PT
        self.n_chains = n_chains # either None which defaults to batch_size or a set integer value
        self.record_acceptance = record_acceptance
        self.record_swaps = record_swaps

        # self.update_betas()
        self.betas = torch.arange(N_PT) / (N_PT - 1)
        self.betas = self.betas.flip(0)

        if n_chains is None:
            self.n_chains = self.batch_size
        else:
            self.n_chains = n_chains

        if self.record_swaps:
            self.particle_id = [torch.arange(N_PT).unsqueeze(1).expand(N_PT, self.n_chains)]
        else:
            self.particle_id = None

        if self.record_acceptance:
            self.mavar_gamma = 0.95
            self.acceptance_rates = torch.zeros(N_PT - 1, device=self.device)
            self.mav_acceptance_rates = torch.zeros(N_PT - 1, device=self.device)

        # gen data
        self.count_swaps = 0
        self.last_at_zero = None
        self.trip_duration = None
        self.update_betas_lr = 0.1
        self.update_betas_lr_decay = 1

    def prep_W(self): # enforces zero sum gauge on the weights
        self.W = self.params['W_raw'] - self.params['W_raw'].sum(-1).unsqueeze(2) / self.q

    ############################################################# RBM Functions
    ## Compute Psudeo likelihood of given visible config
    def pseudo_likelihood(self, v):
        config = v.long()
        ind_x = torch.arange(config.shape[0], dtype=torch.long, device=self.device)
        ind_y = torch.randint(self.v_num, (config.shape[0],), dtype=torch.long, device=self.device)  # high, shape tuple, needs size, low=0 by default
        E_vlayer_ref = self.energy_v(config) + self.params['fields'][ind_y, config[ind_x, ind_y]]
        output_ref = self.compute_output_v(config) - self.W[:, ind_y, config[ind_x, ind_y]].T
        fe = torch.zeros([config.shape[0], self.q], device=self.device)
        for c in range(self.q):
            output = output_ref + self.W[:, ind_y, c].T
            E_vlayer = E_vlayer_ref - self.params['fields'][ind_y, c]
            fe[:, c] = E_vlayer - self.logpartition_h(output)
        return - fe[ind_x, config[ind_x, ind_y]] - torch.logsumexp(- fe, 1)

    ## Used in our Loss Function
    def free_energy(self, v):
        return self.energy_v(v) - self.logpartition_h(self.compute_output_v(v))

    ## Not used but may be useful
    def free_energy_h(self, h):
        return self.energy_h(h) - self.logpartition_v(self.compute_output_h(h))

    ## Total Energy of a given visible and hidden configuration
    def energy(self, v, h, remove_init=False):
        return self.energy_v(v, remove_init=remove_init) + self.energy_h(h, remove_init=remove_init) - self.bilinear_form(v, h)

    ## Total Energy of a given visible and hidden configuration
    def energy_PT(self, v, h, remove_init=False):
        E = torch.zeros((self.N_PT, v.shape[1]), device=self.device)
        for i in range(self.N_PT):
            E[i, :] = self.energy_v(v[i, :, :], remove_init=remove_init) + self.energy_h(h[i, :, :], remove_init=remove_init) - self.bilinear_form(v[i, :, :], h[i, :, :])
        return E

    ## Computes term sum(i, u) h_u w_iu(vi)
    def bilinear_form(self, v, h):
        output = torch.zeros((v.shape[0], self.h_num), device=self.device)
        vd = v.long()
        for u in range(self.h_num):  # for u in h_num
            for i in range(self.v_num):  # for
                # we'll try this first
                output[:, u] += self.W[u][i][vd[:, i]]

        return torch.sum(h * output, 1)

    ############################################################# Individual Layer Functions
    def transform_v(self, I):
        return torch.argmax(I + self.params["fields"], dim=-1)

    def transform_h(self, I):
        a_plus = (self.params[f'gamma+']).unsqueeze(0)
        a_minus = (self.params[f'gamma-']).unsqueeze(0)
        theta_plus = (self.params[f'theta+']).unsqueeze(0)
        theta_minus = (self.params[f'theta-']).unsqueeze(0)
        return ((I + theta_minus) * (I <= torch.minimum(-theta_minus, (theta_plus / torch.sqrt(a_plus) -
                theta_minus / torch.sqrt(a_minus)) / (1 / torch.sqrt(a_plus) + 1 / torch.sqrt(a_minus))))) / \
                a_minus + ((I - theta_plus) * (I >= torch.maximum(theta_plus, (theta_plus / torch.sqrt(a_plus) -
                theta_minus / torch.sqrt(a_minus)) / (1 / torch.sqrt(a_plus) + 1 / torch.sqrt( a_minus))))) / a_plus


    ## Computes g(si) term of potential
    def energy_v(self, config, remove_init=False):
        v = config.long()
        E = torch.zeros(config.shape[0], device=self.device)
        for color in range(self.q):
            A = torch.where(v == color, 1, 0).double()
            if remove_init:
                E -= A.dot(self.params['fields'][:, color] - self.params['fields0'][:, color])
            else:
                E -= A.matmul(self.params['fields'][:, color])

        return E

    ## Computes U(h) term of potential
    def energy_h(self, config, remove_init=False):
        if remove_init:
            a_plus = self.params['gamma+'].sub(self.params['0gamma+'])
            a_minus = self.params['gamma-'].sub(self.params['0gamma-'])
            theta_plus = self.params['theta+'].sub(self.params['0theta+'])
            theta_minus = self.params['theta-'].sub(self.params['0theta-'])
        else:
            a_plus = self.params['gamma+']
            a_minus = self.params['gamma-']
            theta_plus = self.params['theta+']
            theta_minus = self.params['theta-']

        # Applies the dReLU activation function
        zero = torch.zeros_like(config, device=self.device)
        config_plus = torch.maximum(config, zero)
        config_minus = torch.maximum(-config, zero)

        return torch.matmul(config_plus.square(), a_plus) / 2 + torch.matmul(config_minus.square(), a_minus) / 2 + torch.matmul(config_plus, theta_plus) + torch.matmul(config_minus, theta_minus)

    ## Random Config of Visible Potts States
    def random_init_config_v(self, custom_size=False, zeros=False):
        if custom_size:
            size = (*custom_size, self.v_num, self.q)
        else:
            size = (self.batch_size, self.v_num, self.q)

        if zeros:
            return torch.zeros(size[:-1], device=self.device)
        else:
            return self.sample_from_inputs_v(torch.zeros(size, device=self.device).flatten(0, -3), beta=0).reshape(size[:-1])


    def random_init_config_h(self, zeros=False, custom_size=False):
        if custom_size:
            size = (*custom_size, self.h_num)
        else:
            size = (self.batch_size, self.h_num)

        if zeros:
            return torch.zeros(size, device=self.device)
        else:
            return torch.randn(size, device=self.device)

    ## Marginal over hidden units
    def logpartition_h(self, inputs, beta=1):
        if beta == 1:
            a_plus = (self.params['gamma+']).unsqueeze(0)
            a_minus = (self.params['gamma-']).unsqueeze(0)
            theta_plus = (self.params['theta+']).unsqueeze(0)
            theta_minus = (self.params['theta-']).unsqueeze(0)
        else:
            theta_plus = (beta * self.params['theta+'] + (1 - beta) * self.params['0theta+']).unsqueeze(0)
            theta_minus = (beta * self.params['theta-'] + (1 - beta) * self.params['0theta-']).unsqueeze(0)
            a_plus = (beta * self.params['gamma+'] + (1 - beta) * self.params['0gamma+']).unsqueeze(0)
            a_minus = (beta * self.params['gamma-'] + (1 - beta) * self.params['0gamma-']).unsqueeze(0)
        return torch.logaddexp(self.log_erf_times_gauss((-inputs + theta_plus) / torch.sqrt(a_plus)) - 0.5 * torch.log(a_plus), self.log_erf_times_gauss((inputs + theta_minus) / torch.sqrt(a_minus)) - 0.5 * torch.log(a_minus)).sum(
                1) + 0.5 * np.log(2 * np.pi) * self.h_num
        # return y

    ## Marginal over visible units
    def logpartition_v(self, inputs, beta=1):
        if beta == 1:
            return torch.logsumexp(self.params['fields'][None, :, :] + inputs, 2).sum(1)
        else:
            return torch.logsumexp((beta * self.params['fields'] + (1 - beta) * self.params['fields0'])[None, :] + beta * inputs, 2).sum(1)

    ## Mean of hidden
    def mean_h(self, psi, beta=1):
        if beta == 1:
            a_plus = (self.params['gamma+']).unsqueeze(0)
            a_minus = (self.params['gamma-']).unsqueeze(0)
            theta_plus = (self.params['theta+']).unsqueeze(0)
            theta_minus = (self.params['theta-']).unsqueeze(0)
        else:
            theta_plus = (beta * self.params['theta+'] + (1 - beta) * self.params['0theta+']).unsqueeze(0)
            theta_minus = (beta * self.params['theta-'] + (1 - beta) * self.params['0theta-']).unsqueeze(0)
            a_plus = (beta * self.params['gamma+'] + (1 - beta) * self.params['0gamma+']).unsqueeze(0)
            a_minus = (beta * self.params['gamma-'] + (1 - beta) * self.params['0gamma-']).unsqueeze(0)
            psi *= beta

        psi_plus = (-psi + theta_plus) / torch.sqrt(a_plus)
        psi_minus = (psi + theta_minus) / torch.sqrt(a_minus)

        etg_plus = self.erf_times_gauss(psi_plus)
        etg_minus = self.erf_times_gauss(psi_minus)

        p_plus = 1 / (1 + (etg_minus / torch.sqrt(a_minus)) / (etg_plus / torch.sqrt(a_plus)))
        nans = torch.isnan(p_plus)
        p_plus[nans] = 1.0 * (torch.abs(psi_plus[nans]) > torch.abs(psi_minus[nans]))
        p_minus = 1 - p_plus

        mean_pos = (-psi_plus + 1 / etg_plus) / torch.sqrt(a_plus)
        mean_neg = (psi_minus - 1 / etg_minus) / torch.sqrt(a_minus)
        return mean_pos * p_plus + mean_neg * p_minus

    ## Mean of visible
    def mean_v(self, psi, beta=1):
        if beta == 1:
            return nn.Softmax(psi + self.params["fields"].unsqueeze(0))
        else:
            return nn.Softmax(beta * psi + self.params["fields0"].unsqueeze(0) + beta * (self.params["fields"].unsqueeze(0) - self.params["fields0"].unsqueeze(0)))

    ## Compute Input for Hidden Layer from Visible Potts
    def compute_output_v(self, visible_data):
        # compute_output of visible potts layer
        vd = visible_data.long()

        # Newest Version also works, fastest version
        indexTensor = vd.unsqueeze(1).unsqueeze(-1).expand(-1, self.h_num, -1, -1)
        expandedweights = self.W.unsqueeze(0).expand(visible_data.shape[0], -1, -1, -1)
        output = torch.gather(expandedweights, 3, indexTensor).squeeze(3).sum(2)

        return output

    ## Compute Input for Visible Layer from Hidden dReLU
    def compute_output_h(self, config):
        return torch.tensordot(config, self.W, ([1], [0]))

    ## Gibbs Sampling of Potts Visbile Layer
    def sample_from_inputs_v(self, psi, beta=1):
        datasize = psi.shape[0]

        if beta == 1:
            cum_probas = psi + self.params['fields'].unsqueeze(0)
        else:
            cum_probas = beta * psi + beta * self.params['fields'].unsqueeze(0) + (1 - beta) * self.params['fields0'].unsqueeze(0)

        cum_probas = self.cumulative_probabilities(cum_probas)

        rng = torch.rand((datasize, self.v_num), dtype=torch.float64, device=self.device)
        low = torch.zeros((datasize, self.v_num), dtype=torch.long, device=self.device)
        middle = torch.zeros((datasize, self.v_num), dtype=torch.long, device=self.device)
        high = torch.zeros((datasize, self.v_num), dtype=torch.long, device=self.device).fill_(self.q)

        in_progress = low < high
        while True in in_progress:
            # Original Method as matrix operation
            middle[in_progress] = torch.floor((low[in_progress] + high[in_progress]) / 2).long()

            middle_probs = torch.gather(cum_probas, 2, middle.unsqueeze(2)).squeeze(2)
            comparisonfull = rng < middle_probs

            gt = torch.logical_and(comparisonfull, in_progress)
            lt = torch.logical_and(~comparisonfull, in_progress)
            high[gt] = middle[gt]
            low[lt] = torch.add(middle[lt], 1)

            in_progress = low < high

        return high

    ## Gibbs Sampling of dReLU hidden layer
    def sample_from_inputs_h(self, psi, nancheck=False, beta=1):
        if beta == 1:
            a_plus = self.params['gamma+'].unsqueeze(0)
            a_minus = self.params['gamma-'].unsqueeze(0)
            theta_plus = self.params['theta+'].unsqueeze(0)
            theta_minus = self.params['theta-'].unsqueeze(0)
        else:
            theta_plus = (beta * self.params['theta+'] + (1 - beta) * self.params['0theta+']).unsqueeze(0)
            theta_minus = (beta * self.params['theta-'] + (1 - beta) * self.params['0theta-']).unsqueeze(0)
            a_plus = (beta * self.params['gamma+'] + (1 - beta) * self.params['0gamma+']).unsqueeze(0)
            a_minus = (beta * self.params['gamma-'] + (1 - beta) * self.params['0gamma-']).unsqueeze(0)
            psi *= beta

        if nancheck:
            nans = torch.isnan(psi)
            if nans.max():
                nan_unit = torch.nonzero(nans.max(0))[0]
                print('NAN IN INPUT')
                print('Hidden units', nan_unit)

        psi_plus = (-psi).add(theta_plus).div(torch.sqrt(a_plus))
        psi_minus = psi.add(theta_minus).div(torch.sqrt(a_minus))

        etg_plus = self.erf_times_gauss(psi_plus)  # Z+ * sqrt(a+)
        etg_minus = self.erf_times_gauss(psi_minus)  # Z- * sqrt(a-)

        p_plus = 1 / (1 + (etg_minus / torch.sqrt(a_minus)) / (etg_plus / torch.sqrt(a_plus)))  # p+ 1 / (1 +( (Z-/sqrt(a-))/(Z+/sqrt(a+))))    =   (Z+/(Z++Z-)
        nans = torch.isnan(p_plus)

        if True in nans:
            p_plus[nans] = torch.tensor(1.) * (torch.abs(psi_plus[nans]) > torch.abs(psi_minus[nans]))
        p_minus = 1 - p_plus

        is_pos = torch.rand(psi.shape, device=self.device) < p_plus

        rmax = torch.zeros(p_plus.shape, device=self.device)
        rmin = torch.zeros(p_plus.shape, device=self.device)
        rmin[is_pos] = torch.erf(psi_plus[is_pos].mul(self.invsqrt2))  # Part of Phi(x)
        rmax[is_pos] = 1  # pos values rmax set to one
        rmin[~is_pos] = -1  # neg samples rmin set to -1
        rmax[~is_pos] = torch.erf((-psi_minus[~is_pos]).mul(self.invsqrt2))  # Part of Phi(x)

        h = torch.zeros(psi.shape, dtype=torch.float64, device=self.device)
        tmp = (rmax - rmin > 1e-14)
        h = self.sqrt2 * torch.erfinv(rmin + (rmax - rmin) * torch.rand(h.shape, device=self.device))
        h[is_pos] -= psi_plus[is_pos]
        h[~is_pos] += psi_minus[~is_pos]
        h /= torch.sqrt(is_pos * a_plus + ~is_pos * a_minus)
        h[torch.isinf(h) | torch.isnan(h) | ~tmp] = 0
        return h

    ## Visible Potts Supporting Function
    def cumulative_probabilities(self, X, maxi=1e9):
        max, max_indices = X.max(-1)
        max.unsqueeze_(2)
        X -= max
        X.exp_()
        X[X > maxi] = maxi  # For numerical stability.
        X.cumsum_(-1)
        return X / X[:, :, -1].unsqueeze(2)

    ## Hidden dReLU supporting Function
    def erf_times_gauss(self, X):  # This is the "characteristic" function phi
        m = torch.zeros_like(X, dtype=torch.float64)
        tmp1 = X < -6
        m[tmp1] = 2 * torch.exp(X[tmp1] ** 2 / 2)

        tmp2 = X > 0
        t = 1 / (1 + self.pbis * X[tmp2])
        m[tmp2] = t * (self.a1 + self.a2 * t + self.a3 * t ** 2)

        tmp3 = torch.logical_and(~tmp1, ~tmp2)
        t2 = 1 / (1 - self.pbis * X[tmp3])
        m[tmp3] = -t2 * (self.a1 + self.a2 * t2 + self.a3 * t2 ** 2) + 2 * torch.exp(X[tmp3] ** 2 / 2)
        return m

    ## Hidden dReLU supporting Function
    def log_erf_times_gauss(self, X):
        m = torch.zeros_like(X)
        tmp = X < 4
        m[tmp] = 0.5 * X[tmp] ** 2 + torch.log(1 - torch.erf(X[tmp] / np.sqrt(2))) + self.logsqrtpiover2
        m[~tmp] = - torch.log(X[~tmp]) + torch.log(1 - 1 / X[~tmp] ** 2 + 3 / X[~tmp] ** 4)
        return m

    ###################################################### Sampling Functions
    ## Samples hidden from visible and vice versa, returns newly sampled hidden and visible
    def markov_step(self, v, beta=1):
        # Gibbs Sampler
        h = self.sample_from_inputs_h(self.compute_output_v(v), beta=beta)
        nv = self.sample_from_inputs_v(self.compute_output_h(h), beta=beta)
        return nv, h

    def markov_PT_and_exchange(self, v, h, e):
        for i, beta in zip(torch.arange(self.N_PT), self.betas):
            v[i, :, :], h[i, :, :] = self.markov_step(v[i, :, :], beta=beta)
            e[i, :] = self.energy(v[i, :, :], h[i, :, :])

        if self.record_swaps:
            particle_id = torch.arange(self.N_PT).unsqueeze(1).expand(self.N_PT, v.shape[1])

        betadiff = self.betas[1:] - self.betas[:-1]
        for i in np.arange(self.count_swaps % 2, self.N_PT - 1, 2):
            proba = torch.exp(betadiff[i] * e[i + 1, :] - e[i, :]).minimum(torch.ones_like(e[i, :]))
            swap = torch.rand(proba.shape[0], device=self.device) < proba
            if i > 0:
                v[i:i + 2, swap, :] = torch.flip(v[i - 1: i + 1], [0])[:, swap, :]
                h[i:i + 2, swap, :] = torch.flip(h[i - 1: i + 1], [0])[:, swap, :]
                e[i:i + 2, swap] = torch.flip(e[i - 1: i + 1], [0])[:, swap]
                if self.record_swaps:
                    particle_id[i:i + 2, swap] = torch.flip(particle_id[i - 1: i + 1], [0])[:, swap]
            else:
                v[i:i + 2, swap, :] = torch.flip(v[:i + 1], [0])[:, swap, :]
                h[i:i + 2, swap, :] = torch.flip(h[:i + 1], [0])[:, swap, :]
                e[i:i + 2, swap] = torch.flip(e[:i + 1], [0])[:, swap]
                if self.record_swaps:
                    particle_id[i:i + 2, swap] = torch.flip(particle_id[:i + 1], [0])[:, swap]

            if self.record_acceptance:
                self.acceptance_rates[i] = swap.double().mean()
                self.mav_acceptance_rates[i] = self.mavar_gamma * self.mav_acceptance_rates[i] + self.acceptance_rates[
                    i] * (1 - self.mavar_gamma)

        if self.record_swaps:
            self.particle_id.append(particle_id)

        self.count_swaps += 1
        return v, h, e


    ## Markov Step for Parallel Tempering
    def markov_step_PT(self, v, h, e):
        for i, beta in zip(torch.arange(self.N_PT), self.betas):
            v[i, :, :], h[i, :, :] = self.markov_step(v[i, :, :], beta=beta)
            e[i, :] = self.energy(v[i, :, :], h[i, :, :])
        return v, h, e

    def exchange_step_PT(self, v, h, e, compute_energy=True):
        if compute_energy:
            for i in torch.arange(self.N_PT):
                e[i, :] = self.energy(v[i, :, :], h[i, :, :], remove_init=True)

        if self.record_swaps:
            particle_id = torch.arange(self.N_PT).unsqueeze(1).expand(self.N_PT, v.shape[1])

        betadiff = self.betas[1:] - self.betas[:-1]
        for i in np.arange(self.count_swaps % 2, self.N_PT - 1, 2):
            proba = torch.exp(betadiff[i] * e[i+1, :] - e[i, :]).minimum(torch.ones_like(e[i, :]))
            swap = torch.rand(proba.shape[0], device=self.device) < proba
            if i > 0:
                v[i:i + 2, swap, :] = torch.flip(v[i-1: i+1], [0])[:, swap, :]
                h[i:i + 2, swap, :] = torch.flip(h[i-1: i+1], [0])[:, swap, :]
                e[i:i + 2, swap] = torch.flip(e[i-1: i+1], [0])[:, swap]
                if self.record_swaps:
                    particle_id[i:i + 2, swap] = torch.flip(particle_id[i-1: i+1], [0])[:, swap]
            else:
                v[i:i + 2, swap, :] = torch.flip(v[:i+1], [0])[:, swap, :]
                h[i:i + 2, swap, :] = torch.flip(h[:i+1], [0])[:, swap, :]
                e[i:i + 2, swap] = torch.flip(e[:i+1], [0])[:, swap]
                if self.record_swaps:
                    particle_id[i:i + 2, swap] = torch.flip(particle_id[:i+1], [0])[:, swap]

            if self.record_acceptance:
                self.acceptance_rates[i] = swap.double().mean()
                self.mav_acceptance_rates[i] = self.mavar_gamma * self.mav_acceptance_rates[i] + self.acceptance_rates[i] * (1 - self.mavar_gamma)

        if self.record_swaps:
            self.particle_id.append(particle_id)

        self.count_swaps += 1
        return v, h, e

    ######################################################### Pytorch Lightning Functions
    ## Loads Data to be trained from provided fasta file
    def setup(self, stage=None):
        if type(self.fasta_file) is str:
            self.fasta_file = [self.fasta_file]

        assert type(self.fasta_file) is list
        data_pds = []

        for file in self.fasta_file:
            try:
                if self.worker_num == 0:
                    threads = 1
                else:
                    threads = self.worker_num
                seqs, seq_read_counts, all_chars, q_data = fasta_read(file, self.molecule, drop_duplicates=True, threads=threads)
            except IOError:
                print(f"Provided Fasta File '{file}' Not Found")
                print(f"Current Directory '{os.getcwd()}'")
                exit()

            if q_data != self.q:
                print(
                    f"State Number mismatch! Expected q={self.q}, in dataset q={q_data}. All observed chars: {all_chars}")
                exit(-1)

            if self.weights == "fasta":
                weights = np.asarray(seq_read_counts)
                data = pd.DataFrame(data={'sequence': seqs, 'seq_count': weights})
            else:
                data = pd.DataFrame(data={'sequence': seqs})

            data_pds.append(data)

        all_data = pd.concat(data_pds)
        if self.weights is not None and self.weights != "fasta":
            all_data["seq_count"] = self.weights

        self.training_data, self.validation_data = train_test_split(all_data, test_size=0.2, random_state=self.seed)

    ## Sets Up Optimizer as well as Exponential Weight Decasy
    def configure_optimizers(self):
        optim = self.optimizer(self.parameters(), lr=self.lr, weight_decay=self.wd)
        # Exponential Weight Decay after set amount of epochs (set by decay_after)
        decay_gamma = (self.lrf / self.lr) ** (1 / (self.epochs * (1 - self.decay_after)))
        decay_milestone = math.floor(self.decay_after * self.epochs)
        my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=[decay_milestone], gamma=decay_gamma)
        # my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=decay_gamma)
        return optim

    ## Loads Training Data
    def train_dataloader(self, init_fields=True):
        # Get Correct Weights
        if "seq_count" in self.training_data.columns:
            training_weights = self.training_data["seq_count"].tolist()
        else:
            training_weights = None

        train_reader = Categorical(self.training_data, self.q, weights=training_weights, max_length=self.v_num,
                                    shuffle=False, molecule=self.molecule, device=self.device, weight_multiplier=self.weight_multiplier)

        # initialize fields from data
        if init_fields:
            with torch.no_grad():
                initial_fields = train_reader.field_init()
                self.params['fields'] += initial_fields
                self.params['fields0'] += initial_fields

        # Performance was almost identical whether shuffling or not
        if self.sample_type == "pcd":
            shuffle = False
        else:
            shuffle = True

        return DataLoader(
            train_reader,
            batch_size=self.batch_size,
            num_workers=self.worker_num,  # Set to 0 if debug = True
            pin_memory=self.pin_mem,
            shuffle=True
        )

    def val_dataloader(self):
        # Get Correct Validation weights
        if "seq_count" in self.validation_data.columns:
            validation_weights = self.validation_data["seq_count"].tolist()
        else:
            validation_weights = None

        val_reader = Categorical(self.validation_data, self.q, weights=validation_weights, max_length=self.v_num,
                                    shuffle=False, molecule=self.molecule, device=self.device, weight_multiplier=self.weight_multiplier)

        return DataLoader(
            val_reader,
            batch_size=self.batch_size,
            num_workers=self.worker_num,  # Set to 0 to view tensors while debugging
            pin_memory=self.pin_mem,
            shuffle=False
        )

    ## Calls Corresponding Training Function
    def training_step(self, batch, batch_idx):
        # All other functions use self.W for the weights
        if self.loss_type == "free_energy":
            if self.sample_type == "gibbs":
                return self.training_step_CD_free_energy(batch, batch_idx)
            elif self.sample_type == "pt":
                return self.training_step_PT_free_energy(batch, batch_idx)
            elif self.sample_type == "pcd":
                return self.training_step_PCD_free_energy(batch, batch_idx)
        elif self.loss_type == "energy":
            if self.sample_type == "gibbs":
                return self.training_step_CD_energy(batch, batch_idx)
            elif self.sample_type == "pt":
                print("Energy Loss with Parallel Tempering is currently unsupported")
                exit(1)

    def validation_step(self, batch, batch_idx):
        # Needed for pseudo_likelihood calculation
        self.prep_W()

        seqs, V_pos, seq_weights = batch

        pseudo_likelihood = (self.pseudo_likelihood(V_pos) * seq_weights).sum() / seq_weights.abs().sum()

        batch_out = {
             "val_pseudo_likelihood": pseudo_likelihood.detach()
        }

        self.log("ptl/val_pseudo_likelihood", pseudo_likelihood, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return batch_out

    def validation_epoch_end(self, outputs):
        avg_pl = torch.stack([x['val_pseudo_likelihood'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Validation pseudo_likelihood", avg_pl, self.current_epoch)

    ## On Epoch End Collects Scalar Statistics and Distributions of Parameters for the Tensorboard Logger
    def training_epoch_end(self, outputs):
        # These are detached except for loss
        avg_loss = torch.stack([x['loss'].detach() for x in outputs]).mean()
        avg_dF = torch.stack([x["free_energy_diff"] for x in outputs]).mean()
        field_reg = torch.stack([x["field_reg"] for x in outputs]).mean()
        weight_reg = torch.stack([x["weight_reg"] for x in outputs]).mean()
        # energy_reg = torch.stack([x['log']["energy_reg"] for x in outputs]).mean()
        pseudo_likelihood = torch.stack([x['train_pseudo_likelihood'] for x in outputs]).mean()
        # reconstruction = torch.stack([x['log']['reconstruction_loss'] for x in outputs]).mean()

        self.logger.experiment.add_scalars("All Scalars", {"Loss": avg_loss,
                                                           "CD_Loss": avg_dF,
                                                           "Field Reg": field_reg,
                                                           "Weight Reg": weight_reg,
                                                           "Train_pseudo_likelihood": pseudo_likelihood
                                                           # "Reconstruction Loss": reconstruction,
                                                           }, self.current_epoch)

        self.logger.experiment.add_histogram("Weights", self.W.detach(), self.current_epoch)
        for name, p in self.params.items():
            self.logger.experiment.add_histogram(name, p.detach(), self.current_epoch)

    ## This works but not the exact quantity we want to maximize
    def training_step_CD_energy(self, batch, batch_idx):
        seqs, V_pos, seq_weights = batch
        weights = seq_weights.clone().detach()
        V_pos = V_pos.clone().detach()
        V_neg, h_neg, V_pos, h_pos = self(V_pos)

        energy_pos = (self.energy(V_pos, h_pos) * weights).sum() / weights.sum()# energy of training data
        energy_neg = (self.energy(V_neg, h_neg) * weights).sum() / weights.sum() # energy of gibbs sampled visible states

        cd_loss = energy_pos - energy_neg

        pseudo_likelihood = (self.pseudo_likelihood(V_pos) * weights).sum() / weights.sum()

        reg1 = self.lf / 2 * self.params['fields'].square().sum((0, 1))
        tmp = torch.sum(torch.abs(self.W), (1, 2)).square()
        reg2 = self.l1_2 / (2 * self.q * self.v_num * self.h_num) * tmp.sum()

        loss = cd_loss + reg1 + reg2

        logs = {"loss": loss,
                "train_pseudo_likelihood": pseudo_likelihood.detach(),
                "free_energy_diff": cd_loss.detach(),
                "field_reg": reg1.detach(),
                "weight_reg": reg2.detach()
                }

        self.log("ptl/train_pseudo_likelihood", pseudo_likelihood, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return logs

    def forward_PCD(self, batch_idx):
        # Enforces Zero Sum Gauge on Weights
        self.prep_W()

        # Gibbs sampling with Persistent Contrastive Divergence
        # pytorch lightning handles the device
        fantasy_v = self.chain[batch_idx]  # Last sample that was saved to self.chain variable, initialized in constructor to our data points
        # h_pos = self.sample_from_inputs_h(self.compute_output_v(fantasy_v))
        # with torch.no_grad() # only use last sample for gradient calculation, may be helpful but honestly not the slowest thing rn
        for _ in range(self.mc_moves - 1):
            fantasy_v, fantasy_h = self.markov_step(fantasy_v)

        V_neg, fantasy_h = self.markov_step(fantasy_v)
        h_neg = self.sample_from_inputs_h(self.compute_output_v(V_neg))

        self.chain[batch_idx] = fantasy_v.detach()

        return V_neg, h_neg

    def training_step_PCD_free_energy(self, batch, batch_idx):
        # seqs, V_pos, one_hot, seq_weights = batch
        seqs, V_pos, seq_weights = batch
        weights = seq_weights.clone()

        if self.current_epoch == 0 and batch_idx == 0:
            self.chain = [V_pos.detach()]
        elif self.current_epoch == 0:
            self.chain.append(V_pos.detach())

        V_neg, h_neg = self.forward_PCD(batch_idx)

        # psuedo likelihood actually minimized, loss sits around 0 but does it's own thing
        F_v = (self.free_energy(V_pos) * weights).sum() / weights.sum()  # free energy of training data
        F_vp = (self.free_energy(V_neg) * weights).sum() / weights.sum()  # free energy of gibbs sampled visible states
        cd_loss = F_v - F_vp  # Should Give same gradient as Tubiana Implementation minus the batch norm on the hidden unit activations

        pseudo_likelihood = (self.pseudo_likelihood(V_pos) * weights).sum() / weights.sum()

        # Regularization Terms
        reg1 = self.lf / 2 * self.params['fields'].square().sum((0, 1))
        tmp = torch.sum(torch.abs(self.W), (1, 2)).square()
        reg2 = self.l1_2 / (2 * self.q * self.v_num) * tmp.sum()

        loss = cd_loss + reg1 + reg2

        logs = {"loss": loss,
                "train_pseudo_likelihood": pseudo_likelihood.detach(),
                "free_energy_diff": cd_loss.detach(),
                "field_reg": reg1.detach(),
                "weight_reg": reg2.detach(),
                }

        self.log("ptl/train_pseudo_likelihood", pseudo_likelihood, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return logs

    def training_step_PT_free_energy(self, batch, batch_idx):
        seqs, V_pos, seq_weights = batch
        V_pos = V_pos.clone().detach()
        V_neg, h_neg, V_pos, h_pos = self.forward(V_pos)
        weights = seq_weights.clone().detach()

        # psuedo likelihood actually minimized, loss sits around 0 but does it's own thing
        F_v = (self.free_energy(V_pos) * weights).sum() / weights.sum()  # free energy of training data
        F_vp = (self.free_energy(V_neg) * weights).sum() / weights.sum()  # free energy of gibbs sampled visible states
        cd_loss = F_v - F_vp  # Should Give same gradient as Tubiana Implementation minus the batch norm on the hidden unit act

        pseudo_likelihood = (self.pseudo_likelihood(V_pos).clone().detach() * weights).sum() / weights.sum()

        reg1 = self.lf / 2 * self.params['fields'].square().sum((0, 1))
        tmp = torch.sum(torch.abs(self.W), (1, 2)).square()
        reg2 = self.l1_2 / (2 * self.q * self.v_num * self.h_num) * tmp.sum()

        loss = cd_loss + reg1 + reg2

        logs = {"loss": loss.detach(),
                "train_pseudo_likelihood": pseudo_likelihood.detach(),
                "free_energy_diff": cd_loss.detach(),
                "field_reg": reg1.detach(),
                "weight_reg": reg2.detach()
                }

        self.log("ptl/train_pseudo_likelihood", pseudo_likelihood, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return logs

    def training_step_CD_free_energy(self, batch, batch_idx):
        # seqs, V_pos, one_hot, seq_weights = batch
        seqs, V_pos, seq_weights = batch
        weights = seq_weights.clone()
        V_neg, h_neg, V_pos, h_pos = self(V_pos)

        # psuedo likelihood actually minimized, loss sits around 0 but does it's own thing
        F_v = (self.free_energy(V_pos) * weights).sum() / weights.abs().sum()  # free energy of training data
        F_vp = (self.free_energy(V_neg) * weights.abs()).sum() / weights.abs().sum()  # free energy of gibbs sampled visible states
        cd_loss = F_v - F_vp  # Should Give same gradient as Tubiana Implementation minus the batch norm on the hidden unit activations

        # Reconstruction Loss, Did not work very well
        # V_neg_oh = F.one_hot(V_neg, num_classes=self.q)
        # reconstruction_loss = self.reconstruction_loss(V_neg_oh.double(), one_hot.double())*self.q  # not ideal Loss counts matching zeros as t

        pseudo_likelihood = (self.pseudo_likelihood(V_pos) * weights).sum() / weights.sum()

        # Regularization Terms
        reg1 = self.lf/2 * self.params['fields'].square().sum((0, 1))
        tmp = torch.sum(torch.abs(self.W), (1, 2)).square()
        reg2 = self.l1_2 / (2 * self.q * self.v_num) * tmp.sum()

        loss = cd_loss + reg1 + reg2 #  + reconstruction_loss

        logs = {"loss": loss,
                "train_pseudo_likelihood": pseudo_likelihood.detach(),
                "free_energy_diff": cd_loss.detach(),
                "field_reg": reg1.detach(),
                "weight_reg": reg2.detach(),
                # "reconstruction_loss": reconstruction_loss.detach()
                }

        self.log("ptl/train_pseudo_likelihood", pseudo_likelihood, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return logs

    def forward(self, V_pos):
        # Enforces Zero Sum Gauge on Weights
        self.prep_W()
        if self.sample_type == "gibbs":
            # Gibbs sampling
            # pytorch lightning handles the device
            fantasy_v = V_pos.clone()
            h_pos = self.sample_from_inputs_h(self.compute_output_v(fantasy_v))
            # with torch.no_grad() # only use last sample for gradient calculation, may be helpful but honestly not the slowest thing rn
            for _ in range(self.mc_moves-1):
                fantasy_v, fantasy_h = self.markov_step(fantasy_v)

            V_neg, fantasy_h = self.markov_step(fantasy_v)
            h_neg = self.sample_from_inputs_h(self.compute_output_v(V_neg))

            return V_neg, h_neg, V_pos, h_pos

        elif self.sample_type == "pt":
            # Parallel Tempering
            h_pos = self.sample_from_inputs_h(self.compute_output_v(V_pos))

            n_chains = V_pos.shape[0]
            fantasy_v = self.random_init_config_v(custom_size=(self.N_PT, n_chains))
            fantasy_h = self.random_init_config_h(custom_size=(self.N_PT, n_chains))
            fantasy_E = self.energy_PT(fantasy_v, fantasy_h)

            for _ in range(self.mc_moves):
                fantasy_v, fantasy_h, fantasy_E = self.markov_PT_and_exchange(fantasy_v, fantasy_h, fantasy_E)
                self.update_betas()

            V_neg = fantasy_v[0, :, :]
            h_neg = fantasy_h[0, :, :]

            return V_neg, h_neg, V_pos, h_pos

    # X must be a pandas dataframe with the sequences in string format under the column 'sequence'
    # Returns the likelihood for each sequence in an array
    def predict(self, X):
        # X must be a pandas dataframe
        # Needs to be set
        self.prep_W()
        reader = Categorical(X, self.q, weights=None, max_length=self.v_num, shuffle=False, molecule=self.molecule, device=self.device)
        data_loader = torch.utils.data.DataLoader(
            reader,
            batch_size=self.batch_size,
            num_workers=self.worker_num,  # Set to 0 if debug = True
            pin_memory=self.pin_mem,
            shuffle=False
        )
        with torch.no_grad():
            likelihood = []
            for i, batch in enumerate(data_loader):
                seqs, V_pos, seq_weights = batch
                likelihood += self.likelihood(V_pos).detach().tolist()

        return X.sequence.tolist(), likelihood

    def saliency_map(self, X):
        reader = Categorical(X, self.q, weights=None, max_length=self.v_num, shuffle=False, molecule=self.molecule, device=self.device)
        data_loader = torch.utils.data.DataLoader(
            reader,
            batch_size=self.batch_size,
            num_workers=self.worker_num,  # Set to 0 if debug = True
            pin_memory=self.pin_mem,
            shuffle=False
        )
        saliency_maps = []
        for i, batch in enumerate(data_loader):
            seqs, V_pos, seq_weights = batch
            variable_pos = Variable(V_pos, requires_grad=True)
            V_neg, h_neg, v_pos_out, h_pos = self(variable_pos)
            weights = seq_weights
            F_v = (self.free_energy(variable_pos) * weights).sum() / weights.sum()  # free energy of training data
            F_vp = (self.free_energy(V_neg) * weights).sum() / weights.sum()  # free energy of gibbs sampled visible states
            cd_loss = F_v - F_vp  # Should Give same gradient as Tubiana Implementation minus the batch norm on the hidden unit activations
            # free_energy = F_v.detach()

            # Regularization Terms
            reg1 = self.lf / 2 * self.params['fields'].square().sum((0, 1))
            reg2 = torch.zeros((1,), device=self.device)
            reg3 = torch.zeros((1,), device=self.device)
            for iid, i in enumerate(self.hidden_convolution_keys):
                W_shape = self.convolution_topology[i]["weight_dims"]  # (h_num,  input_channels, kernel0, kernel1)
                x = torch.sum(torch.abs(self.params[f"{i}_W"]), (3, 2, 1)).square()
                reg2 += x.sum(0) * self.l1_2 / (2 * W_shape[1] * W_shape[2] * W_shape[3])
                # Size of Convolution Filters weight_size = (h_num, input_channels, kernel[0], kernel[1])
                reg3 += self.ld / ((self.params[f"{i}_W"].abs() - self.params[f"{i}_W"].squeeze(1).abs()).abs().sum((1, 2, 3)).mean() + 1)

            loss = cd_loss + reg1 + reg2 + reg3
            loss.backward()

            saliency_maps.append(variable_pos.grad.data.detach())

        return torch.cat(saliency_maps, dim=0)

    # Don't use this
    def predict_psuedo(self, X):
        self.prep_W()
        reader = Categorical(X, self.q, weights=None, max_length=self.v_num, shuffle=False, molecule=self.molecule, device=self.device)
        data_loader = torch.utils.data.DataLoader(
            reader,
            batch_size=self.batch_size,
            num_workers=self.worker_num,  # Set to 0 if debug = True
            pin_memory=self.pin_mem,
            shuffle=False
        )
        with torch.no_grad():
            likelihood = []
            for i, batch in enumerate(data_loader):
                seqs, V_pos, seq_weights = batch
                likelihood += self.pseudo_likelihood(V_pos).detach().tolist()

        return X.sequence.tolist(), likelihood

    # Return param as a numpy array
    def get_param(self, param_name):
        if param_name == "W":
            W_raw = self.params['W_raw'].clone()
            tensor = W_raw - W_raw.sum(-1).unsqueeze(2) / self.q
            return tensor.detach().numpy()
        else:
            try:
                tensor = self.params[param_name].clone()
                return tensor.detach.numpy()
            except KeyError:
                print(f"Key {param_name} not found")
                exit()

    def update_betas(self, beta=1):
        with torch.no_grad():
            if self.acceptance_rates.mean() > 0:
                self.stiffness = torch.maximum(1 - (self.mav_acceptance_rates / self.mav_acceptance_rates.mean()), torch.zeros_like(self.mav_acceptance_rates)) + 1e-4 * torch.ones(self.N_PT - 1)
                diag = self.stiffness[0:-1] + self.stiffness[1:]
                if self.N_PT > 3:
                    offdiag_g = -self.stiffness[1:-1]
                    offdiag_d = -self.stiffness[1:-1]
                    M = torch.diag(offdiag_g, -1) + torch.diag(diag, 0) + torch.diag(offdiag_d, 1)
                else:
                    M = torch.diag(diag, 0)
                B = torch.zeros(self.N_PT - 2)
                B[0] = self.stiffness[0] * beta
                self.betas[1:-1] = self.betas[1:-1] * (1 - self.update_betas_lr) + self.update_betas_lr * torch.linalg.solve(M, B)
                self.update_betas_lr *= self.update_betas_lr_decay

    def AIS(self, M=10, n_betas=10000, batches=None, verbose=0, beta_type='adaptive'):
        with torch.no_grad():
            if beta_type == 'linear':
                betas = torch.arange(n_betas) / torch.tensor(n_betas - 1, dtype=torch.float64)
            elif beta_type == 'root':
                betas = torch.sqrt(torch.arange(n_betas)) / torch.tensor(n_betas - 1, dtype=torch.float64)
            elif beta_type == 'adaptive':
                Nthermalize = 200
                Nchains = 20
                N_PT = 11
                self.adaptive_PT_lr = 0.05
                self.adaptive_PT_decay = True
                self.adaptive_PT_lr_decay = 10 ** (-1 / float(Nthermalize))
                if verbose:
                    t = time.time()
                    print('Learning betas...')
                self.gen_data(N_PT=N_PT, Nchains=Nchains, Lchains=1, Nthermalize=Nthermalize, update_betas=True)
                if verbose:
                    print('Elapsed time: %s, Acceptance rates: %s' % (time.time() - t, self.mav_acceptance_rates))
                betas = []
                sparse_betas = self.betas.flip(0)
                for i in range(N_PT - 1):
                    betas += list(sparse_betas[i] + (sparse_betas[i + 1] - sparse_betas[i]) * torch.arange(n_betas / (N_PT-1)) / (n_betas / (N_PT - 1) - 1))
                betas = torch.tensor(betas)
                # if verbose:
                # import matplotlib.pyplot as plt
                # plt.plot(betas); plt.title('Interpolating temperatures');plt.show()

            # Initialization.
            log_weights = torch.zeros(M)
            config = self.gen_data(Nchains=M, Lchains=1, Nthermalize=0, beta=0)

            log_Z_init = torch.zeros(1)
            log_Z_init += self.logpartition_h(torch.zeros((1, self.h_num), device=self.device), beta=0)
            log_Z_init += self.logpartition_v(torch.zeros((1, self.v_num, self.q), device=self.device), beta=0)

            if verbose:
                print(f'Initial evaluation: log(Z) = {log_Z_init.data}')

            for i in range(1, n_betas):
                if verbose:
                    if (i % 2000 == 0):
                        print(f'Iteration {i}, beta: {betas[i]}')
                        print('Current evaluation: log(Z)= %s +- %s' % ((log_Z_init + log_weights).mean(), (log_Z_init + log_weights).std() / np.sqrt(M)))

                config[0], config[1] = self.markov_step(config[0])
                energy = self.energy(config[0], config[1])
                log_weights += -(betas[i] - betas[i - 1]) * energy
            self.log_Z_AIS = (log_Z_init + log_weights).mean()
            self.log_Z_AIS_std = (log_Z_init + log_weights).std() / np.sqrt(M)
            if verbose:
                print('Final evaluation: log(Z)= %s +- %s' % (self.log_Z_AIS, self.log_Z_AIS_std))
            return self.log_Z_AIS, self.log_Z_AIS_std

    def likelihood(self, data, recompute_Z=False):
        if (not hasattr(self, 'log_Z_AIS')) | recompute_Z:
            self.AIS()
        return -self.free_energy(data) - self.log_Z_AIS

    def cgf_from_inputs_h(self, I):
        with torch.no_grad():
            B = I.shape[0]
            out = torch.zeros(I.shape, device=self.device)
            sqrt_gamma_plus = torch.sqrt(self.params["gamma+"]).expand(B, -1)
            sqrt_gamma_minus = torch.sqrt(self.params["gamma-"]).expand(B, -1)
            log_gamma_plus = torch.log(self.params["gamma+"]).expand(B, -1)
            log_gamma_minus = torch.log(self.params["gamma-"]).expand(B, -1)

            Z_plus = -self.log_erf_times_gauss((-I + self.params['theta+'].expand(B, -1)) / sqrt_gamma_plus) - 0.5 * log_gamma_plus
            Z_minus = self.log_erf_times_gauss((I + self.params['theta-'].expand(B, -1)) / sqrt_gamma_minus) - 0.5 * log_gamma_minus
            map = Z_plus > Z_minus
            out[map] = Z_plus[map] + torch.log(1 + torch.exp(Z_minus[map] - Z_plus[map]))
            out[~map] = Z_minus[~map] + torch.log(1 + torch.exp(Z_plus[~map] - Z_minus[~map]))
            return out

    def gen_data(self, Nchains=10, Lchains=100, Nthermalize=0, Nstep=1, N_PT=1, config_init=[], beta=1, batches=None, reshape=True, record_replica=False, record_acceptance=None, update_betas=None, record_swaps=False):
        """
        Generate Monte Carlo samples from the RBM. Starting from random initial conditions, Gibbs updates are performed to sample from equilibrium.
        Inputs :
            Nchains (10): Number of Markov chains
            Lchains (100): Length of each chain
            Nthermalize (0): Number of Gibbs sampling steps to perform before the first sample of a chain.
            Nstep (1): Number of Gibbs sampling steps between each sample of a chain
            N_PT (1): Number of Monte Carlo Exchange replicas to use. This==useful if the mixing rate==slow. Watch self.acceptance_rates_g to check that it==useful (acceptance rates about 0==useless)
            batches (10): Number of batches. Must divide Nchains. higher==better for speed (more vectorized) but more RAM consuming.
            reshape (True): If True, the output==(Nchains x Lchains, n_visibles/ n_hiddens) (chains mixed). Else, the output==(Nchains, Lchains, n_visibles/ n_hiddens)
            config_init ([]). If not [], a Nchains X n_visibles numpy array that specifies initial conditions for the Markov chains.
            beta (1): The inverse temperature of the model.
        """
        with torch.no_grad():

            if batches == None:
                batches = Nchains
            n_iter = int(Nchains / batches)
            Ndata = Lchains * batches
            if record_replica:
                reshape = False

            if (N_PT > 1):
                if record_acceptance == None:
                    record_acceptance = True

                if update_betas == None:
                    update_betas = False

                if record_acceptance:
                    self.mavar_gamma = 0.95

                if update_betas:
                    record_acceptance = True
                    self.update_betas_lr = 0.1
                    self.update_betas_lr_decay = 1
            else:
                record_acceptance = False
                update_betas = False

            if (N_PT > 1) and record_replica:
                visible_data = torch.zeros((Nchains, N_PT, Lchains, self.v_num), dtype=torch.long)
                hidden_data = torch.zeros((Nchains, N_PT, Lchains, self.h_num), dtype=torch.float64)
                data = [visible_data, hidden_data]
            else:
                visible_data = torch.zeros((Nchains, Lchains, self.v_num), dtype=torch.long)
                hidden_data = torch.zeros((Nchains, Lchains, self.h_num), dtype=torch.float64)
                data = [visible_data, hidden_data]

            if config_init is not []:
                if type(config_init) == torch.tensor:
                    h_layer = self.random_init_config_h()
                    config_init = [config_init, h_layer]

            for i in range(n_iter):
                if config_init == []:
                    config = self._gen_data(Nthermalize, Ndata, Nstep, N_PT=N_PT, batches=batches, reshape=False, beta=beta,
                                            record_replica=record_replica, record_acceptance=record_acceptance, update_betas=update_betas, record_swaps=record_swaps)
                else:
                    config_init = [config_init[0][batches * i:batches * (i + 1)],  config_init[1][batches * i:batches * (i + 1)]]
                    config = self._gen_data(Nthermalize, Ndata, Nstep, N_PT=N_PT, batches=batches, reshape=False, beta=beta,
                                            record_replica=record_replica, config_init=config_init, record_acceptance=record_acceptance,
                                            update_betas=update_betas, record_swaps=record_swaps)

                if (N_PT > 1) & record_replica:
                    data[0][batches * i:batches * (i + 1), :, :, :] = torch.swapaxes(config[0], 0, 2).clone()
                    data[1][batches * i:batches * (i + 1), :, :, :] = torch.swapaxes(config[1], 0, 2).clone()
                else:
                    data[0][batches * i:batches * (i + 1), :, :] = torch.swapaxes(config[0], 0, 1).clone()
                    data[1][batches * i:batches * (i + 1), :, :] = torch.swapaxes(config[1], 0, 1).clone()

            if reshape:
                return [data[0].reshape([Nchains * Lchains, self.v_num]), data[1].reshape([Nchains * Lchains, self.h_num])]
            else:
                return data

    def _gen_data(self, Nthermalize, Ndata, Nstep, N_PT=1, batches=1, reshape=True, config_init=[], beta=1, record_replica=False, record_acceptance=True, update_betas=False, record_swaps=False):
        with torch.no_grad():
            self.N_PT = N_PT
            if self.N_PT > 1:
                if update_betas or len(self.betas) != N_PT:
                    self.betas = torch.flip(torch.arange(N_PT) / (N_PT - 1) * beta, [0])

                self.acceptance_rates = torch.zeros(N_PT - 1)
                self.mav_acceptance_rates = torch.zeros(N_PT - 1)

            self.count_swaps = 0
            self.record_swaps = record_swaps

            if self.record_swaps:
                self.particle_id = [torch.arange(N_PT)[:, None].repeat(batches, dim=1)]

            Ndata /= batches
            Ndata = int(Ndata)
            if N_PT > 1:
                config = [self.random_init_config_v(custom_size=(N_PT, batches)), self.random_init_config_h(custom_size=(N_PT, batches))]
                if config_init != []:
                    config[0] = config_init[0]
                    config[1] = config_init[1]
                energy = torch.zeros([N_PT, batches])
            else:
                if config_init != []:
                    config = config_init
                else:
                    config = [self.random_init_config_v(custom_size=(batches,)), self.random_init_config_h(custom_size=(batches,))]

            for _ in range(Nthermalize):
                if N_PT > 1:
                    energy = self.energy_PT(config[0], config[1])
                    config[0], config[1], energy = self.markov_PT_and_exchange(config[0], config[1], energy)
                    if update_betas:
                        self.update_betas(beta=beta)
                else:
                    config[0], config[1] = self.markov_step(config[0], beta=beta)

            if N_PT > 1:
                if Ndata > 1:
                    if record_replica:
                        data_gen_v = self.random_init_config_v(custom_size=(Ndata, N_PT, batches), zeros=True)
                        data_gen_h = self.random_init_config_h(custom_size=(Ndata, N_PT, batches), zeros=True)
                        data_gen_v[0] = config[0].clone().unsqueeze(0)
                        data_gen_h[0] = config[1].clone().unsqueeze(0)
                    else:
                        data_gen_v = self.random_init_config_v(custom_size=(Ndata, N_PT), zeros=True)
                        data_gen_h = self.random_init_config_h(custom_size=(Ndata, N_PT), zeros=True)
                        data_gen_v[0] = config[0][0].clone().unsqueeze(0)
                        data_gen_h[0] = config[1][0].clone().unsqueeze(0)
            else:
                data_gen_v = self.random_init_config_v(custom_size=(Ndata, batches), zeros=True)
                data_gen_h = self.random_init_config_h(custom_size=(Ndata, batches), zeros=True)
                data_gen_v[0] = config[0].clone().unsqueeze(0)
                data_gen_h[0] = config[1].clone().unsqueeze(0)

            for n in range(Ndata - 1):
                for _ in range(Nstep):
                    if N_PT > 1:
                        energy = self.energy_PT(config[0], config[1])
                        config[0], config[1], energy = self.markov_PT_and_exchange(config[0], config[1], energy)
                        if update_betas:
                            self.update_betas(beta=beta)
                    else:
                        config[0], config[1] = self.markov_step(config[0], beta=beta)

                if N_PT > 1 and Ndata > 1:
                    if record_replica:
                        data_gen_v[n+1] = config[0].clone()
                        data_gen_h[n+1] = config[1].clone()
                    else:
                        data_gen_v[n+1] = config[0][0].clone()
                        data_gen_h[n+1] = config[1][0].clone()
                else:
                    data_gen_v[n+1] = config[0].clone()
                    data_gen_h[n+1] = config[1].clone()

            if Ndata > 1:
                data = [data_gen_v, data_gen_h]

            if self.record_swaps:
                print('cleaning particle trajectories')
                positions = torch.tensor(self.particle_id)
                invert = torch.zeros([batches, Ndata, N_PT])
                for b in range(batches):
                    for i in range(Ndata):
                        for k in range(N_PT):
                            invert[b, i, k] = torch.nonzero(positions[i, :, b] == k)[0]
                self.particle_id = invert
                self.last_at_zero = torch.zeros([batches, Ndata, N_PT])
                for b in range(batches):
                    for i in range(Ndata):
                        for k in range(N_PT):
                            tmp = torch.nonzero(self.particle_id[b, :i, k] == 0)[0]
                            if len(tmp) > 0:
                                self.last_at_zero[b, i, k] = i - 1 - tmp.max()
                            else:
                                self.last_at_zero[b, i, k] = 1000
                self.last_at_zero[:, 0, 0] = 0

                self.trip_duration = torch.zeros([batches, Ndata])
                for b in range(batches):
                    for i in range(Ndata):
                        self.trip_duration[b, i] = self.last_at_zero[b, i, np.nonzero(invert[b, i, :] == 9)[0]]

            if reshape:
                data[0] = data[0].reshape([Ndata * batches, self.v_num])
                data[1] = data[1].reshape([Ndata * batches, self.h_num])
            else:
                data[0] = data[0]
                data[1] = data[1]

            return data




if __name__ == '__main__':
    # data_file = '../invivo/sham2_ipsi_c1.fasta'  # cpu is faster
    # large_data_file = '../invivo/chronic1_spleen_c1.fasta' # gpu is faster
    lattice_data = './lattice_proteins_verification/Lattice_Proteins_MSA.fasta'
    # b3_c1 = "../pig/b3_c1.fasta"
    # bivalent_data = "./bivalent_aptamers_verification/s100_8th.fasta"

    config = rbm_configs.lattice_default_config
    # Edit config for dataset specific hyperparameters
    config["fasta_file"] = lattice_data
    config["sequence_weights"] = None
    config["epochs"] = 50

    # Training Code
    # rbm = RBM(config, debug=True)
    # logger = TensorBoardLogger('./lattice_proteins_verification/', name="lattice_rbm")
    # plt = Trainer(max_epochs=config['epochs'], logger=logger, gpus=1)  # gpus=1,
    # plt.fit(rbm)

    checkp = "./lattice_proteins_verification/lattice_rbm/version_8/checkpoints/epoch=49-step=99.ckpt"
    rbm = RBM.load_from_checkpoint(checkp)

    # results = gen_data_lowT(rbm, which="marginal")
    results = gen_data_zeroT(rbm, which="joint")
    visible, hiddens = results

    E = rbm.energy(visible, hiddens)
    print("E", E.shape)
