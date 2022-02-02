import time

import numpy
import pandas as pd
import pytorch_lightning.profiler
import seaborn

from pytorch_lightning.utilities.cloud_io import load as pl_load
import argparse
import json
import pandas as pd

import rbm_utils
import math

import numpy as np
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # just for confusion matrix generation

from ray import tune
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW, Adagrad  # Supported Optimizers
import multiprocessing # Just to set the worker number



from rbm_utils import aadict, dnadict, rnadict


class RBMCaterogical(Dataset):
    # Takes in pd dataframe with sequences and weights of sequences (key: "sequences", weights: "sequence_count")
    # Also used to calculate the independent fields for parameter fields initialization
    def __init__(self, dataset, weights=None, max_length=20, shuffle=True, base_to_id='protein', device='cpu', one_hot=False):

        # Drop Duplicates/ Reset Index from most likely shuffled sequences
        # self.dataset = dataset.reset_index(drop=True).drop_duplicates("sequence")
        self.dataset = dataset.reset_index(drop=True)

        self.shuffle = shuffle
        self.on_epoch_end()

        # dictionaries mapping One letter code to integer for all macro molecule types
        if base_to_id == 'protein':
            self.base_to_id = aadict
            self.n_bases = 21
        elif base_to_id == 'dna':
            self.base_to_id = dnadict
            self.n_bases = 5
        elif base_to_id == 'rna':
            self.base_to_id = rnadict
            self.n_bases = 5

        self.device = device # Makes sure everything is on correct device

        self.max_length = max_length # Number of Visible Nodes
        self.oh = one_hot
        # self.train_labels = self.dataset.binary.to_numpy()
        self.total = len(self.dataset.index)
        self.seq_data = self.dataset.sequence.to_numpy()
        self.train_data = self.categorical(self.seq_data)
        if self.oh:
            self.train_oh = self.one_hot(self.train_data)

        if weights is not None:
            if len(self.train_data) == len(weights):
                print("Provided Weights are not the correct length")
                exit(1)
            self.train_weights = np.asarray(weights)
            self.train_weights /= np.linalg.norm(self.train_weights)
        else:
            # all equally weighted
            self.train_weights = np.asarray([1. for x in range(self.total)])


    def __getitem__(self, index):

        self.count += 1
        if (self.count % self.dataset.shape[0] == 0):
            self.on_epoch_end()

        seq = self.seq_data[index]
        cat_seq = self.train_data[index]
        weight = self.train_weights[index]

        if self.oh:
            one_hot = self.train_oh[index]
            return seq, cat_seq, one_hot, weight
        else:
            return seq, cat_seq, weight

    def categorical(self, seq_dataset):
        cat_np = torch.zeros((seq_dataset.shape[0], self.max_length), dtype=torch.long)
        for i in range(seq_dataset.shape[0]):
            seq = seq_dataset[i]
            for n, base in enumerate(seq):
               cat_np[i, n] = self.base_to_id[base]
        return cat_np

    def one_hot(self, cat_dataset):
        one_hot_vector = F.one_hot(cat_dataset, num_classes=self.n_bases)
        return one_hot_vector

    # verified to work exactly as done in tubiana's implementation
    def field_init(self):
        out = torch.zeros((self.max_length, self.n_bases), device=self.device)
        for b in range(self.total):
            seqoi, weight = self.train_data[b], self.train_weights[b]
            for n in range(self.max_length):
                out[n, seqoi[n]] += weight
        out.div_(self.total)

        # invert softmax
        eps = 1e-6
        fields = torch.log((1 - eps) * out + eps / self.n_bases)
        fields -= fields.sum(1).unsqueeze(1) / self.n_bases
        return fields

    def __len__(self):
        return self.train_data.shape[0]

    def on_epoch_end(self):
        self.count = 0
        if self.shuffle:
            self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)





class RBM(pl.LightningModule):
    def __init__(self, config, debug=False):
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

        # Only use for hyperparam optimization

        # Sets workers for the train and validation dataloaders
        if debug:
            # Enables inspection of the torch tensors using breakpoints
            self.worker_num = 0
        else:
            self.worker_num = multiprocessing.cpu_count()


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


        if weights == None:
            self.weights = None
        elif type(weights) == "str":
            if weights == "fasta": # Assumes weights are in fasta file
                self.weights = "fasta"
        elif type(weights) == torch.tensor:
            self.weights = weights.numpy()
        elif type(weights) == np.array:
            self.weights = weights
        else:
            print("Provided Weights Not Supported, Must be None, a numpy array, torch tensor, or 'fasta'")
            exit(1)


        # loss types are 'energy' and 'free_energy' for now, controls the loss function primarily
        # sample types control whether gibbs sampling from the data points or parallel tempering from random configs are used
        # Switches How the training of the RBM is performed

        if loss_type not in ['energy', 'free_energy']:
            print(f"Loss Type {loss_type} not supported")
            exit(1)
        if sample_type not in ['gibbs', 'pt']:
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
        torch.set_default_dtype(torch.float64)  # Double Precision

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

    ############################################################# RBM Functions
    ## Compute Psudeo likelihood of given visible config
    def psuedolikelihood(self, v):
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
    def free_energy_h(self,h):
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
    def random_init_config_v(self, batch_size=None):
        if batch_size is not None:
            return self.sample_from_inputs_v(torch.zeros((batch_size, self.v_num, self.q), device=self.device), beta=0)
        else:
            return self.sample_from_inputs_v(torch.zeros((self.batch_size, self.v_num, self.q), device=self.device), beta=0)

    ## Random Config of Hidden dReLU States
    def random_init_config_h(self, batch_size=None):
        if batch_size is not None:
            return torch.randn((batch_size, self.h_num), device=self.device)
        else:
            return torch.randn((self.batch_size, self.h_num), device=self.device)

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

    # Looking for better Interpretation of weights with potential
    def energy_per_state(self):
        # inputs 21 x v_num
        inputs = torch.arange(self.q).unsqueeze(1).expand(-1, self.v_num)


        indexTensor = inputs.unsqueeze(1).unsqueeze(-1).expand(-1, self.h_num, -1, -1)
        expandedweights = self.W.unsqueeze(0).expand(inputs.shape[0], -1, -1, -1)
        output = torch.gather(expandedweights, 3, indexTensor).squeeze(3)
        out = torch.swapaxes(output, 1, 2)
        energy = torch.zeros((self.q, self.v_num, self.h_num))
        for i in range(self.q):
            for j in range(self.v_num):
                energy[i, j, :] = self.logpartition_h(out[i, j, :])

        # Iu_flat = output.reshape((self.q*self.h_num, self.v_num))
        # Iu = self.compute_output_v(inputs)

        e_h = F.normalize(energy, dim=0)
        view = torch.swapaxes(e_h, 0, 2)

        W = self.get_param("W")

        rbm_utils.Sequence_logo_all(W, name="allweights" + '.pdf', nrows=5, ncols=1, figsize=(10,5) ,ticks_every=10,ticks_labels_size=10,title_size=12, dpi=400, molecule="protein")
        rbm_utils.Sequence_logo_all(view.detach(), name="energything" + '.pdf', nrows=5, ncols=1, figsize=(10,5) ,ticks_every=10,ticks_labels_size=10,title_size=12, dpi=400, molecule="protein")

    ## Marginal over visible units
    def logpartition_v(self, inputs, beta=1):
        if beta == 1:
            return torch.logsumexp(self.params['fields'][None, :, :] + inputs, 2).sum(1)
        else:
            return torch.logsumexp((beta * self.params['fields'] + (1 - beta) * self.params['fields0'])[None, :] + beta * inputs, 2).sum(1)

    ## Compute Input for Hidden Layer from Visible Potts
    def compute_output_v(self, visible_data):
        # output = torch.zeros((visible_data.shape[0], self.h_num), device=self.device)

        # compute_output of visible potts layer
        vd = visible_data.long()

        # Newest Version also works, fasterst version
        indexTensor = vd.unsqueeze(1).unsqueeze(-1).expand(-1, self.h_num, -1, -1)
        expandedweights = self.W.unsqueeze(0).expand(visible_data.shape[0], -1, -1, -1)
        output = torch.gather(expandedweights, 3, indexTensor).squeeze(3).sum(2)

        # vd shape batch_size x visible
        # output shape batch size x hidden
        # Weight shape hidden x visible x q

        # 2nd fastest this works
        # for u in range(self.h_num):
        #     weight_view = self.W[u].expand(vd.shape[0], -1, -1)
        #     output[:, u] += torch.gather(weight_view, 2, vd.unsqueeze(2)).sum(1).squeeze(1)

        # previous implementation
        # for u in range(self.h_num):  # for u in h_num
        #     for v in range(self.v_num):  # for v in v_num
        #         output1[:, u] += self.W[u, v, vd[:, v]]

        return output

    ## Compute Input for Visible Layer from Hidden dReLU
    def compute_output_h(self, config):
        return torch.tensordot(config, self.W, ([1], [0]))

    ## Gibbs Sampling of Potts Visbile Layer
    ## This thing is the source of my frustration with the ram
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
            # Original Method
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

        etg_plus = self.erf_times_gauss(psi_plus)  # Z+
        etg_minus = self.erf_times_gauss(psi_minus)  # Z-

        p_plus = 1 / (1 + (etg_minus / torch.sqrt(a_minus)) / (etg_plus / torch.sqrt(a_plus)))  # p+
        nans = torch.isnan(p_plus)

        if True in nans:
            p_plus[nans] = torch.tensor(1.) * (torch.abs(psi_plus[nans]) > torch.abs(psi_minus[nans]))
        p_minus = 1 - p_plus

        is_pos = torch.rand(psi.shape, device=self.device) < p_plus

        rmax = torch.zeros(p_plus.shape, device=self.device)
        rmin = torch.zeros(p_plus.shape, device=self.device)
        rmin[is_pos] = torch.erf(psi_plus[is_pos].div(np.sqrt(2)))  # Part of Phi(x)
        rmax[is_pos] = 1  # pos values rmax set to one
        rmin[~is_pos] = -1  # neg samples rmin set to -1
        rmax[~is_pos] = torch.erf((-psi_minus[~is_pos]).div(np.sqrt(2)))  # Part of Phi(x)
        # Pos vals stored as erf(x/sqrt(2)) where x is psi_plus and

        h = torch.zeros(psi.shape, dtype=torch.float64, device=self.device)
        tmp = (rmax - rmin > 1e-14)
        h = np.sqrt(2) * torch.erfinv(rmin + (rmax - rmin) * torch.rand(h.shape, device=self.device))
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
        tmp = X < 6
        m[tmp] = torch.exp(0.5 * X[tmp] ** 2) * (1 - torch.erf(X[tmp] / np.sqrt(2))) * np.sqrt(np.pi / 2)
        m[~tmp] = (1 / X[~tmp] - 1 / X[~tmp] ** 3 + 3 / X[~tmp] ** 5)
        return m

    ## Hidden dReLU supporting Function
    def log_erf_times_gauss(self, X):
        m = torch.zeros_like(X)
        tmp = X < 6
        m[tmp] = (0.5 * X[tmp] ** 2 + torch.log(1 - torch.erf(X[tmp] / np.sqrt(2))) - np.log(2))
        m[~tmp] = (0.5 * np.log(2 / np.pi) - np.log(2) - torch.log(X[~tmp]) + torch.log(1 - 1 / X[~tmp] ** 2 + 3 / X[~tmp] ** 4))
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
    def prepare_data(self):
        try:
            # make and partition data
            if self.weights == "fasta":
                seqs, seq_read_counts = fasta_read(self.fasta_file, seq_read_counts=True, drop_duplicates=True)
                self.weights = np.asarray(seq_read_counts)
            else:
                seqs = fasta_read(self.fasta_file, seq_read_counts=False, drop_duplicates=True)
                seq_read_counts = self.weights

            if self.weights is None:
                data = pd.DataFrame(data={'sequence': seqs})
            else:
                data = pd.DataFrame(data={'sequence': seqs, 'seq_count': self.weights})

        except IOError:
            print(f"Provided Fasta File '{self.fasta_file}' Not Found")
            print(f"Current Directory '{os.curdir}'")
            exit()

        train, validate = train_test_split(data, test_size=0.2, random_state=self.seed)
        self.validation_data = validate
        self.training_data = train

        # self.training_data = data  # still pandas dataframe

    ## Sets Up Optimizer as well as Exponential Weight Decasy
    def configure_optimizers(self):
        optim = self.optimizer(self.parameters(), lr=self.lr, weight_decay=self.wd)
        # optim = self.optimizer(self.weight_param)
        # Exponential Weight Decay after set amount of epochs (set by decay_after)
        decay_gamma = (self.lrf / self.lr) ** (1 / (self.epochs * (1 - self.decay_after)))
        decay_milestone = math.floor(self.decay_after * self.epochs)
        my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=[decay_milestone], gamma=decay_gamma)
        # my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=decay_gamma)
        return optim

    ## Loads Training Data
    def train_dataloader(self):
        # Get Correct Weights
        if "seq_count" in self.training_data.columns:
            training_weights = self.training_data["seq_count"].tolist()
        else:
            training_weights = None

        train_reader = RBMCaterogical(self.training_data, weights=training_weights, max_length=self.v_num, shuffle=False, base_to_id=self.molecule, device=self.device)

        # initialize fields from data
        with torch.no_grad():
            initial_fields = train_reader.field_init()
            self.params['fields'] += initial_fields
            self.params['fields0'] += initial_fields

        if hasattr(self, "trainer"): # Sets Pim Memory when GPU is being used
            pin_mem = self.trainer.on_gpu
        else:
            pin_mem = False

        train_loader = torch.utils.data.DataLoader(
            train_reader,
            batch_size=self.batch_size,
            num_workers=self.worker_num,  # Set to 0 if debug = True
            pin_memory=pin_mem,
            shuffle=True
        )

        return train_loader

    def val_dataloader(self):
        # Get Correct Validation weights
        if "seq_count" in self.validation_data.columns:
            validation_weights = self.validation_data["seq_count"].tolist()
        else:
            validation_weights = None

        val_reader = RBMCaterogical(self.validation_data, weights=validation_weights, max_length=self.v_num, shuffle=False, base_to_id=self.molecule, device=self.device)

        if hasattr(self, "trainer"): # Sets Pim Memory when GPU is being used
            pin_mem = self.trainer.on_gpu
        else:
            pin_mem = False

        train_loader = torch.utils.data.DataLoader(
            val_reader,
            batch_size=self.batch_size,
            num_workers=self.worker_num,  # Set to 0 to view tensors while debugging
            pin_memory=pin_mem,
            shuffle=False
        )

        return train_loader

    ## Calls Corresponding Training Function
    def training_step(self, batch, batch_idx):
        # All other functions use self.W for the weights
        if self.loss_type == "free_energy":
            if self.sample_type == "gibbs":
                return self.training_step_CD_free_energy(batch, batch_idx)
            elif self.sample_type == "pt":
                return self.training_step_PT_free_energy(batch, batch_idx)
        elif self.loss_type == "energy":
            if self.sample_type == "gibbs":
                return self.training_step_CD_energy(batch, batch_idx)
            elif self.sample_type == "pt":
                print("Energy Loss with Parallel Tempering is currently unsupported")
                exit(1)

    def validation_step(self, batch, batch_idx):
        # Needed for PsuedoLikelihood calculation
        self.W = self.params['W_raw'] - self.params['W_raw'].sum(-1).unsqueeze(2) / self.q

        seqs, V_pos, seq_weights = batch

        psuedolikelihood = (self.psuedolikelihood(V_pos) * seq_weights).sum() / seq_weights.sum()

        batch_out = {
             "val_psuedolikelihood": psuedolikelihood.detach()
        }

        self.log("ptl/val_psuedolikelihood", psuedolikelihood, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return batch_out

    def validation_epoch_end(self, outputs):
        avg_pl = torch.stack([x['val_psuedolikelihood'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Validation Psuedolikelihood", avg_pl, self.current_epoch)

    ## On Epoch End Collects Scalar Statistics and Distributions of Parameters for the Tensorboard Logger
    def training_epoch_end(self, outputs):
        # These are detached
        avg_loss = torch.stack([x['loss'].detach() for x in outputs]).mean()
        avg_dF = torch.stack([x["free_energy_diff"] for x in outputs]).mean()
        field_reg = torch.stack([x["field_reg"] for x in outputs]).mean()
        weight_reg = torch.stack([x["weight_reg"] for x in outputs]).mean()
        # energy_reg = torch.stack([x['log']["energy_reg"] for x in outputs]).mean()
        psuedolikelihood = torch.stack([x['train_psuedolikelihood'] for x in outputs]).mean()
        # reconstruction = torch.stack([x['log']['reconstruction_loss'] for x in outputs]).mean()

        self.logger.experiment.add_scalars("All Scalars", {"Loss": avg_loss,
                                                           "CD_Loss": avg_dF,
                                                           "Field Reg": field_reg,
                                                           "Weight Reg": weight_reg,
                                                           "Train_Psuedolikelihood": psuedolikelihood
                                                           # "Reconstruction Loss": reconstruction,
                                                           }, self.current_epoch)

        self.logger.experiment.add_histogram("Weights", self.W.detach(), self.current_epoch)
        for name, p in self.params.items():
            self.logger.experiment.add_histogram(name, p.detach(), self.current_epoch)

    ## This works but not the exact quantity we want to maximize
    def training_step_CD_energy(self, batch, batch_idx):
        seqs, V_pos, seq_weights = batch
        weights = seq_weights.clone.detach()
        V_pos = V_pos.clone().detach()
        V_neg, h_neg, V_pos, h_pos = self(V_pos)

        energy_pos = (self.energy(V_pos, h_pos) * weights).sum() / weights.sum()# energy of training data
        energy_neg = (self.energy(V_neg, h_neg) * weights).sum() / weights.sum() # energy of gibbs sampled visible states

        cd_loss = energy_pos - energy_neg

        psuedolikelihood = (self.psuedolikelihood(V_pos) * weights).sum() / weights.sum()

        reg1 = self.lf / 2 * self.params['fields'].square().sum((0, 1))
        tmp = torch.sum(torch.abs(self.W), (1, 2)).square()
        reg2 = self.l1_2 / (2 * self.q * self.v_num * self.h_num) * tmp.sum()

        loss = cd_loss + reg1 + reg2

        logs = {"loss": loss.detach(),
                "train_psuedolikelihood": psuedolikelihood.detach(),
                "free_energy_diff": cd_loss.detach(),
                "field_reg": reg1.detach(),
                "weight_reg": reg2.detach()
                }

        self.log("ptl/train_psuedolikelihood", psuedolikelihood, on_step=True, on_epoch=True, prog_bar=True, logger=True)
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

        psuedolikelihood = (self.psuedolikelihood(V_pos).clone().detach() * weights).sum() / weights.sum()

        reg1 = self.lf / 2 * self.params['fields'].square().sum((0, 1))
        tmp = torch.sum(torch.abs(self.W), (1, 2)).square()
        reg2 = self.l1_2 / (2 * self.q * self.v_num * self.h_num) * tmp.sum()

        loss = cd_loss + reg1 + reg2

        logs = {"loss": loss.detach(),
                "train_psuedolikelihood": psuedolikelihood.detach(),
                "free_energy_diff": cd_loss.detach(),
                "field_reg": reg1.detach(),
                "weight_reg": reg2.detach()
                }

        self.log("ptl/train_psuedolikelihood", psuedolikelihood, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return logs

    def training_step_CD_free_energy(self, batch, batch_idx):
        # seqs, V_pos, one_hot, seq_weights = batch
        seqs, V_pos, seq_weights = batch
        weights = seq_weights.clone()
        V_neg, h_neg, V_pos, h_pos = self(V_pos)

        # psuedo likelihood actually minimized, loss sits around 0 but does it's own thing
        F_v = (self.free_energy(V_pos) * weights).sum() / weights.sum()  # free energy of training data
        F_vp = (self.free_energy(V_neg) * weights).sum() / weights.sum()  # free energy of gibbs sampled visible states
        cd_loss = F_v - F_vp  # Should Give same gradient as Tubiana Implementation minus the batch norm on the hidden unit activations

        # Reconstruction Loss, Did not work very well
        # V_neg_oh = F.one_hot(V_neg, num_classes=self.q)
        # reconstruction_loss = self.reconstruction_loss(V_neg_oh.double(), one_hot.double())*self.q  # not ideal Loss counts matching zeros as t

        psuedolikelihood = (self.psuedolikelihood(V_pos) * weights).sum() / weights.sum()

        # Regularization Terms
        reg1 = self.lf/2 * self.params['fields'].square().sum((0, 1))
        tmp = torch.sum(torch.abs(self.W), (1, 2)).square()
        reg2 = self.l1_2 / (2 * self.q * self.v_num) * tmp.sum()

        loss = cd_loss + reg1 + reg2 #  + reconstruction_loss

        logs = {"loss": loss,
                "train_psuedolikelihood": psuedolikelihood.detach(),
                "free_energy_diff": cd_loss.detach(),
                "field_reg": reg1.detach(),
                "weight_reg": reg2.detach(),
                # "reconstruction_loss": reconstruction_loss.detach()
                }

        self.log("ptl/train_psuedolikelihood", psuedolikelihood, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return logs

    ## Gradient Clipping for poor behavior, have no need for it yet
    # def on_after_backward(self):
    #     self.grad_norm_clip_value = 10
    #     torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm_clip_value)

    def forward(self, V_pos):
        # Enforces Zero Sum Gauge on Weights
        self.W = self.params['W_raw'] - self.params['W_raw'].sum(-1).unsqueeze(2) / self.q
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
            fantasy_v = self.random_init_config_v(batch_size=n_chains * self.N_PT).reshape([self.N_PT, n_chains, self.v_num])
            fantasy_h = self.random_init_config_h(batch_size=n_chains * self.N_PT).reshape([self.N_PT, n_chains, self.h_num])
            fantasy_E = self.energy_PT(fantasy_v, fantasy_h)

            for _ in range(self.mc_moves):
                fantasy_v, fantasy_h, fantasy_E = self.markov_PT_and_exchange(fantasy_v, fantasy_h, fantasy_E)
                self.update_betas()

            V_neg = fantasy_v[0, :, :]
            h_neg = fantasy_h[0, :, :]

            return V_neg, h_neg, V_pos, h_pos

    ## For debugging of main functions
    def sampling_test(self):
        self.prepare_data()
        train_reader = RBMCaterogical(self.training_data, weights=self.weights, max_length=self.v_num, shuffle=False, base_to_id=self.base_to_id, device=self.device)

        # initialize fields from data
        with torch.no_grad():
            initial_fields = train_reader.field_init()
            self.params['fields'] += initial_fields
            self.params['fields0'] += initial_fields

        self.W = self.params['W_raw'] - self.params['W_raw'].sum(-1).unsqueeze(2) / self.q

        v = self.random_init_config_v()
        h = self.sample_from_inputs_h(self.compute_output_v(v))
        v2 = self.sample_from_inputs_v(self.compute_output_h(h))

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
                hidden_data = torch.zeros((Nchains, N_PT, Lchains, self.v_num), dtype=torch.float64)
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
                config = [self.random_init_config_v(batch_size=batches*N_PT).reshape((N_PT, batches, self.v_num)), self.random_init_config_h(batch_size=batches*N_PT).reshape((N_PT, batches, self.h_num))]
                if config_init != []:
                    config[0] = config_init[0]
                    config[1] = config_init[1]
                energy = torch.zeros([N_PT, batches])
            else:
                if config_init != []:
                    config = config_init
                else:
                    config = [self.random_init_config_v(batch_size=batches), self.random_init_config_h(batch_size=batches)]

            for _ in range(Nthermalize):
                if N_PT > 1:
                    energy = self.energy_PT(config[0], config[1])
                    config[0], config[1], energy = self.markov_PT_and_exchange(config[0], config[1], energy)
                    if update_betas:
                        self.update_betas(beta=beta)
                else:
                    config[0], config[1] = self.markov_step(config[0], beta=beta)

            if N_PT > 1:
                if record_replica:
                    data = [config[0].clone().unsqueeze(0), config[1].clone().unsqueeze(0)]
                else:
                    data = [config[0][0].clone().unsqueeze(0), config[1][0].clone().unsqueeze(0)]
            else:
                data = [config[0].clone().unsqueeze(0), config[1].clone().unsqueeze(0)]


            for _ in range(Ndata - 1):
                for _ in range(Nstep):
                    if N_PT > 1:
                        energy = self.energy_PT(config[0], config[1])
                        config[0], config[1], energy = self.markov_PT_and_exchange(config[0], config[1], energy)
                        if update_betas:
                            self.update_betas(beta=beta)
                    else:
                        config[0], config[1] = self.markov_step(config[0], beta=beta)

                if N_PT > 1:
                    if record_replica:
                        data[0].append(config[0].clone())
                        data[1].append(config[1].clone())
                    else:
                        data[0].append(config[0][0].clone())
                        data[1].append(config[1][0].clone())
                else:
                    data[0].append(config[0].clone())
                    data[1].append(config[1].clone())

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




# returns list of strings containing sequences
# optionally returns the affinities

# Fasta File Reader
def fasta_read(fastafile, seq_read_counts=False, drop_duplicates=False):
    o = open(fastafile)
    titles = []
    seqs = []
    for line in o:
        if line.startswith('>'):
            if seq_read_counts:
                titles.append(float(line.rstrip().split('-')[1]))
        else:
            seqs.append(line.rstrip())
    o.close()
    if drop_duplicates:
        # seqs = list(set(seqs))
        all_seqs = pd.DataFrame(seqs).drop_duplicates()
        seqs = all_seqs.values.tolist()
        seqs = [j for i in seqs for j in i]

    if seq_read_counts:
        return seqs, titles
    else:
        return seqs

# Get Model from checkpoint File with specified version and directory
def get_checkpoint(version, dir=""):
    checkpoint_dir = dir + "/version_" + str(version) + "/checkpoints/"

    for file in os.listdir(checkpoint_dir):
        if file.endswith(".ckpt"):
            checkpoint_file = os.path.join(checkpoint_dir, file)
    return checkpoint_file


def get_beta_and_W(rbm):
    W = rbm.get_param("W")
    return np.sqrt((W ** 2).sum(-1).sum(-1)), W

def all_weights(rbm, name, rows, columns, h, w, molecule='rna'):
    beta, W = get_beta_and_W(rbm)
    order = np.argsort(beta)[::-1]
    fig = rbm_utils.Sequence_logo_all(W[order], name=name + '.pdf', nrows=rows, ncols=columns, figsize=(h,w) ,ticks_every=10,ticks_labels_size=10,title_size=12, dpi=400, molecule=molecule)


if __name__ == '__main__':
    # pytorch lightning loop
    data_file = '../invivo/sham2_ipsi_c1.fasta'  # cpu is faster
    large_data_file = '../invivo/chronic1_spleen_c1.fasta' # gpu is faster
    lattice_data = './lattice_proteins_verification/Lattice_Proteins_MSA.fasta'

    config = {"fasta_file": lattice_data,
              "molecule": "protein",
              "h_num": 10,  # number of hidden units, can be variable
              "v_num": 27,
              "q": 21,
              "batch_size": 10000,
              "mc_moves": 6,
              "seed": 38,
              "lr": 0.0065,
              "lr_final": None,
              "decay_after": 0.75,
              "loss_type": "free_energy",
              "sample_type": "gibbs",
              "sequence_weights": None,
              "optimizer": "AdamW",
              "epochs": 100,
              "weight_decay": 0.001,  # l2 norm on all parameters
              "l1_2": 0.185,
              "lf": 0.002,
              }


    # Training Code
    rbm_lat = RBM(config, debug=True)
    logger = TensorBoardLogger('tb_logs', name='lattice_trial')
    plt = pl.Trainer(max_epochs=config['epochs'], logger=logger, gpus=1)  # gpus=1,
    plt.fit(rbm_lat)


    # check weights
    # version = 0
    # # checkpoint = get_checkpoint(version, dir="./tb_logs/lattice_trial/")
    # checkpoint = "/mnt/D1/globus/rbm_hyperparam_results/train_rbm_ad5d7_00005_5_l1_2=0.3832,lf=0.00011058,lr=0.065775,weight_decay=0.086939_2022-01-18_11-02-53/checkpoints/epoch=99-step=499.ckpt"
    # rbm = RBM.load_from_checkpoint(checkpoint)
    # rbm.energy_per_state()
    # all_weights(rbm, "./lattice_proteins_verification" + "/allweights", 5, 1, 10, 2, molecule="protein")

    # checkpoint = torch.load(checkpoint_file)
    # model.prepare_data()
    # model.criterion.weight = torch.tensor([0., 0.]) # need to add as this is saved by the checkpoint file
    # model.load_state_dict(checkpoint['state_dict'])



    ## Need to finish debugging AIS
    # rbm = RBM(config)
    # rbm.sampling_test()
    # rbm.AIS()

    # rbm.prepare_data()


    # d = iter(rbm.train_dataloader())
    # seqs, v_pos, weights = d.next()
    # logger.experiment.add_graph(rbm, v_pos)

    # profiler = torch.profiler.profile(profile_memory=True)
    # profiler = pytorch_lightning.profiler.SimpleProfiler(profile_memory=True)
    # profiler = pytorch_lightning.profiler.PyTorchProfiler(profile_memory=True)

    # logger = TensorBoardLogger('tb_logs', name='bench_trial')
    # # plt = pl.Trainer(max_epochs=epochs, logger=logger, gpus=0, profiler=profiler)  # gpus=1,
    # plt = pl.Trainer(max_epochs=epochs, logger=logger, gpus=0, profiler="advanced")  # gpus=1,
    # # tic = time.perf_counter()
    # plt.fit(rbm)
    # toc = time.perf_counter()
    # tim = toc-tic
    #
    # print("Trial took", tim, "seconds")



    # version 11 of fixed trial is 50 epochs of pt sampled


    # check weights
    # version = 15
    # # checkpoint = get_checkpoint(version, dir="./tb_logs/trial/")
    # checkpoint = get_checkpoint(version, dir="./tb_logs/bench_trial/")
    #
    # rbm = RBM.load_from_checkpoint(checkpoint)
    # all_weights(rbm, "./tb_logs/bench_trial/version_" + str(version) + "/allweights", 5, 1, 10, 2, molecule="protein")




    # plt = pl.Trainer(gpus=1, max_epochs=10)
    # plt = pl.Trainer(gpus=1, profiler='advanced', max_epochs=10)
    # plt = pl.Trainer(profiler='advanced', max_epochs=10)
    # plt = pl.Trainer(max_epochs=1)
    # plt.fit(rbm)


    # total = 0
    # for i, batch in enumerate(d):
    #     print(len(batch))
    #     seqs, tens = batch
    #     if i == 0:
    #         # rbm.testing()
    #         rbm.training_step(batch, i)


