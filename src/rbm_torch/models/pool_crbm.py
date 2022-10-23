import time
import pandas as pd
import math
import json
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.profiler import SimpleProfiler, PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split, GroupShuffleSplit

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW, Adagrad, Adadelta  # Supported Optimizers
from multiprocessing import cpu_count # Just to set the worker number
from torch.autograd import Variable

from rbm_torch.utils.utils import Categorical, fasta_read, conv2d_dim, pool1d_dim, BatchNorm1D, label_samples, process_weights, configure_optimizer, StratifiedBatchSampler, WeightedSubsetRandomSampler  #Sequence_logo, gen_data_lowT, gen_data_zeroT, all_weights, Sequence_logo_all,
from rbm_torch.utils.data_prep import weight_transform, pearson_transform
from torch.utils.data import WeightedRandomSampler

class pool_CRBM(LightningModule):
    def __init__(self, config, debug=False, precision="double", meminfo=False):
        super().__init__()

        # Pytorch Basic Options #
        ################################
        self.seed = config['seed']
        torch.manual_seed(self.seed)  # For reproducibility
        supported_precisions = {"double": torch.float64, "single": torch.float32}
        try:
            torch.set_default_dtype(supported_precisions[precision])
        except:
            print(f"Precision {precision} not supported.")
            exit(-1)
        ################################


        self.mc_moves = config['mc_moves']  # Number of MC samples to take to update hidden and visible configurations
        self.batch_size = config['batch_size']  # Pretty self explanatory
        self.epochs = config['epochs']  # number of training iterations, needed for our weight decay function

        # Data Input Options #
        ###########################################
        self.fasta_file = config['fasta_file']
        self.v_num = config['v_num']  # Number of visible nodes
        self.q = config['q']  # Number of categories the input sequence has (ex. DNA:4 bases + 1 gap)
        self.validation_size = config['validation_set_size']
        self.test_size = config['test_set_size']
        assert self.validation_size < 1.0
        assert self.test_size < 1.0

        self.molecule = config['molecule']  # can be protein, rna or dna currently
        assert self.molecule in ["dna", "rna", "protein"]

        # Batch sampling strategy, can be random or stratified
        try:
            self.sampling_strategy = config["sampling_strategy"]
        except KeyError:
            self.sampling_strategy = "random"
        assert self.sampling_strategy in ["random", "stratified", "weighted", "stratified_weighted"]

        # Only used is sampling strategy is weighted
        try:
            self.sampling_weights = config["sampling_weights"]
        except KeyError:
            self.sampling_weights = None


        # Sequence Weighting Weights
        # Not pretty but necessary to either provide the weights or to import from the fasta file
        # To import from the provided fasta file weights="fasta" in intialization of RBM
        weights = config['sequence_weights']
        self.weights = process_weights(weights)

        # Stratify the datasets, training, validationa, and test
        try:
            self.stratify = config["stratify_datasets"]
        except KeyError:
            self.stratify = False


        ###########################################


        # Dataloader Configuration Options #
        ###########################################
        # Sets worker number for both dataloaders
        if debug:
            self.worker_num = 0
        else:
            try:
                self.worker_num = config["data_worker_num"]
            except KeyError:
                self.worker_num = cpu_count()

        # Sets Pim Memory when GPU is being used
        # this attribute is set in load_run_file
        try:
            if config["gpus"] > 0:
                self.pin_mem = True
            else:
                self.pin_mem = False
        except KeyError:
            self.pin_mem = False
        ###########################################


        # Global Optimizer Settings #
        ###########################################
        # configure optimizer
        optimizer = config['optimizer']
        self.optimizer = configure_optimizer(optimizer)

        self.lr = config['lr']
        lr_final = config['lr_final']

        if lr_final is None:
            self.lrf = self.lr * 1e-2
        else:
            self.lrf = lr_final

        self.wd = config['weight_decay']  # Put into weight decay option in configure_optimizer, l2 regularizer
        self.decay_after = config['decay_after']  # hyperparameter for when the lr decay should occur
        ###########################################


        # loss types are 'energy' and 'free_energy' for now, controls the loss function primarily
        # sample types control whether gibbs sampling, pcd, from the data points or parallel tempering from random configs are used
        # Switches How the training of the RBM is performed

        self.loss_type = config['loss_type']
        self.sample_type = config['sample_type']

        assert self.loss_type in ['energy', 'free_energy']
        assert self.sample_type in ['gibbs', 'pt', 'pcd']


        # Regularization Options #
        ###########################################
        self.l1_2 = config['l1_2']  # regularization on weights, ex. 0.25
        self.lf = config['lf']  # regularization on fields, ex. 0.001
        self.ld = config['ld']
        self.lgap = config['lgap']
        self.lbs = config['lbs']
        ###########################################


        self.convolution_topology = config["convolution_topology"]

        if type(self.v_num) is int:
            # Normal dist. times this value sets initial weight values
            self.weight_intial_amplitude = np.sqrt(0.1 / self.v_num)
            self.register_parameter("fields", nn.Parameter(torch.zeros((self.v_num, self.q), device=self.device)))
            self.register_parameter("fields0", nn.Parameter(torch.zeros((self.v_num, self.q), device=self.device)))
        elif type(self.v_num) is tuple:  # Normal dist. times this value sets initial weight values

            self.weight_intial_amplitude = np.sqrt(0.1 / math.prod(list(self.v_num)))
            self.register_parameter("fields", nn.Parameter(torch.zeros((*self.v_num, self.q), device=self.device)))
            self.register_parameter("fields0", nn.Parameter(torch.zeros((*self.v_num, self.q), device=self.device)))

        self.hidden_convolution_keys = list(self.convolution_topology.keys())

        # Hidden Layer Weights, Two options (1) provided weights or (2) learns the weights as a model parameter
        # try:
        #     hidden_layer_weights = [v['weight'] for x, v in self.convolution_topology.items()]
        #     self.register_parameter("hidden_layer_W", nn.Parameter(torch.tensor(hidden_layer_weights, device=self.device), requires_grad=False))
        # except KeyError:
        #     print("Hidden layer weights not provided or incomplete. Attempting to learn instead.")
        #     self.register_parameter("hidden_layer_W", nn.Parameter(torch.ones((len(self.hidden_convolution_keys)), device=self.device), requires_grad=True))

        self.use_pearson = config["use_pearson"]
        self.pearson_xvar = "none"
        if self.use_pearson:
            self.pearson_xvar = config["pearson_xvar"] # label or fitness_value

            assert self.pearson_xvar in ["values", "labels"]

        # if self.pearson_xvar == "labels" or self.stratify or self.sampling_strategy == "stratified":
        self.label_spacing = config["label_spacing"]
        self.label_groups = len(self.label_spacing) - 1
        try:
            self.group_fraction = config["group_fraction"]
        except KeyError:
            self.group_fraction = [1/self.label_groups for i in self.label_groups]

        try:
            self.sample_multiplier = config["sample_multiplier"]
        except KeyError:
            self.sample_multiplier = 1.

        assert len(self.label_spacing) - 1 == self.label_groups
        self.labels_in_batch = True


        self.use_batch_norm = config["use_batch_norm"]
        self.dr = 0.
        if "dr" in config.keys():
            self.dr = config["dr"]

        self.pools = []
        self.unpools = []

        for key in self.hidden_convolution_keys:
            # Set information about the convolutions that will be useful
            dims = conv2d_dim([self.batch_size, 1, self.v_num, self.q], self.convolution_topology[key])
            self.convolution_topology[key]["weight_dims"] = dims["weight_shape"]
            self.convolution_topology[key]["convolution_dims"] = dims["conv_shape"]
            self.convolution_topology[key]["output_padding"] = dims["output_padding"]

            # deal with pool and unpool initialization
            pool_input_size = dims["conv_shape"][:-1]

            pool_kernel = pool_input_size[2]
            pool_stride = 1
            pool_padding = 0

            self.pools.append(nn.MaxPool1d(pool_kernel, stride=pool_stride, return_indices=True, padding=pool_padding))
            self.unpools.append(nn.MaxUnpool1d(pool_kernel, stride=pool_stride, padding=pool_padding))

            if self.use_batch_norm:
                setattr(self, f"batch_norm_{key}", BatchNorm1D(affine=False, momentum=0.1))
                setattr(self, f"batch_norm_{key}", BatchNorm1D(affine=False, momentum=0.1))

            # Convolution Weights
            self.register_parameter(f"{key}_W", nn.Parameter(self.weight_intial_amplitude * torch.randn(self.convolution_topology[key]["weight_dims"], device=self.device)))
            # hidden layer parameters
            self.register_parameter(f"{key}_theta+", nn.Parameter(torch.zeros(self.convolution_topology[key]["number"], device=self.device)))
            self.register_parameter(f"{key}_theta-", nn.Parameter(torch.zeros(self.convolution_topology[key]["number"], device=self.device)))
            self.register_parameter(f"{key}_gamma+", nn.Parameter(torch.ones(self.convolution_topology[key]["number"], device=self.device)))
            self.register_parameter(f"{key}_gamma-", nn.Parameter(torch.ones(self.convolution_topology[key]["number"], device=self.device)))
            # Used in PT Sampling / AIS
            self.register_parameter(f"{key}_0theta+", nn.Parameter(torch.zeros(self.convolution_topology[key]["number"], device=self.device), requires_grad=False))
            self.register_parameter(f"{key}_0theta-", nn.Parameter(torch.zeros(self.convolution_topology[key]["number"], device=self.device), requires_grad=False))
            self.register_parameter(f"{key}_0gamma+", nn.Parameter(torch.ones(self.convolution_topology[key]["number"], device=self.device), requires_grad=False))
            self.register_parameter(f"{key}_0gamma-", nn.Parameter(torch.ones(self.convolution_topology[key]["number"], device=self.device), requires_grad=False))


        # Saves Our hyperparameter options into the checkpoint file generated for Each Run of the Model
        # i. e. Simplifies loading a model that has already been run
        self.save_hyperparameters()

        # Constants for faster math
        self.logsqrtpiover2 = torch.tensor(0.2257913526, device=self.device, requires_grad=False)
        self.pbis = torch.tensor(0.332672, device=self.device, requires_grad=False)
        self.a1 = torch.tensor(0.3480242, device=self.device, requires_grad=False)
        self.a2 = torch.tensor(- 0.0958798, device=self.device, requires_grad=False)
        self.a3 = torch.tensor(0.7478556, device=self.device, requires_grad=False)
        self.invsqrt2 = torch.tensor(0.7071067812, device=self.device, requires_grad=False)
        self.sqrt2 = torch.tensor(1.4142135624, device=self.device, requires_grad=False)

        # Initialize PT members
        if self.sample_type == "pt":
            try:
                self.N_PT = config["N_PT"]
            except KeyError:
                print("No member N_PT found in provided config.")
                exit(-1)
            self.initialize_PT(self.N_PT, n_chains=None, record_acceptance=True, record_swaps=True)

        self.meminfo = meminfo

    @property
    def h_layer_num(self):
        return len(self.hidden_convolution_keys)

    # Initializes Members for both PT and gen_data functions
    def initialize_PT(self, N_PT, n_chains=None, record_acceptance=False, record_swaps=False):
        self.record_acceptance = record_acceptance
        self.record_swaps = record_swaps

        # self.update_betas()
        self.betas = torch.arange(N_PT) / (N_PT - 1)
        self.betas = self.betas.flip(0)

        if n_chains is None:
            n_chains = self.batch_size

        self.particle_id = [torch.arange(N_PT).unsqueeze(1).expand(N_PT, n_chains)]

        # if self.record_acceptance:
        self.mavar_gamma = 0.95
        self.acceptance_rates = torch.zeros(N_PT - 1, device=self.device)
        self.mav_acceptance_rates = torch.zeros(N_PT - 1, device=self.device)

        # gen data
        self.count_swaps = 0
        self.last_at_zero = None
        self.trip_duration = None
        # self.update_betas_lr = 0.1
        # self.update_betas_lr_decay = 1

    ## Used in our Loss Function
    def free_energy(self, v):
        return self.energy_v(v) - self.logpartition_h(self.compute_output_v(v))

    ## Not used but may be useful
    def free_energy_h(self, h):
        return self.energy_h(h) - self.logpartition_v(self.compute_output_h(h))

    ## Total Energy of a given visible and hidden configuration
    def energy(self, v, h, remove_init=False, hidden_sub_index=-1):
        return self.energy_v(v, remove_init=remove_init) + self.energy_h(h, sub_index=hidden_sub_index, remove_init=remove_init) - self.bidirectional_weight_term(v, h, hidden_sub_index=hidden_sub_index)

    def energy_PT(self, v, h, N_PT, remove_init=False):
        # if N_PT is None:
        #     N_PT = self.N_PT
        E = torch.zeros((N_PT, v.shape[1]), device=self.device)
        for i in range(N_PT):
            E[i] = self.energy_v(v[i], remove_init=remove_init) + self.energy_h(h, sub_index=i, remove_init=remove_init) - self.bidirectional_weight_term(v[i], h, hidden_sub_index=i)
        return E

    def bidirectional_weight_term(self, v, h, hidden_sub_index=-1):
        conv = self.compute_output_v(v)
        E = torch.zeros((len(self.hidden_convolution_keys), conv[0].shape[0]), device=self.device)
        for iid, i in enumerate(self.hidden_convolution_keys):
            if hidden_sub_index != -1:
                h_uk = h[iid][hidden_sub_index]
            else:
                h_uk = h[iid]
            E[iid] = h_uk.mul(conv[iid]).sum(1)

        if E.shape[0] > 1:
            return E.sum(0)
        else:
            return E.squeeze(0)

    ############################################################# Individual Layer Functions
    def transform_v(self, I):
        return F.one_hot(torch.argmax(I + getattr(self, "fields").unsqueeze(0), dim=-1), self.q)
        # return self.one_hot_tmp.scatter(2, torch.argmax(I + getattr(self, "fields").unsqueeze(0), dim=-1).unsqueeze(-1), 1.)

    def transform_h(self, I):
        output = []
        for kid, key in enumerate(self.hidden_convolution_keys):
            a_plus = (getattr(self, f'{key}_gamma+')).unsqueeze(0).unsqueeze(2)
            a_minus = (getattr(self, f'{key}_gamma-')).unsqueeze(0).unsqueeze(2)
            theta_plus = (getattr(self, f'{key}_theta+')).unsqueeze(0).unsqueeze(2)
            theta_minus = (getattr(self, f'{key}_theta-')).unsqueeze(0).unsqueeze(2)
            tmp = ((I[kid] + theta_minus) * (I[kid] <= torch.minimum(-theta_minus, (theta_plus / torch.sqrt(a_plus) -
                                                                                    theta_minus / torch.sqrt(a_minus)) / (1 / torch.sqrt(a_plus) + 1 / torch.sqrt(a_minus))))) / \
                  a_minus + ((I[kid] - theta_plus) * (I[kid] >= torch.maximum(theta_plus, (theta_plus / torch.sqrt(a_plus) -
                                                                                           theta_minus / torch.sqrt(a_minus)) / (1 / torch.sqrt(a_plus) + 1 / torch.sqrt(a_minus))))) / a_plus
            output.append(tmp)
        return output

    ## Computes g(si) term of potential
    def energy_v(self, config, remove_init=False):
        # config is a one hot vector
        v = config.type(torch.get_default_dtype())
        E = torch.zeros(config.shape[0], device=self.device)
        for i in range(self.q):
            if remove_init:
                E -= v[:, :, i].dot(getattr(self, "fields")[:, i] - getattr(self, "fields0")[:, i])
            else:
                E -= v[:, :, i].matmul(getattr(self, "fields")[:, i])

        return E

    ## Computes U(h) term of potential
    def energy_h(self, config, remove_init=False, sub_index=-1):
        # config is list of h_uks
        if sub_index != -1:
            E = torch.zeros((len(self.hidden_convolution_keys), config[0].shape[1]), device=self.device)
        else:
            E = torch.zeros((len(self.hidden_convolution_keys), config[0].shape[0]), device=self.device)

        for iid, i in enumerate(self.hidden_convolution_keys):
            if remove_init:
                a_plus = getattr(self, f'{i}_gamma+').sub(getattr(self, f'{i}_0gamma+')).unsqueeze(0)
                a_minus = getattr(self, f'{i}_gamma-').sub(getattr(self, f'{i}_0gamma-')).unsqueeze(0)
                theta_plus = getattr(self, f'{i}_theta+').sub(getattr(self, f'{i}_0theta+')).unsqueeze(0)
                theta_minus = getattr(self, f'{i}_theta-').sub(getattr(self, f'{i}_0theta-')).unsqueeze(0)
            else:
                a_plus = getattr(self, f'{i}_gamma+').unsqueeze(0)
                a_minus = getattr(self, f'{i}_gamma-').unsqueeze(0)
                theta_plus = getattr(self, f'{i}_theta+').unsqueeze(0)
                theta_minus = getattr(self, f'{i}_theta-').unsqueeze(0)

            if sub_index != -1:
                con = config[iid][sub_index]
            else:
                con = config[iid]

            # Applies the dReLU activation function
            zero = torch.zeros_like(con, device=self.device)
            config_plus = torch.maximum(con, zero)
            config_minus = -1*torch.maximum(-con, zero)

            E[iid] = ((config_plus.square() * a_plus) / 2 + (config_minus.square() * a_minus) / 2 + (config_plus * theta_plus) + (config_minus * theta_minus)).sum(1)
            # E[iid] *= (10/self.convolution_topology[i]["convolution_dims"][2]) # Normalize across different convolution sizes
        if E.shape[0] > 1:
            return E.sum(0)
        else:
            return E.squeeze(0)

    ## Random Config of Visible Potts States
    def random_init_config_v(self, custom_size=False, zeros=False):
        if custom_size:
            size = (*custom_size, self.v_num, self.q)
        else:
            size = (self.batch_size, self.v_num, self.q)

        if zeros:
            return torch.zeros(size, device=self.device)
        else:
            return self.sample_from_inputs_v(torch.zeros(size, device=self.device).flatten(0, -3), beta=0).reshape(size)

    ## Random Config of Hidden dReLU States
    def random_init_config_h(self, zeros=False, custom_size=False):
        config = []
        for iid, i in enumerate(self.hidden_convolution_keys):
            batch, h_num, convx_num, convy_num = self.convolution_topology[i]["convolution_dims"]

            if custom_size:
                size = (*custom_size, h_num)
            else:
                size = (self.batch_size, h_num)

            if zeros:
                config.append(torch.zeros(size, device=self.device))
            else:
                config.append(torch.randn(size, device=self.device))

        return config

    def clone_h(self, hidden_config, reduce_dims=[], expand_dims=[], sub_index=-1):
        new_config = []
        for hc in hidden_config:
            if sub_index != -1:
                new_h = hc[sub_index].clone()
            else:
                new_h = hc.clone()
            for dim in reduce_dims:
                new_h = new_h.squeeze(dim)
            for dim in expand_dims:
                new_h = new_h.unsqueeze(dim)
            new_config.append(new_h)
        return new_config

    ## Marginal over hidden units
    def logpartition_h(self, inputs, beta=1):
        # Input is list of matrices I_uk
        marginal = torch.zeros((len(self.hidden_convolution_keys), inputs[0].shape[0]), device=self.device)
        for iid, i in enumerate(self.hidden_convolution_keys):
            if beta == 1:
                a_plus = (getattr(self, f'{i}_gamma+')).unsqueeze(0)
                a_minus = (getattr(self, f'{i}_gamma-')).unsqueeze(0)
                theta_plus = (getattr(self, f'{i}_theta+')).unsqueeze(0)
                theta_minus = (getattr(self, f'{i}_theta-')).unsqueeze(0)
            else:
                theta_plus = (beta * getattr(self, f'{i}_theta+') + (1 - beta) * getattr(self, f'{i}_0theta+')).unsqueeze(0)
                theta_minus = (beta * getattr(self, f'{i}_theta-') + (1 - beta) * getattr(self, f'{i}_0theta-')).unsqueeze(0)
                a_plus = (beta * getattr(self, f'{i}_gamma+') + (1 - beta) * getattr(self, f'{i}_0gamma+')).unsqueeze(0)
                a_minus = (beta * getattr(self, f'{i}_gamma-') + (1 - beta) * getattr(self, f'{i}_0gamma-')).unsqueeze(0)

            # in_neg = inputs[iid][:, :, 1]
            # in_pos = inputs[iid][:, :, 0]
            y = torch.logaddexp(self.log_erf_times_gauss((-inputs[iid] + theta_plus) / torch.sqrt(a_plus)) -
                                0.5 * torch.log(a_plus), self.log_erf_times_gauss((inputs[iid] + theta_minus) / torch.sqrt(a_minus)) - 0.5 * torch.log(a_minus)).sum(
                1) + 0.5 * np.log(2 * np.pi) * inputs[iid].shape[1]
            marginal[iid] = y  # 10 added so hidden layer has stronger effect on free energy, also in energy_h
            # marginal[iid] /= self.convolution_topology[i]["convolution_dims"][2]
        return marginal.sum(0)

    ## Marginal over visible units
    def logpartition_v(self, inputs, beta=1):
        if beta == 1:
            return torch.logsumexp(getattr(self, "fields")[None, :, :] + inputs, 2).sum(1)
        else:
            return torch.logsumexp((beta * getattr(self, "fields") + (1 - beta) * getattr(self, "fields0"))[None, :] + beta * inputs, 2).sum(1)

    ## Mean of hidden layer specified by hidden_key
    def mean_h(self, psi, hidden_key=None, beta=1):
        if hidden_key is None:
            means = []
            for kid, key in enumerate(self.hidden_convolution_keys):
                if beta == 1:
                    a_plus = (getattr(self, f'{key}_gamma+')).unsqueeze(0)
                    a_minus = (getattr(self, f'{key}_gamma-')).unsqueeze(0)
                    theta_plus = (getattr(self, f'{key}_theta+')).unsqueeze(0)
                    theta_minus = (getattr(self, f'{key}_theta-')).unsqueeze(0)
                else:
                    theta_plus = (beta * getattr(self, f'{key}_theta+') + (1 - beta) * getattr(self, f'{key}_0theta+')).unsqueeze(0)
                    theta_minus = (beta * getattr(self, f'{key}_theta-') + (1 - beta) * getattr(self, f'{key}_0theta-')).unsqueeze(0)
                    a_plus = (beta * getattr(self, f'{key}_gamma+') + (1 - beta) * getattr(self, f'{key}_0gamma+')).unsqueeze(0)
                    a_minus = (beta * getattr(self, f'{key}_gamma-') + (1 - beta) * getattr(self, f'{key}_0gamma-')).unsqueeze(0)
                    psi[kid] *= beta

                # if psi[kid].dim() == 3:
                #     a_plus = a_plus.unsqueeze(2)
                #     a_minus = a_minus.unsqueeze(2)
                #     theta_plus = theta_plus.unsqueeze(2)
                #     theta_minus = theta_minus.unsqueeze(2)

                psi_plus = (-psi[kid] + theta_plus) / torch.sqrt(a_plus)
                psi_minus = (psi[kid] + theta_minus) / torch.sqrt(a_minus)

                etg_plus = self.erf_times_gauss(psi_plus)
                etg_minus = self.erf_times_gauss(psi_minus)

                p_plus = 1 / (1 + (etg_minus / torch.sqrt(a_minus)) / (etg_plus / torch.sqrt(a_plus)))
                nans = torch.isnan(p_plus)
                p_plus[nans] = 1.0 * (torch.abs(psi_plus[nans]) > torch.abs(psi_minus[nans]))
                p_minus = 1 - p_plus

                mean_pos = (-psi_plus + 1 / etg_plus) / torch.sqrt(a_plus)
                mean_neg = (psi_minus - 1 / etg_minus) / torch.sqrt(a_minus)
                means.append(mean_pos * p_plus + mean_neg * p_minus)
            return means
        else:
            if beta == 1:
                a_plus = (getattr(self, f'{hidden_key}_gamma+')).unsqueeze(0)
                a_minus = (getattr(self, f'{hidden_key}_gamma-')).unsqueeze(0)
                theta_plus = (getattr(self, f'{hidden_key}_theta+')).unsqueeze(0)
                theta_minus = (getattr(self, f'{hidden_key}_theta-')).unsqueeze(0)
            else:
                theta_plus = (beta * getattr(self, f'{hidden_key}_theta+') + (1 - beta) * getattr(self, f'{hidden_key}_0theta+')).unsqueeze(0)
                theta_minus = (beta * getattr(self, f'{hidden_key}_theta-') + (1 - beta) * getattr(self, f'{hidden_key}_0theta-')).unsqueeze(0)
                a_plus = (beta * getattr(self, f'{hidden_key}_gamma+') + (1 - beta) * getattr(self, f'{hidden_key}_0gamma+')).unsqueeze(0)
                a_minus = (beta * getattr(self, f'{hidden_key}_gamma-') + (1 - beta) * getattr(self, f'{hidden_key}_0gamma-')).unsqueeze(0)
                psi *= beta

            # if psi.dim() == 3:
            #     a_plus = a_plus.unsqueeze(2)
            #     a_minus = a_minus.unsqueeze(2)
            #     theta_plus = theta_plus.unsqueeze(2)
            #     theta_minus = theta_minus.unsqueeze(2)

            psi_plus = (psi + theta_plus) / torch.sqrt(a_plus)  #  min pool
            psi_minus = (psi + theta_minus) / torch.sqrt(a_minus)  # max pool

            etg_plus = self.erf_times_gauss(psi_plus)
            etg_minus = self.erf_times_gauss(psi_minus)

            p_plus = 1 / (1 + (etg_minus / torch.sqrt(a_minus)) / (etg_plus / torch.sqrt(a_plus)))
            nans = torch.isnan(p_plus)
            p_plus[nans] = 1.0 * (torch.abs(psi_plus[nans]) > torch.abs(psi_minus[nans]))
            p_minus = 1 - p_plus

            mean_pos = (-psi_plus + 1 / etg_plus) / torch.sqrt(a_plus)
            mean_neg = (psi_minus - 1 / etg_minus) / torch.sqrt(a_minus)
            return mean_pos * p_plus + mean_neg * p_minus

    ## Compute Input for Hidden Layer from Visible Potts, Uses one hot vector
    def compute_output_v(self, X):  # X is the one hot vector
        outputs = []
        # hidden_layer_W = getattr(self, "hidden_layer_W")
        # total_weights = hidden_layer_W.sum()
        self.max_inds = []
        self.min_inds = []
        for iid, i in enumerate(self.hidden_convolution_keys):
            # convx = self.convolution_topology[i]["convolution_dims"][2]
            weights = getattr(self, f"{i}_W")
            conv = F.conv2d(X.unsqueeze(1).type(torch.get_default_dtype()), weights, stride=self.convolution_topology[i]["stride"],
                            padding=self.convolution_topology[i]["padding"],
                            dilation=self.convolution_topology[i]["dilation"]).squeeze(3)

            max_pool, max_inds = self.pools[iid](conv.abs())

            flat_conv = conv.flatten(start_dim=2)
            max_conv_values = flat_conv.gather(2, index=max_inds.flatten(start_dim=2)).view_as(max_inds)

            # max_conv_values = torch.gather(conv, 2, max_inds)
            max_pool *= max_conv_values/max_conv_values.abs()

            self.max_inds.append(max_inds)

            out = max_pool.flatten(start_dim=2)

            if self.use_batch_norm:
                batch_norm = getattr(self, f"batch_norm_{i}")  # get individual batch norm
                out = batch_norm(out)  # apply batch norm

            out.squeeze_(2)

            if self.dr > 0.:
                # dropout_mask = F.dropout(torch.ones((out.shape[1]), device=self.device))
                # out = out * dropout_mask
                out = F.dropout(out, p=self.dr, training=self.training)

            outputs.append(out)

        return outputs

    ## Compute Input for Visible Layer from Hidden dReLU
    def compute_output_h(self, Y):  # from h_uk (B, hidden_num)
        outputs = []
        # nonzero_masks = []
        # hidden_layer_W = getattr(self, "hidden_layer_W")
        # total_weights = hidden_layer_W.sum()
        for iid, i in enumerate(self.hidden_convolution_keys):

            # zero = torch.zeros_like(Y[iid], device=self.device)
            # Y_pos = torch.maximum(Y[iid], zero)
            # Y_neg = torch.maximum(-Y[iid], zero)

            # max_reconst = self.unpools[iid](Y_pos.unsqueeze(2), self.max_inds[iid])
            # min_reconst = -1*self.unpools[iid](Y_neg.unsqueeze(2), self.min_inds[iid])

            reconst = self.unpools[iid](Y[iid].view_as(self.max_inds[iid]), self.max_inds[iid])

            if reconst.ndim == 3:
                reconst.unsqueeze_(3)


            # convx = self.convolution_topology[i]["convolution_dims"][2]
            outputs.append(F.conv_transpose2d(reconst, getattr(self, f"{i}_W"),
                                              stride=self.convolution_topology[i]["stride"],
                                              padding=self.convolution_topology[i]["padding"],
                                              dilation=self.convolution_topology[i]["dilation"],
                                              output_padding=self.convolution_topology[i]["output_padding"]).squeeze(1))
            # outputs[-1] *= hidden_layer_W[iid] / total_weights
            # nonzero_masks.append((outputs[-1] != 0.).type(torch.get_default_dtype())) # * getattr(self, "hidden_layer_W")[iid])  # Used for calculating mean of outputs, don't want zeros to influence mean
            # outputs[-1] /= convx  # multiply by 10/k to normalize by convolution dimension
        if len(outputs) > 1:
            # Returns mean output from all hidden layers, zeros are ignored
            # mean_denominator = torch.sum(torch.stack(nonzero_masks), 0) + 1e-6
            return torch.sum(torch.stack(outputs), 0)  # / mean_denominator
        else:
            return outputs[0]

    ## Gibbs Sampling of Potts Visbile Layer
    def sample_from_inputs_v(self, psi, beta=1):  # Psi ohe (Batch_size, v_num, q)   fields (self.v_num, self.q)
        datasize = psi.shape[0]

        if beta == 1:
            cum_probas = psi + getattr(self, "fields").unsqueeze(0)
        else:
            cum_probas = beta * psi + beta * getattr(self, "fields").unsqueeze(0) + (1 - beta) * getattr(self, "fields0").unsqueeze(0)

        cum_probas = self.cumulative_probabilities(cum_probas)

        rng = torch.rand((datasize, self.v_num), dtype=torch.float64, device=self.device)
        low = torch.zeros((datasize, self.v_num), dtype=torch.long, device=self.device)
        middle = torch.zeros((datasize, self.v_num), dtype=torch.long, device=self.device)
        high = torch.zeros((datasize, self.v_num), dtype=torch.long, device=self.device)
        high.fill_(self.q)

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

        return F.one_hot(high, self.q)
        # return self.one_hot_tmp.scatter(2, high.unsqueeze(-1), 1)

    ## Gibbs Sampling of dReLU hidden layer
    def sample_from_inputs_h(self, psi, nancheck=False, beta=1):  # psi is a list of hidden [input]
        h_uks = []
        for iid, i in enumerate(self.hidden_convolution_keys):
            if beta == 1:
                a_plus = getattr(self, f'{i}_gamma+').unsqueeze(0)
                a_minus = getattr(self, f'{i}_gamma-').unsqueeze(0)
                theta_plus = getattr(self, f'{i}_theta+').unsqueeze(0)
                theta_minus = getattr(self, f'{i}_theta-').unsqueeze(0)
            else:
                theta_plus = (beta * getattr(self, f'{i}_theta+') + (1 - beta) * getattr(self, f'{i}_0theta+')).unsqueeze(0)
                theta_minus = (beta * getattr(self, f'{i}_theta-') + (1 - beta) * getattr(self, f'{i}_0theta-')).unsqueeze(0)
                a_plus = (beta * getattr(self, f'{i}_gamma+') + (1 - beta) * getattr(self, f'{i}_0gamma+')).unsqueeze(0)
                a_minus = (beta * getattr(self, f'{i}_gamma-') + (1 - beta) * getattr(self, f'{i}_0gamma-')).unsqueeze(0)
                psi[iid] *= beta

            if nancheck:
                nans = torch.isnan(psi[iid])
                if nans.max():
                    nan_unit = torch.nonzero(nans.max(0))[0]
                    print('NAN IN INPUT')
                    print('Hidden units', nan_unit)

            psi_plus = (-psi[iid]).add(theta_plus).div(torch.sqrt(a_plus))
            psi_minus = psi[iid].add(theta_minus).div(torch.sqrt(a_minus))
            # psi_plus = ((psi[iid][:, :, 1]).add(theta_plus).div(torch.sqrt(a_plus)))  # min pool
            # psi_minus = (psi[iid][:, :, 0].add(theta_minus).div(torch.sqrt(a_minus))) # max pool

            etg_plus = self.erf_times_gauss(psi_plus)  # Z+ * sqrt(a+)
            etg_minus = self.erf_times_gauss(psi_minus)  # Z- * sqrt(a-)

            p_plus = 1 / (1 + (etg_minus / torch.sqrt(a_minus)) / (etg_plus / torch.sqrt(a_plus)))  # p+ 1 / (1 +( (Z-/sqrt(a-))/(Z+/sqrt(a+))))    =   (Z+/(Z++Z-)
            nans = torch.isnan(p_plus)

            if True in nans:
                p_plus[nans] = torch.tensor(1.) * (torch.abs(psi_plus[nans]) > torch.abs(psi_minus[nans]))
            p_minus = 1 - p_plus

            is_pos = torch.rand(p_plus.shape, device=self.device) < p_plus

            rmax = torch.zeros(p_plus.shape, device=self.device)
            rmin = torch.zeros(p_plus.shape, device=self.device)
            rmin[is_pos] = torch.erf(psi_plus[is_pos].mul(self.invsqrt2))  # Part of Phi(x)
            rmax[is_pos] = 1  # pos values rmax set to one
            rmin[~is_pos] = -1  # neg samples rmin set to -1
            rmax[~is_pos] = torch.erf((-psi_minus[~is_pos]).mul(self.invsqrt2))  # Part of Phi(x)

            h = torch.zeros(psi[iid].shape, dtype=torch.float64, device=self.device)
            tmp = (rmax - rmin > 1e-14)
            h = self.sqrt2 * torch.erfinv(rmin + (rmax - rmin) * torch.rand(h.shape, device=self.device))
            h[is_pos] -= psi_plus[is_pos]
            h[~is_pos] += psi_minus[~is_pos]
            h /= torch.sqrt(is_pos * a_plus + ~is_pos * a_minus)
            h[torch.isinf(h) | torch.isnan(h) | ~tmp] = 0
            h_uks.append(h)
        return h_uks

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
        m = torch.zeros_like(X, device=self.device)
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
        m = torch.zeros_like(X, device=self.device)
        tmp = X < 4
        m[tmp] = 0.5 * X[tmp] ** 2 + torch.log(1 - torch.erf(X[tmp] / self.sqrt2)) + self.logsqrtpiover2
        m[~tmp] = - torch.log(X[~tmp]) + torch.log(1 - 1 / X[~tmp] ** 2 + 3 / X[~tmp] ** 4)
        return m

    ###################################################### Sampling Functions
    ## Samples hidden from visible and vice versa, returns newly sampled hidden and visible
    def markov_step(self, v, beta=1):
        # Gibbs Sampler
        h = self.sample_from_inputs_h(self.compute_output_v(v), beta=beta)
        return self.sample_from_inputs_v(self.compute_output_h(h), beta=beta), h

    def markov_PT_and_exchange(self, v, h, e, N_PT):
        for i, beta in zip(torch.arange(N_PT), self.betas):
            v[i], htmp = self.markov_step(v[i], beta=beta)
            for hid in range(self.h_layer_num):
                h[hid][i] = htmp[hid]
            e[i] = self.energy(v[i], h, hidden_sub_index=i)

        if self.record_swaps:
            particle_id = torch.arange(N_PT).unsqueeze(1).expand(N_PT, v.shape[1])

        betadiff = self.betas[1:] - self.betas[:-1]
        for i in np.arange(self.count_swaps % 2, N_PT - 1, 2):
            proba = torch.exp(betadiff[i] * e[i + 1] - e[i]).minimum(torch.ones_like(e[i]))
            swap = torch.rand(proba.shape[0], device=self.device) < proba
            if i > 0:
                v[i:i + 2, swap] = torch.flip(v[i - 1: i + 1], [0])[:, swap]
                for hid in range(self.h_layer_num):
                    h[hid][i:i + 2, swap] = torch.flip(h[hid][i - 1: i + 1], [0])[:, swap]
                # h[i:i + 2, swap] = torch.flip(h[i - 1: i + 1], [0])[:, swap]
                e[i:i + 2, swap] = torch.flip(e[i - 1: i + 1], [0])[:, swap]
                if self.record_swaps:
                    particle_id[i:i + 2, swap] = torch.flip(particle_id[i - 1: i + 1], [0])[:, swap]
            else:
                v[i:i + 2, swap] = torch.flip(v[:i + 1], [0])[:, swap]
                for hid in range(self.h_layer_num):
                    h[hid][i:i + 2, swap] = torch.flip(h[hid][:i + 1], [0])[:, swap]
                e[i:i + 2, swap] = torch.flip(e[:i + 1], [0])[:, swap]
                if self.record_swaps:
                    particle_id[i:i + 2, swap] = torch.flip(particle_id[:i + 1], [0])[:, swap]

            if self.record_acceptance:
                self.acceptance_rates[i] = swap.type(torch.get_default_dtype()).mean()
                self.mav_acceptance_rates[i] = self.mavar_gamma * self.mav_acceptance_rates[i] + self.acceptance_rates[
                    i] * (1 - self.mavar_gamma)

        if self.record_swaps:
            self.particle_id.append(particle_id)

        self.count_swaps += 1
        return v, h, e

    ## Markov Step for Parallel Tempering
    # def markov_step_PT(self, v, h, e, N_PT):
    #     for i, beta in zip(torch.arange(N_PT), self.betas):
    #         v[i], htmp = self.markov_step(v[i], beta=beta)
    #         for kid, k in self.hidden_convolution_keys:
    #             h[kid][i] = htmp[kid]
    #         e[i] = self.energy(v[i], h, hidden_sub_index=i)
    #     return v, h, e
    #
    # def exchange_step_PT(self, v, h, e, N_PT, compute_energy=True):
    #     if compute_energy:
    #         for i in torch.arange(N_PT):
    #             e[i] = self.energy(v[i], h, hidden_sub_index=i)
    #
    #     if self.record_swaps:
    #         particle_id = torch.arange(N_PT).unsqueeze(1).expand(N_PT, v.shape[1])
    #
    #     betadiff = self.betas[1:] - self.betas[:-1]
    #     for i in np.arange(self.count_swaps % 2, N_PT - 1, 2):
    #         proba = torch.exp(betadiff[i] * e[i + 1] - e[i]).minimum(torch.ones_like(e[i]))
    #         swap = torch.rand(proba.shape[0], device=self.device) < proba
    #         if i > 0:
    #             v[i:i + 2, swap] = torch.flip(v[i - 1: i + 1], [0])[:, swap]
    #             for hid in range(self.h_layer_num):
    #                 h[hid][i:i + 2, swap] = torch.flip(h[hid][i - 1: i + 1], [0])[:, swap]
    #             e[i:i + 2, swap] = torch.flip(e[i - 1: i + 1], [0])[:, swap]
    #             if self.record_swaps:
    #                 particle_id[i:i + 2, swap] = torch.flip(particle_id[i - 1: i + 1], [0])[:, swap]
    #         else:
    #             v[i:i + 2, swap] = torch.flip(v[:i + 1], [0])[:, swap]
    #             for hid in range(self.h_layer_num):
    #                 h[hid][i:i + 2, swap] = torch.flip(h[hid][:i + 1], [0])[:, swap]
    #             e[i:i + 2, swap] = torch.flip(e[:i + 1], [0])[:, swap]
    #             if self.record_swaps:
    #                 particle_id[i:i + 2, swap] = torch.flip(particle_id[:i + 1], [0])[:, swap]
    #
    #         if self.record_acceptance:
    #             self.acceptance_rates[i] = swap.type(torch.get_default_dtype()).mean()
    #             self.mav_acceptance_rates[i] = self.mavar_gamma * self.mav_acceptance_rates[i] + self.acceptance_rates[i] * (1 - self.mavar_gamma)
    #
    #     if self.record_swaps:
    #         self.particle_id.append(particle_id)
    #
    #     self.count_swaps += 1
    #     return v, h, e

    ######################################################### Pytorch Lightning Functions
    # Clamps hidden potential values to acceptable range
    def on_before_zero_grad(self, optimizer):
        with torch.no_grad():
            for key in self.hidden_convolution_keys:
                for param in ["gamma+", "gamma-"]:
                    getattr(self, f"{key}_{param}").data.clamp_(0.05, 1.0)
                for param in ["theta+", "theta-"]:
                    getattr(self, f"{key}_{param}").data.clamp_(0.0, 1.0)
                getattr(self, f"{key}_W").data.clamp_(-1.0, 1.0)

    ## Loads Data to be trained from provided fasta file
    def setup(self, stage=None):
        self.additional_data = False
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
                seqs, seq_read_counts, all_chars, q_data = fasta_read(file, self.molecule, drop_duplicates=False, threads=threads)
            except IOError:
                print(f"Provided Fasta File '{file}' Not Found")
                print(f"Current Directory '{os.getcwd()}'")
                exit()

            if q_data != self.q:
                print(
                    f"State Number mismatch! Expected q={self.q}, in dataset q={q_data}. All observed chars: {all_chars}")
                exit(-1)

            data = pd.DataFrame(data={'sequence': seqs, 'fasta_count': seq_read_counts})

            if type(self.weights) == str and "fasta" in self.weights:
                weights = np.asarray(seq_read_counts)
                data["seq_count"] = weights
            #     data = pd.DataFrame(data={'sequence': seqs, 'seq_count': weights})
            # else:
            #     data = pd.DataFrame(data={'sequence': seqs})

            data_pds.append(data)

        all_data = pd.concat(data_pds)
        if type(self.weights) is np.ndarray:
            all_data["seq_count"] = self.weights

        assert len(all_data["sequence"][0]) == self.v_num  # make sure v_num is same as data_length

        # stratify_labels = None
        # if self.stratify or self.pearson_xvar == "label" or self.sampling_strategy == "stratified":

        # w8s = all_data.seq_count.to_numpy()
        labels = label_samples(all_data["fasta_count"], self.label_spacing, self.label_groups)

        all_data["label"] = labels
        # stratify_labels = labels

        # else:
        #     all_data["label"] = 0.


        train_sets, val_sets, test_sets = [], [], []
        for i in range(self.label_groups):
            label_df = all_data[all_data["label"] == i]
            if self.test_size > 0.:
                # Split label df into train and test sets, taking into account duplicates
                train_inds, test_inds = next(GroupShuffleSplit(test_size=self.test_size, n_splits=1, random_state=self.seed).split(label_df, groups=label_df['sequence']))
                test_sets += label_df.index[test_inds].to_list()

                # Further split training set into train and test set
                train_inds, val_inds = next(GroupShuffleSplit(test_size=self.validation_size, n_splits=1, random_state=self.seed).split(label_df[train_inds], groups=label_df['sequence']))
                train_sets += label_df.index[train_inds].to_list()
                val_sets += label_df.index[val_inds].to_list()

            else:
                # Split label df into train and validation sets, taking into account duplicates
                train_inds, val_inds = next(GroupShuffleSplit(test_size=self.validation_size, n_splits=1, random_state=self.seed).split(label_df, groups=label_df['sequence']))
                train_sets += label_df.index[train_inds].to_list()
                val_sets += label_df.index[val_inds].to_list()

        self.training_data = all_data.iloc[train_sets]
        self.validation_data = all_data.iloc[val_sets]

        if self.sampling_weights is not None:
            self.sampling_weights = self.sampling_weights[train_sets]

        self.dataset_indices = {"train_indices": train_sets, "val_indices": val_sets}
        if self.test_size > 0:
            self.test_data = all_data.iloc[test_sets]
            self.dataset_indices["test_indices"] = test_sets


        # self.training_data, self.validation_data = train_test_split(available_data, test_size=self.validation_size, stratify=stratify_labels, random_state=self.seed)
        #
        # self.dataset_indices = {"train_indices": self.training_data.index.to_list(), "val_indices": self.validation_data.index.to_list()}
        # if self.test_size > 0:
        #     self.dataset_indices ["test_indices"] = self.test_data.index.tolist()


        # if self.weights == "fasta_processed":
        #     all_data["seq_count"] = weight_transform(all_data.seq_count.to_list(), exponent_base=3, exponent_min=-9, exponent_max=1)
        #
        #     self.additional_data = True
        #     all_data["additional_data"] = pearson_transform(all_data.seq_count.to_list())

        # if self.test_size > 0.:
        #     available_data, self.test_data = train_test_split(all_data, test_size=self.test_size, stratify=stratify_labels, random_state=self.seed)
        # else:
        #     available_data = all_data
        #
        # self.training_data, self.validation_data = train_test_split(available_data, test_size=self.validation_size, stratify=stratify_labels, random_state=self.seed)
        #
        # self.dataset_indices = {"train_indices": self.training_data.index.to_list(), "val_indices": self.validation_data.index.to_list()}
        # if self.test_size > 0:
        #     self.dataset_indices ["test_indices"] = self.test_data.index.tolist()


    def on_train_start(self):
        # Log which sequences belong to each dataset
        with open(self.logger.log_dir + "/dataset_indices.json", "w") as f:
            json.dump(self.dataset_indices, f)

    ## Sets Up Optimizer as well as Exponential Weight Decasy
    def configure_optimizers(self):
        optim = self.optimizer(self.parameters(), lr=self.lr, weight_decay=self.wd)
        # optim = self.optimizer(self.weight_param)
        # Exponential Weight Decay after set amount of epochs (set by decay_after)
        decay_gamma = (self.lrf / self.lr) ** (1 / (self.epochs * (1 - self.decay_after)))
        decay_milestone = math.floor(self.decay_after * self.epochs)
        my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=[decay_milestone], gamma=decay_gamma)
        # my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=decay_gamma)
        optim_dict = {"lr_scheduler": my_lr_scheduler,
                      "optimizer": optim}
        return optim_dict

    ## Loads Training Data
    def train_dataloader(self, init_fields=True):
        # Get Correct Weights
        training_weights = None
        if "seq_count" in self.training_data.columns:
            training_weights = self.training_data["seq_count"].tolist()

        # additional_data = None
        # if "additional_data" in self.training_data.columns:
        #     additional_data = self.training_data["additional_data"].tolist()


        labels = False
        if self.pearson_xvar == "labels":
            labels = True

        train_reader = Categorical(self.training_data, self.q, weights=training_weights, max_length=self.v_num,
                                   molecule=self.molecule, device=self.device, one_hot=True, labels=labels, additional_data=None)

        # initialize fields from data
        if init_fields:
            with torch.no_grad():
                initial_fields = train_reader.field_init()
                self.fields += initial_fields
                self.fields0 += initial_fields

        shuffle = True
        if self.sample_type == "pcd":
            shuffle = False

        if self.sampling_strategy == "stratified":
            return torch.utils.data.DataLoader(
                train_reader,
                batch_sampler=StratifiedBatchSampler(self.training_data["label"].to_numpy(), batch_size=self.batch_size, shuffle=shuffle),
                num_workers=self.worker_num,  # Set to 0 if debug = True
                pin_memory=self.pin_mem
            )
        elif self.sampling_strategy == "weighted":
            return torch.utils.data.DataLoader(
                train_reader,
                sampler=WeightedRandomSampler(weights=self.sampling_weights, num_samples=self.batch_size*self.sample_multiplier, replacement=True),
                num_workers=self.worker_num,  # Set to 0 if debug = True
                batch_size=self.batch_size,
                pin_memory=self.pin_mem
            )
        elif self.sampling_strategy == "stratified_weighted":
            return torch.utils.data.DataLoader(
                train_reader,
                batch_sampler=WeightedSubsetRandomSampler(self.sampling_weights, self.training_data["label"].to_numpy(), self.group_fraction, self.batch_size, self.sample_multiplier),
                num_workers=self.worker_num,  # Set to 0 if debug = True
                pin_memory=self.pin_mem
            )
        else:
            self.sampling_strategy = "random"
            return torch.utils.data.DataLoader(
                train_reader,
                batch_size=self.batch_size,
                num_workers=self.worker_num,  # Set to 0 if debug = True
                pin_memory=self.pin_mem,
                shuffle=shuffle
            )


    def val_dataloader(self):
        # Get Correct Validation weights
        validation_weights = None
        if "seq_count" in self.validation_data.columns:
            validation_weights = self.validation_data["seq_count"].tolist()

        # additional_data = None
        # if "additional_data" in self.validation_data.columns:
        #     additional_data = self.validation_data["additional_data"].tolist()

        labels = False
        if self.pearson_xvar == "labels":
            labels = True

        val_reader = Categorical(self.validation_data, self.q, weights=validation_weights, max_length=self.v_num,
                                 molecule=self.molecule, device=self.device, one_hot=True, labels=labels, additional_data=None)

        if self.sampling_strategy == "stratified":
            return torch.utils.data.DataLoader(
                val_reader,
                batch_sampler=StratifiedBatchSampler(self.validation_data["label"].to_numpy(), batch_size=self.batch_size, shuffle=False),
                num_workers=self.worker_num,  # Set to 0 if debug = True
                pin_memory=self.pin_mem
            )
        else:
            return torch.utils.data.DataLoader(
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
        if self.pearson_xvar == "labels" and self.additional_data:
            seqs, one_hot, seq_weights, labels, additional_data = batch
        if self.pearson_xvar == "labels":
            seqs, one_hot, seq_weights, labels = batch
        if self.additional_data:
            seqs, one_hot, seq_weights, additional_data = batch
        else:
            seqs, one_hot, seq_weights = batch

        # pseudo_likelihood = (self.pseudo_likelihood(one_hot) * seq_weights).sum() / seq_weights.sum()
        free_energy = self.free_energy(one_hot)
        free_energy_avg = (free_energy * seq_weights).sum() / seq_weights.abs().sum()

        if self.use_pearson:
            if self.pearson_xvar == "values":
                # correlation coefficient between free energy and fitness values
                vy = seq_weights - torch.mean(seq_weights)
            elif self.pearson_xvar == "labels":
                labels = labels.double()
                vy = labels - torch.mean(labels)

            vx = -1*(free_energy - torch.mean(free_energy))  # multiply be negative one so lowest free energy vals get paired with the highest copy number/fitness values or with higher label values
            pearson_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + 1e-6) * torch.sqrt(torch.sum(vy ** 2) + 1e-6))


        batch_out = {
            # "val_pseudo_likelihood": pseudo_likelihood.detach()
            "val_free_energy": free_energy_avg.detach()
        }

        if self.use_pearson:
            batch_out["val_pearson_corr"] = pearson_correlation.detach()
            self.log("ptl/val_pearson_corr", batch_out["val_pearson_corr"], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)

        # logging on step, for whatever reason allocates 512 bytes on gpu after every epoch.
        self.log("ptl/val_free_energy", batch_out["val_free_energy"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #
        return batch_out

    def validation_epoch_end(self, outputs):
        result_dict = {}
        for key, value in outputs[0].items():
            result_dict[key] = torch.stack([x[key] for x in outputs]).mean()

        # avg_pl = torch.stack([x['val_pseudo_likelihood'] for x in outputs]).mean()
        # self.logger.experiment.add_scalar("Validation pseudo_likelihood", avg_pl, self.current_epoch)
        # avg_fe = torch.stack([x['val_free_energy'] for x in outputs]).mean()
        #
        # scalars = {"Validation Free Energy": avg_fe}
        # if self.use_pearson:
        #     avg_pearson = torch.stack([x['val_pearson_corr'] for x in outputs]).mean()
        #     scalars["val_pearson_corr"] = avg_pearson

        self.logger.experiment.add_scalars("Val Scalars", result_dict, self.current_epoch)

    ## On Epoch End Collects Scalar Statistics and Distributions of Parameters for the Tensorboard Logger
    def training_epoch_end(self, outputs):
        result_dict = {}
        for key, value in outputs[0].items():
            if key == "loss":
                result_dict[key] = torch.stack([x[key].detach() for x in outputs]).mean()
            else:
                result_dict[key] = torch.stack([x[key] for x in outputs]).mean()


        # avg_loss = torch.stack([x['loss'].detach() for x in outputs]).mean()
        # avg_dF = torch.stack([x["free_energy_diff"] for x in outputs]).mean()
        # field_reg = torch.stack([x["field_reg"] for x in outputs]).mean()
        # weight_reg = torch.stack([x["weight_reg"] for x in outputs]).mean()
        # distance_reg = torch.stack([x["distance_reg"] for x in outputs]).mean()
        # free_energy = torch.stack([x["train_free_energy"] for x in outputs]).mean()
        # # pseudo_likelihood = torch.stack([x['train_pseudo_likelihood'] for x in outputs]).mean()

        # all_scalars = {"Loss": avg_loss,
        #                "CD_Loss": avg_dF,
        #                "Field Reg": field_reg,
        #                "Weight Reg": weight_reg,
        #                "Distance Reg": distance_reg,
        #                # "Train_pseudo_likelihood": pseudo_likelihood,
        #                "Train Free Energy": free_energy,
        #                }
        #
        #
        # if self.use_pearson:
        #     pearson_corr = torch.stack([x["train_pearson_corr"] for x in outputs]).mean()
        #     pearson_loss = torch.stack([x["train_pearson_loss"] for x in outputs]).mean()
        #     all_scalars[f"Pearson Loss"] = pearson_loss
        #     all_scalars["Train Pearson Corr"] = pearson_corr

        self.logger.experiment.add_scalars("All Scalars", result_dict, self.current_epoch)

        for name, p in self.named_parameters():
            self.logger.experiment.add_histogram(name, p.detach(), self.current_epoch)

        if self.meminfo:
            # print("GPU Reserved:", torch.cuda.memory_reserved(0))
            print(f"GPU Allocated Mem Epcoh {self.current_epoch}:", torch.cuda.memory_allocated(0))

    ## Not yet rewritten for crbm
    def training_step_CD_energy(self, batch, batch_idx):
        seqs, one_hot, seq_weights = batch

        V_neg_oh, h_neg, V_pos_oh, h_pos = self(one_hot)

        # Calculate CD loss
        E_p = (self.energy(V_pos_oh, h_pos) * seq_weights/seq_weights.sum()).sum()   # energy of training data
        # E_p = (self.energy(V_pos_oh, h_pos) * weights).sum()  # energy of training data
        E_n = (self.energy(V_neg_oh, h_neg) * seq_weights/seq_weights.sum()).sum()   # energy of gibbs sampled visible states
        # E_n = (self.energy(V_neg_oh, h_neg) * weights).sum()  # energy of gibbs sampled visible states
        cd_loss = E_p - E_n

        # Regularization Terms
        reg1, reg2, reg3, bs_loss, gap_loss, reg_dict = self.regularization_terms()

        # Calculate Loss
        loss = cd_loss + reg1 + reg2 + reg3 + bs_loss + gap_loss

        logs = {"loss": loss,
                "free_energy_diff": cd_loss.detach(),
                "train_free_energy": E_p.detach(),
                **reg_dict
                }

        self.log("ptl/train_free_energy", logs["train_free_energy"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return logs

    def regularization_terms(self):
        reg1 = self.lf / (2 * self.v_num * self.q) * getattr(self, "fields").square().sum((0, 1))
        reg2 = torch.zeros((1,), device=self.device)
        reg3 = torch.zeros((1,), device=self.device)

        bs_loss = torch.zeros((1,), device=self.device)  # encourages weights to use both positive and negative contributions
        gap_loss = torch.zeros((1,), device=self.device)  # discourages high values for gaps

        for iid, i in enumerate(self.hidden_convolution_keys):
            W_shape = self.convolution_topology[i]["weight_dims"]  # (h_num,  input_channels, kernel0, kernel1)
            W = getattr(self, f"{i}_W")
            x = torch.sum(torch.abs(getattr(self, f"{i}_W")), (3, 2, 1)).square()
            reg2 += x.sum() * self.l1_2 / (2 * W_shape[1] * W_shape[2] * W_shape[3])
            reg3 += self.ld / ((getattr(self, f"{i}_W").abs() - getattr(self, f"{i}_W").squeeze(1).abs()).abs().sum((1, 2, 3)).mean() + 1)
            gap_loss += self.lgap * W[:, :, :, -1].abs().sum()

            denom = torch.sum(torch.abs(W), (3, 2, 1))
            zeroW = torch.zeros_like(W, device=self.device)
            Wpos = torch.maximum(W, zeroW)
            Wneg = torch.minimum(W, zeroW)
            bs_loss += self.lbs * torch.abs(Wpos.sum((1, 2, 3)) / denom - torch.abs(Wneg.sum((1, 2, 3))) / denom).sum()

        # Passed to training logger
        reg_dict = {
            "field_reg": reg1.detach(),
            "weight_reg": reg2.detach(),
            "distance_reg": reg3.detach(),
            "gap_reg": gap_loss.detach(),
            "both_side_reg": bs_loss.detach()
        }

        return reg1, reg2, reg3, bs_loss, gap_loss, reg_dict

    # Not yet rewritten for CRBM
    def training_step_PT_free_energy(self, batch, batch_idx):
        seqs, one_hot, seq_weights = batch

        V_neg_oh, h_neg, V_pos_oh, h_pos = self(one_hot)

        # delete tensors not involved in loss calculation
        # pytorch garbage collection does not catch these
        del h_neg
        del h_pos

        # Calculate CD loss
        F_v = (self.free_energy(V_pos_oh) * seq_weights).sum() / seq_weights.sum()  # free energy of training data
        F_vp = (self.free_energy(V_neg_oh) * seq_weights).sum() / seq_weights.sum()  # free energy of gibbs sampled visible states
        cd_loss = F_v - F_vp

        # Regularization Terms
        reg1, reg2, reg3, bs_loss, gap_loss, reg_dict = self.regularization_terms()

        # Calc loss
        loss = cd_loss + reg1 + reg2 + reg3 + bs_loss + gap_loss

        logs = {"loss": loss,
                "free_energy_diff": cd_loss.detach(),
                "train_free_energy": F_v.detach(),
                **reg_dict
                }

        self.log("ptl/train_free_energy", logs["train_free_energy"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return logs

    def training_step_CD_free_energy(self, batch, batch_idx):
        if self.pearson_xvar == "labels":
            seqs, one_hot, seq_weights, labels = batch
        else:
            seqs, one_hot, seq_weights = batch
        # if self.meminfo:
        #     print("GPU Allocated Training Step Start:", torch.cuda.memory_allocated(0))

        V_neg_oh, h_neg, V_pos_oh, h_pos = self(one_hot)

        # print("GPU Allocated After Forward:", torch.cuda.memory_allocated(0))

        # delete tensors not involved in loss calculation
        # pytorch garbage collection does not catch these
        del h_neg
        del h_pos

        # F_v = (self.free_energy(V_pos_oh) * weights).sum()  # free energy of training data
        free_energy = self.free_energy(V_pos_oh)
        F_v = (self.free_energy(V_pos_oh) * seq_weights/seq_weights.sum()).sum()  # free energy of training data
        # F_vp = (self.free_energy(V_neg_oh) * weights.abs()).sum() # free energy of gibbs sampled visible states
        F_vp = (self.free_energy(V_neg_oh) * seq_weights/seq_weights.sum()).sum()  # free energy of gibbs sampled visible states
        free_energy_diff = F_v - F_vp
        # cd_loss = (free_energy_diff/torch.abs(free_energy_diff)) * torch.log(torch.abs(free_energy_diff) + 10)
        cd_loss = free_energy_diff

        if self.use_pearson:
            if self.pearson_xvar == "values":
                # correlation coefficient between free energy and fitness values
                vy = seq_weights - torch.mean(seq_weights)
            elif self.pearson_xvar == "labels":
                labels = labels.double()
                vy = labels - torch.mean(labels)

            # correlation coefficient between free energy and fitness values/labels
            vx = -1 * (free_energy - torch.mean(free_energy))  # multiply be negative one so lowest free energy vals get paired with the highest copy number/fitness values

            pearson_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + 1e-6) * torch.sqrt(torch.sum(vy ** 2) + 1e-6))
            pearson_loss = (1 - pearson_correlation) # * (self.current_epoch/self.epochs + 1) * 10

        # if self.meminfo:
        #     print("GPU Allocated After CD_Loss:", torch.cuda.memory_allocated(0))

        # Regularization Terms
        reg1, reg2, reg3, bs_loss, gap_loss, reg_dict = self.regularization_terms()

        # Debugging
        # nancheck = torch.isnan(torch.tensor([cd_loss, F_v, F_vp, reg1, reg2, reg3], device=self.device))
        # if True in nancheck:
        #     print(nancheck)
        #     torch.save(V_pos_oh, "vpos_err.pt")
        #     torch.save(V_neg_oh, "vneg_err.pt")
        #     torch.save(one_hot, "oh_err.pt")
        #     torch.save(seq_weights, "seq_weights_err.pt")

        # Calculate Loss
        loss = cd_loss + reg1 + reg2 + reg3 + bs_loss + gap_loss

        logs = {"loss": loss,
                "free_energy_diff": cd_loss.detach(),
                "train_free_energy": F_v.detach(),
                **reg_dict
                }

        if self.use_pearson:
            loss += pearson_loss * 10  # * pearson_multiplier
            logs["loss"] = loss
            logs["train_pearson_corr"] = pearson_correlation.detach()
            logs["train_pearson_loss"] = pearson_loss.detach()
            self.log("ptl/train_pearson_corr", logs["train_pearson_corr"], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log("ptl/train_free_energy", logs["train_free_energy"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # if self.meminfo:
        #     print("GPU Allocated Final:", torch.cuda.memory_allocated(0))

        return logs

    def training_step_PCD_free_energy(self, batch, batch_idx):
        if self.pearson_xvar == "labels" and self.additional_data:
            seqs, one_hot, seq_weights, labels, additional_data = batch
        if self.pearson_xvar == "labels":
            seqs, one_hot, seq_weights, labels = batch
        if self.additional_data:
            seqs, one_hot, seq_weights, additional_data = batch
        else:
            seqs, one_hot, seq_weights = batch

        if self.current_epoch == 0 and batch_idx == 0:
            self.chain = [one_hot.detach()]
        elif self.current_epoch == 0:
            self.chain.append(one_hot.detach())

        V_oh_neg, h_neg = self.forward_PCD(batch_idx)

        # delete tensors not involved in loss calculation
        # pytorch garbage collection does not catch these
        del h_neg

        # psuedo likelihood actually minimized, loss sits around 0 but does it's own thing
        free_energy = self.free_energy(one_hot)
        F_v = (free_energy * seq_weights).sum() / seq_weights.sum()  # free energy of training data
        F_vp = (self.free_energy(V_oh_neg) * seq_weights).sum() / seq_weights.sum()  # free energy of gibbs sampled visible states
        cd_loss = F_v - F_vp  # Should Give same gradient as Tubiana Implementation minus the batch norm on the hidden unit activations

        if self.use_pearson:
            if self.pearson_xvar == "values":
                # correlation coefficient between free energy and fitness values
                if self.additional_data:
                    vy = additional_data - torch.mean(additional_data)
                else:
                    vy = seq_weights - torch.mean(seq_weights)

            elif self.pearson_xvar == "labels":
                labels = labels.double()
                vy = labels - torch.mean(labels)

            # correlation coefficient between free energy and fitness values/labels
            vx = -1 * (free_energy - torch.mean(free_energy))  # multiply be negative one so lowest free energy vals get paired with the highest copy number/fitness values

            pearson_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + 1e-6) * torch.sqrt(torch.sum(vy ** 2) + 1e-6))
            pearson_loss = (1 - pearson_correlation)  # * (self.current_epoch/self.epochs + 1) * 10


        # Regularization Terms
        reg1, reg2, reg3, bs_loss, gap_loss, reg_dict = self.regularization_terms()

        # Calculate Loss
        loss = cd_loss + reg1 + reg2 + reg3 + bs_loss + gap_loss

        logs = {"loss": loss,
                "free_energy_diff": cd_loss.detach(),
                "train_free_energy": F_v.detach(),
                **reg_dict
                }

        if self.use_pearson:
            loss += pearson_loss * 50  # * pearson_multiplier
            logs["loss"] = loss
            logs["train_pearson_corr"] = pearson_correlation.detach()
            logs["train_pearson_loss"] = pearson_loss.detach()
            self.log("ptl/train_pearson_corr", logs["train_pearson_corr"], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log("ptl/train_free_energy", logs["train_free_energy"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return logs


    def forward_PCD(self, batch_idx):
        # Gibbs sampling with Persistent Contrastive Divergence
        # pytorch lightning handles the device
        fantasy_v = self.chain[batch_idx]  # Last sample that was saved to self.chain variable, initialized in training step
        # h_pos = self.sample_from_inputs_h(self.compute_output_v(fantasy_v))
        # with torch.no_grad() # only use last sample for gradient calculation, may be helpful but honestly not the slowest thing rn
        for _ in range(self.mc_moves - 1):
            fantasy_v, fantasy_h = self.markov_step(fantasy_v)

        V_neg, fantasy_h = self.markov_step(fantasy_v)
        h_neg = self.sample_from_inputs_h(self.compute_output_v(V_neg))

        self.chain[batch_idx] = V_neg.detach()

        return V_neg, h_neg

    def forward(self, V_pos_ohe):

        # Trying this out
        # self.one_hot_tmp = torch.zeros_like(V_pos_ohe, device=self.device)

        if self.sample_type == "gibbs":
            # Gibbs sampling
            # pytorch lightning handles the device
            with torch.no_grad():  # only use last sample for gradient calculation, Enabled to minimize memory usage, hopefully won't have much effect on performance
                fantasy_v, fantasy_h = self.markov_step(V_pos_ohe)
                for _ in range(self.mc_moves - 2):
                    fantasy_v, fantasy_h = self.markov_step(fantasy_v)

            V_neg, fantasy_h = self.markov_step(fantasy_v)

            # V_neg, h_neg, V_pos, h_pos
            return V_neg, self.sample_from_inputs_h(self.compute_output_v(V_neg)), V_pos_ohe, self.sample_from_inputs_h(self.compute_output_v(V_pos_ohe))

        elif self.sample_type == "pt":
            # Initialize_PT is called before the forward function is called. Therefore, N_PT will be filled

            # Parallel Tempering
            n_chains = V_pos_ohe.shape[0]

            with torch.no_grad():
                fantasy_v = self.random_init_config_v(custom_size=(self.N_PT, n_chains))
                fantasy_h = self.random_init_config_h(custom_size=(self.N_PT, n_chains))
                fantasy_E = self.energy_PT(fantasy_v, fantasy_h, self.N_PT)

                for _ in range(self.mc_moves - 1):
                    fantasy_v, fantasy_h, fantasy_E = self.markov_PT_and_exchange(fantasy_v, fantasy_h, fantasy_E, self.N_PT)
                    self.update_betas(self.N_PT)

            fantasy_v, fantasy_h, fantasy_E = self.markov_PT_and_exchange(fantasy_v, fantasy_h, fantasy_E, self.N_PT)
            self.update_betas(self.N_PT)

            # V_neg, h_neg, V_pos, h_pos
            return fantasy_v[0], fantasy_h[0], V_pos_ohe, self.sample_from_inputs_h(self.compute_output_v(V_pos_ohe))

    # X must be a pandas dataframe with the sequences in string format under the column 'sequence'
    # Returns the likelihood for each sequence in an array
    def predict(self, X):
        # Read in data
        reader = Categorical(X, self.q, weights=None, max_length=self.v_num, molecule=self.molecule, device=self.device, one_hot=True)
        data_loader = torch.utils.data.DataLoader(
            reader,
            batch_size=self.batch_size,
            num_workers=self.worker_num,  # Set to 0 if debug = True
            pin_memory=self.pin_mem,
            shuffle=False
        )
        self.eval()
        with torch.no_grad():
            likelihood = []
            for i, batch in enumerate(data_loader):
                seqs, one_hot, seq_weights = batch
                likelihood += self.likelihood(one_hot).detach().tolist()

        return X.sequence.tolist(), likelihood

    # X must be a pandas dataframe with the sequences in string format under the column 'sequence'
    # Returns the saliency map for all sequences in X
    def saliency_map(self, X):
        reader = Categorical(X, self.q, weights=None, max_length=self.v_num, molecule=self.molecule, device=self.device, one_hot=True)
        data_loader = torch.utils.data.DataLoader(
            reader,
            batch_size=self.batch_size,
            num_workers=self.worker_num,  # Set to 0 if debug = True
            pin_memory=self.pin_mem,
            shuffle=False
        )
        saliency_maps = []
        self.eval()
        for i, batch in enumerate(data_loader):
            seqs, one_hot, seq_weights = batch
            one_hot_v = Variable(one_hot.type(torch.get_default_dtype()), requires_grad=True)
            V_neg, h_neg, V_pos, h_pos = self(one_hot_v)
            weights = seq_weights
            F_v = (self.free_energy(V_pos) * weights).sum() / weights.sum()  # free energy of training data
            F_vp = (self.free_energy(V_neg) * weights).sum() / weights.sum()  # free energy of gibbs sampled visible states
            cd_loss = F_v - F_vp  # Should Give same gradient as Tubiana Implementation minus the batch norm on the hidden unit activations

            # Regularization Terms
            reg1 = self.lf / 2 * getattr(self, "fields").square().sum((0, 1))
            reg2 = torch.zeros((1,), device=self.device)
            reg3 = torch.zeros((1,), device=self.device)
            for iid, i in enumerate(self.hidden_convolution_keys):
                W_shape = self.convolution_topology[i]["weight_dims"]  # (h_num,  input_channels, kernel0, kernel1)
                x = torch.sum(torch.abs(getattr(self, f"{i}_W")), (3, 2, 1)).square()
                reg2 += x.sum(0) * self.l1_2 / (2 * W_shape[1] * W_shape[2] * W_shape[3])
                # Size of Convolution Filters weight_size = (h_num, input_channels, kernel[0], kernel[1])
                reg3 += self.ld / ((getattr(self, f"{i}_W").abs() - getattr(self, f"{i}_W").squeeze(1).abs()).abs().sum((1, 2, 3)).mean() + 1)

            loss = cd_loss + reg1 + reg2 + reg3
            loss.backward()

            saliency_maps.append(one_hot_v.grad.data.detach())

        return torch.cat(saliency_maps, dim=0)

    def fill_gaps_in_parameters(self, fill=1e-6):
        with torch.no_grad():
            fields = getattr(self, "fields")
            fields[:, -1].fill_(fill)

            for iid, i in enumerate(self.hidden_convolution_keys):
                W = getattr(self, f"{i}_W")
                W[:, :, :, -1].fill_(fill)

    # disabled until I figure out pseudo likelihood function
    # def predict_psuedo(self, X):
    #     reader = RBMCaterogical(X, weights=None, max_length=self.v_num, base_to_id=self.molecule, device=self.device)
    #     data_loader = torch.utils.data.DataLoader(
    #         reader,
    #         batch_size=self.batch_size,
    #         num_workers=self.worker_num,  # Set to 0 if debug = True
    #         pin_memory=self.pin_mem,
    #         shuffle=False
    #     )
    #     with torch.no_grad():
    #         likelihood = []
    #         for i, batch in enumerate(data_loader):
    #             seqs, one_hot, seq_weights = batch
    #             likelihood += self.pseudo_likelihood(one_hot).detach().tolist()
    #
    #     return X.sequence.tolist(), likelihood

    # Return param as a numpy array
    def get_param(self, param_name):
        try:
            tensor = getattr(self, param_name).clone()
            return tensor.detach().numpy()
        except KeyError:
            print(f"Key {param_name} not found")
            exit()

    def update_betas(self, N_PT, beta=1, update_betas_lr=0.1, update_betas_lr_decay=1):
        with torch.no_grad():
            if N_PT < 3:
                return
            if self.acceptance_rates.mean() > 0:
                self.stiffness = torch.maximum(1 - (self.mav_acceptance_rates / self.mav_acceptance_rates.mean()), torch.zeros_like(self.mav_acceptance_rates)) + 1e-4 * torch.ones(N_PT - 1)
                diag = self.stiffness[0:-1] + self.stiffness[1:]
                if N_PT > 3:
                    offdiag_g = -self.stiffness[1:-1]
                    offdiag_d = -self.stiffness[1:-1]
                    M = torch.diag(offdiag_g, -1) + torch.diag(diag, 0) + torch.diag(offdiag_d, 1)
                else:
                    M = torch.diag(diag, 0)
                B = torch.zeros(N_PT - 2)
                B[0] = self.stiffness[0] * beta
                self.betas[1:-1] = self.betas[1:-1] * (1 - update_betas_lr) + update_betas_lr * torch.linalg.solve(M, B)
                update_betas_lr *= update_betas_lr_decay

    def AIS(self, M=10, n_betas=10000, batches=None, verbose=0, beta_type='adaptive'):
        with torch.no_grad():
            if beta_type == 'linear':
                betas = torch.arange(n_betas, device=self.device) / torch.tensor(n_betas - 1, dtype=torch.float64, device=self.device)
            elif beta_type == 'root':
                betas = torch.sqrt(torch.arange(n_betas, device=self.device)) / torch.tensor(n_betas - 1, dtype=torch.float64, device=self.device)
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
                    betas += list(sparse_betas[i] + (sparse_betas[i + 1] - sparse_betas[i]) * torch.arange(n_betas / (N_PT - 1), device=self.device) / (n_betas / (N_PT - 1) - 1))
                betas = torch.tensor(betas, device=self.device)
                # if verbose:
                # import matplotlib.pyplot as plt
                # plt.plot(betas); plt.title('Interpolating temperatures');plt.show()

            # Initialization.
            log_weights = torch.zeros(M, device=self.device)
            # config = self.gen_data(Nchains=M, Lchains=1, Nthermalize=0, beta=0)

            config = [self.sample_from_inputs_v(self.random_init_config_v(custom_size=(M,))),
                      self.sample_from_inputs_h(self.random_init_config_h(custom_size=(M,)))]

            log_Z_init = torch.zeros(1, device=self.device)

            log_Z_init += self.logpartition_h(self.random_init_config_h(custom_size=(1,), zeros=True), beta=0)
            log_Z_init += self.logpartition_v(self.random_init_config_v(custom_size=(1,), zeros=True), beta=0)

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

    def cgf_from_inputs_h(self, I, hidden_key):
        with torch.no_grad():
            B = I.shape[0]

            if hidden_key not in self.hidden_convolution_keys:
                print(f"Hidden Convolution Key {hidden_key} not found!")
                exit(-1)

            Wdims = self.convolution_topology[hidden_key]["weight_dims"]

            out = torch.zeros_like(I, device=self.device)

            sqrt_gamma_plus = torch.sqrt(getattr(self, f"{hidden_key}_gamma+")).expand(B, -1)
            sqrt_gamma_minus = torch.sqrt(getattr(self, f"{hidden_key}_gamma-")).expand(B, -1)
            log_gamma_plus = torch.log(getattr(self, f"{hidden_key}_gamma+")).expand(B, -1)
            log_gamma_minus = torch.log(getattr(self, f"{hidden_key}_gamma-")).expand(B, -1)

            Z_plus = -self.log_erf_times_gauss((-I + getattr(self, f'{hidden_key}_theta+').expand(B, -1)) / sqrt_gamma_plus) - 0.5 * log_gamma_plus
            Z_minus = self.log_erf_times_gauss((I + getattr(self, f'{hidden_key}_theta-').expand(B, -1)) / sqrt_gamma_minus) - 0.5 * log_gamma_minus
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
            self.initialize_PT(N_PT, n_chains=Nchains, record_acceptance=record_acceptance, record_swaps=record_swaps)

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

                # if record_acceptance:
                #     self.mavar_gamma = 0.95

                if update_betas:
                    record_acceptance = True
                    # self.update_betas_lr = 0.1
                    # self.update_betas_lr_decay = 1
            else:
                record_acceptance = False
                update_betas = False

            if (N_PT > 1) and record_replica:
                visible_data = self.random_init_config_v(custom_size=(Nchains, N_PT, Lchains), zeros=True)
                hidden_data = self.random_init_config_h(custom_size=(Nchains, N_PT, Lchains), zeros=True)
                data = [visible_data, hidden_data]
            else:
                visible_data = self.random_init_config_v(custom_size=(Nchains, Lchains), zeros=True)
                hidden_data = self.random_init_config_h(custom_size=(Nchains, Lchains), zeros=True)
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
                    config_init = [config_init[0][batches * i:batches * (i + 1)], config_init[1][batches * i:batches * (i + 1)]]
                    config = self._gen_data(Nthermalize, Ndata, Nstep, N_PT=N_PT, batches=batches, reshape=False, beta=beta,
                                            record_replica=record_replica, config_init=config_init, record_acceptance=record_acceptance,
                                            update_betas=update_betas, record_swaps=record_swaps)

                if (N_PT > 1) & record_replica:
                    data[0][batches * i:batches * (i + 1)] = torch.swapaxes(config[0], 0, 2).clone()
                    for hid in range(self.h_layer_num):
                        data[1][hid][batches * i:batches * (i + 1)] = torch.swapaxes(config[1][hid], 0, 2).clone()
                else:
                    data[0][batches * i:batches * (i + 1)] = torch.swapaxes(config[0], 0, 1).clone()
                    for hid in range(self.h_layer_num):
                        data[1][hid][batches * i:batches * (i + 1)] = torch.swapaxes(config[1][hid], 0, 1).clone()

            if reshape:
                return [data[0].flatten(0, -3), [hd.flatten(0, -3) for hd in data[1]]]
            else:
                return data

    def _gen_data(self, Nthermalize, Ndata, Nstep, N_PT=1, batches=1, reshape=True, config_init=[], beta=1, record_replica=False, record_acceptance=True, update_betas=False, record_swaps=False):
        with torch.no_grad():

            if N_PT > 1:
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

            if config_init != []:
                config = config_init
            else:
                if N_PT > 1:
                    config = [self.random_init_config_v(custom_size=(N_PT, batches)), self.random_init_config_h(custom_size=(N_PT, batches))]
                else:
                    config = [self.random_init_config_v(custom_size=(batches,)), self.random_init_config_h(custom_size=(batches,))]

            for _ in range(Nthermalize):
                if N_PT > 1:
                    energy = self.energy_PT(config[0], config[1], N_PT)
                    config[0], config[1], energy = self.markov_PT_and_exchange(config[0], config[1], energy, N_PT)
                    if update_betas:
                        self.update_betas(N_PT, beta=beta)
                else:
                    config[0], config[1] = self.markov_step(config[0], beta=beta)

            if N_PT > 1:
                if record_replica:
                    data = [config[0].clone().unsqueeze(0), self.clone_h(config[1], expand_dims=[0])]
                else:
                    data = [config[0][0].clone().unsqueeze(0), self.clone_h(config[1], expand_dims=[0], sub_index=0)]
            else:
                data = [config[0].clone().unsqueeze(0), self.clone_h(config[1], expand_dims=[0])]

            if N_PT > 1:
                if Ndata > 1:
                    if record_replica:
                        data_gen_v = self.random_init_config_v(custom_size=(Ndata, N_PT, batches), zeros=True)
                        data_gen_h = self.random_init_config_h(custom_size=(Ndata, N_PT, batches), zeros=True)
                        data_gen_v[0] = config[0].clone()

                        clone = self.clone_h(config[1])
                        for hid in range(self.h_layer_num):
                            data_gen_h[hid][0] = clone[hid]
                    else:
                        data_gen_v = self.random_init_config_v(custom_size=(Ndata, batches), zeros=True)
                        data_gen_h = self.random_init_config_h(custom_size=(Ndata, batches), zeros=True)
                        data_gen_v[0] = config[0][0].clone()

                        clone = self.clone_h(config[1], sub_index=0)
                        for hid in range(self.h_layer_num):
                            data_gen_h[hid][0] = clone[hid]
            else:
                data_gen_v = self.random_init_config_v(custom_size=(Ndata, batches), zeros=True)
                data_gen_h = self.random_init_config_h(custom_size=(Ndata, batches), zeros=True)
                data_gen_v[0] = config[0].clone()

                clone = self.clone_h(config[1])
                for hid in range(self.h_layer_num):
                    data_gen_h[hid][0] = clone[hid]

            for n in range(Ndata - 1):
                for _ in range(Nstep):
                    if N_PT > 1:
                        energy = self.energy_PT(config[0], config[1], N_PT)
                        config[0], config[1], energy = self.markov_PT_and_exchange(config[0], config[1], energy, N_PT)
                        if update_betas:
                            self.update_betas(N_PT, beta=beta)
                    else:
                        config[0], config[1] = self.markov_step(config[0], beta=beta)

                if N_PT > 1 and Ndata > 1:
                    if record_replica:
                        data_gen_v[n + 1] = config[0].clone()

                        clone = self.clone_h(config[1])
                        for hid in range(self.h_layer_num):
                            data_gen_h[hid][n + 1] = clone[hid]

                    else:
                        data_gen_v[n + 1] = config[0][0].clone()

                        clone = self.clone_h(config[1], sub_index=0)
                        for hid in range(self.h_layer_num):
                            data_gen_h[hid][n + 1] = clone[hid]

                else:
                    data_gen_v[n + 1] = config[0].clone()

                    clone = self.clone_h(config[1])
                    for hid in range(self.h_layer_num):
                        data_gen_h[hid][n + 1] = clone[hid]

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
                data[0] = data[0].flatten(0, -3)
                data[1] = [hd.flatten(0, -3) for hd in data[1]]
            else:
                data[0] = data[0]
                data[1] = data[1]

            return data
