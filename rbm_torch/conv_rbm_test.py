import time
import pandas as pd
from rbm import RBMCaterogical
import math
import numpy as np
# import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt  # just for confusion matrix generation

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW, Adagrad  # Supported Optimizers
import multiprocessing  # Just to set the worker number

import configs
from rbm_utils import aadict, dnadict, rnadict, Sequence_logo_all, fasta_read

def conv2d_dim(input_shape, conv_topology):
    [batch_size, input_channels, v_num, q] = input_shape
    stride = conv_topology["stride"]
    padding = conv_topology["padding"]
    kernel = conv_topology["kernel"]
    dilation = conv_topology["dilation"]
    h_num = conv_topology["number"]

    # Copied From https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    convx_num = int(math.floor((v_num + padding[0]*2 - dilation[0] * (kernel[0]-1) - 1)/stride[0] + 1))
    convy_num = int(math.floor((q + padding[1] * 2 - dilation[1] * (kernel[1]-1) - 1)/stride[1] + 1))  # most configurations will set this to 1

    # Copied from https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    recon_x = (convx_num - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel[0] - 1) + 1
    recon_y = (convy_num - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel[1] - 1) + 1

    # Pad for the unsampled visible units (depends on stride, and tensor size)
    output_padding = (v_num - recon_x, q - recon_y)

    # Size of Convolution Filters
    weight_size = (h_num, input_channels, kernel[0], kernel[1])

    # Size of Hidden unit Inputs h_uk
    conv_output_size = (batch_size, h_num, convx_num, convy_num)

    return {"weight_shape": weight_size, "conv_shape": conv_output_size, "output_padding": output_padding}



 # config["convolution_topology"] = {
 #        "hidden1": {"number": 5, "kernel": (9, config["q"]), "stride": (3, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
 #        "hidden2": {"number": 5, "kernel": (7, config["q"]), "stride": (5, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
 #        "hidden3": {"number": 5, "kernel": (3, config["q"]), "stride": (2, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
 #        "hidden4": {"number": 5, "kernel": (config["v_num"], config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
 #    }



class hidden:
    def __init__(self, convolution_topology, hidden_keys, datalengths, q):
        conv_top = {}
        data = {}
        for iid, i in enumerate(hidden_keys):
            conv_top[iid] = convolution_topology[i]

            conv_top[iid]["weight_dims"] = {}
            conv_top[iid]["conv_dims"] = {}
            for dl in datalengths:
                example_input = (50, 1, dl, q)
                dims = conv2d_dim(example_input, conv_top[iid])
                conv_top[iid]["weight_dims"][dl] = dims["weight_shape"]
                conv_top[iid]["conv_shape"][dl] = dims["conv_shape"]
                conv_top[iid]["output_padding"][dl] = dims["output_padding"]

        self.hidden_params = conv_top










class CRBM(LightningModule):
    def __init__(self, config, debug=False):
        super().__init__()
        # self.h_num = config['h_num']  # Number of hidden node clusters, can be variable
        self.v_num = config['v_num']   # Number of visible nodes
        self.q = config['q']  # Number of categories the input sequence has (ex. DNA:4 bases + 1 gap)
        self.mc_moves = config['mc_moves']  # Number of MC samples to take to update hidden and visible configurations
        self.batch_size = config['batch_size']  # Pretty self explanatory

        self.epsilon = 1e-6  # Number added to denominators for numerical stability
        self.epochs = config['epochs'] # number of training iterations, needed for our weight decay function

        # Data Input
        self.fasta_file = config['fasta_file']
        self.molecule = config['molecule'] # can be protein, rna or dna currently
        assert self.molecule in ["dna", "rna", "protein"]

        try:
            self.worker_num = config["data_worker_num"]
        except KeyError:
            if debug:
                self.worker_num = 0
            else:
                self.worker_num = multiprocessing.cpu_count()

        # Sets Pim Memory when GPU is being used
        if hasattr(self, "trainer"):
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
        self.ld = config['ld']
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

        ## Might Need if grad values blow up
        # self.grad_norm_clip_value = 1000 # i have no context for setting this value at all lol, it isn't in use currently but may be later

        # Pytorch Basic Options
        torch.manual_seed(self.seed)  # For reproducibility
        torch.set_default_dtype(torch.float64)  # Double Precision

        self.convolution_topology = config["convolution_topology"]

        # Parameters that shouldn't change
        self.params = nn.ParameterDict({
            # visible layer parameters
            'fields': nn.Parameter(torch.zeros((self.v_num, self.q), device=self.device)),
            'fields0': nn.Parameter(torch.zeros((self.v_num, self.q), device=self.device), requires_grad=False)
        })

        self.hidden_convolution_keys = self.convolution_topology.keys()
        self.h_layer_num = len(self.hidden_convolution_keys)
        for key in self.hidden_convolution_keys:
            # Set information about the convolutions that will be useful
            dims = conv2d_dim([self.batch_size, 1, self.v_num, self.q], self.convolution_topology[key])
            self.convolution_topology[key]["weight_dims"] = dims["weight_shape"]
            self.convolution_topology[key]["convolution_dims"] = dims["conv_shape"]
            self.convolution_topology[key]["output_padding"] = dims["output_padding"]
            # Convolution Weights
            self.params[f"{key}_W"] = nn.Parameter(self.weight_intial_amplitude * torch.randn(self.convolution_topology[key]["weight_dims"], device=self.device))
            # hidden layer parameters
            self.params[f'{key}_theta+'] = nn.Parameter(torch.zeros(self.convolution_topology[key]["number"], device=self.device))
            self.params[f'{key}_theta-'] = nn.Parameter(torch.zeros(self.convolution_topology[key]["number"], device=self.device))
            self.params[f'{key}_gamma+'] = nn.Parameter(torch.ones(self.convolution_topology[key]["number"], device=self.device))
            self.params[f'{key}_gamma-'] = nn.Parameter(torch.ones(self.convolution_topology[key]["number"], device=self.device))
            # Used in PT Sampling / AIS
            self.params[f'{key}_0theta+'] = nn.Parameter(torch.zeros(self.convolution_topology[key]["number"], device=self.device))
            self.params[f'{key}_0theta-'] = nn.Parameter(torch.zeros(self.convolution_topology[key]["number"], device=self.device))
            self.params[f'{key}_0gamma+'] = nn.Parameter(torch.ones(self.convolution_topology[key]["number"], device=self.device))
            self.params[f'{key}_0gamma-'] = nn.Parameter(torch.ones(self.convolution_topology[key]["number"], device=self.device))


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
    # TODO: Figure this out
    # Currently gives wrong values
    # def psuedolikelihood(self, v):
    #     with torch.no_grad():
    #         categorical = v.argmax(2)
    #         ind_x = torch.arange(categorical.shape[0], dtype=torch.long, device=self.device)
    #         ind_y = torch.randint(self.v_num, (categorical.shape[0],), dtype=torch.long, device=self.device)  # high, shape tuple, needs size, low=0 by default
    #
    #         E_vlayer_ref = self.energy_v(v) + self.params['fields'][ind_y, categorical[ind_x, ind_y]]
    #
    #         v_output = self.compute_output_v(v)
    #
    #         fe = torch.zeros([categorical.shape[0], self.q], device=self.device)
    #         zero_state = torch.zeros(v.shape, device=self.device)
    #
    #         # ind_tks = []
    #         v_shuffle_output = []
    #         output_ref = []
    #         ind_ks = []
    #         shuffle_indx = []
    #         for iid, i in enumerate(self.hidden_convolution_keys):
    #             # Will fill this at correct indices to make corresponding layer
    #             zeros = torch.zeros(v_output[iid].shape, dtype=torch.double, device=self.device)
    #
    #             # Corrupted shuffle index
    #             shuffle_indx.append(torch.randint(v_output[iid].shape[2], v_output[iid].shape, dtype=torch.long, device=self.device))
    #             v_output_shuffled = v_output[iid].gather(2, shuffle_indx[iid])
    #
    #             # Random k for each huk
    #             ind_ks.append(torch.randint(v_output_shuffled.shape[2], v_output_shuffled.shape[:2], device=self.device).unsqueeze(2))
    #
    #             # huks with random huk values filled
    #             v_shuffle_output.append(zeros.scatter(2, ind_ks[iid], v_output_shuffled))
    #
    #             output_ref.append(v_output[iid] - v_shuffle_output[iid])
    #
    #         for c in range(self.q):
    #             E_vlayer = E_vlayer_ref - self.params['fields'][ind_y, c]
    #
    #             rc_state = zero_state.clone()
    #             rc_state[:, :, c] = 1
    #             rc_output = self.compute_output_v(rc_state)
    #             rc_shuffled = []
    #             output = []
    #             for iid, i in enumerate(self.hidden_convolution_keys):
    #                 # Re use stored shuffle index
    #                 rc_shuffled.append(rc_output[iid].gather(2, shuffle_indx[iid]))
    #                 # Extract only 1 huk value per huk
    #                 rc_out_single = zeros.scatter(2, ind_ks[iid], rc_shuffled[iid])
    #                 output.append(output_ref[iid] + rc_out_single)
    #
    #
    #             fe[:, c] += E_vlayer - self.logpartition_h(output)
    #
    #         fe_estimate = fe.gather(1, categorical[ind_x, ind_y].unsqueeze(1)).squeeze(1)
    #
    #         # To shuffle our visible states
    #         # ind_k = torch.randint(self.v_num, (self.v_num, ), dtype=torch.long, device=self.device)
    #
    #         # # Shuffle all h_uk
    #         # shuffle_indx = torch.randint(v_output[iid].shape[2], v_output[iid].shape, dtype=torch.long, device=self.device)
    #         # v_shuffle_output.append(v_output[iid].gather(2, shuffle_indx))
    #
    #         # First attempt, shuffling of visible nodes
    #         # v_shuffle = v.clone()[:, ind_k, :]
    #         # v_shuffle_output = self.compute_output_v(v_shuffle)
    #
    #         # for iid, i in enumerate(self.hidden_convolution_keys):
    #         #     output_ref.append(v_output[iid] - v_shuffle_output[iid])
    #         # output_ref = self.compute_output_v(v) - random_config_output
    #
    #         # # Original try
    #         # for c in range(self.q):
    #         #     random_config_state = zero_state.clone()
    #         #     random_config_state[:, :, c] = 1
    #         #     random_config_state_output = self.compute_output_v(random_config_state)
    #         #
    #         #     E_vlayer = E_vlayer_ref - self.params['fields'][ind_y, c]
    #         #
    #         #     output = []
    #         #     for iid, i in enumerate(self.hidden_convolution_keys):
    #         #         output.append(output_ref[iid] + random_config_state_output[iid])
    #         #
    #         #     fec = E_vlayer - self.logpartition_h(output)
    #         #     fe[:, c] += fec
    #         #
    #         # fe_estimate = fe.gather(1, categorical[ind_x, ind_y].unsqueeze(1)).squeeze(1)
    #
    #     return - fe_estimate - torch.logsumexp(- fe, 1)


    ## Used in our Loss Function
    def free_energy(self, v):
        return self.energy_v(v) - self.logpartition_h(self.compute_output_v(v))

    ## Not used but may be useful
    def free_energy_h(self, h):
        return self.energy_h(h) - self.logpartition_v(self.compute_output_h(h))

    ## Total Energy of a given visible and hidden configuration
    def energy(self, v, h, remove_init=False, hidden_sub_index=-1):
        return self.energy_v(v, remove_init=remove_init) + self.energy_h(h, sub_index=hidden_sub_index, remove_init=remove_init) - self.bidirectional_weight_term(v, h, hidden_sub_index=hidden_sub_index)

        ## Total Energy of a given visible and hidden configuration

    def energy_PT(self, v, h, remove_init=False):
        E = torch.zeros((self.N_PT, v.shape[1]), device=self.device)
        for i in range(self.N_PT):
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
            E[iid] = h_uk.mul(conv[iid]).sum(2).sum(1) # 10/k sum u sum k huk(Iuk)

        if E.shape[0] > 1:
            return E.sum(0)
        else:
            return E

    ############################################################# Individual Layer Functions
    ## Computes g(si) term of potential
    def energy_v(self, config, remove_init=False):
        # config is a one hot vector
        v = config.double()
        E = torch.zeros(config.shape[0], device=self.device)
        for i in range(self.q):
            if remove_init:
                E -= v[:, :, i].dot(self.params['fields'][:, i] - self.params['fields0'][:, i])
            else:
                E -= v[:, :, i].matmul(self.params['fields'][:, i])

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
                a_plus = self.params[f'{i}_gamma+'].sub(self.params[f'{i}_0gamma+']).unsqueeze(0).unsqueeze(2)
                a_minus = self.params[f'{i}_gamma-'].sub(self.params[f'{i}_0gamma-']).unsqueeze(0).unsqueeze(2)
                theta_plus = self.params[f'{i}_theta+'].sub(self.params[f'{i}_0theta+']).unsqueeze(0).unsqueeze(2)
                theta_minus = self.params[f'{i}_theta-'].sub(self.params[f'{i}_0theta-']).unsqueeze(0).unsqueeze(2)
            else:
                a_plus = self.params[f'{i}_gamma+'].unsqueeze(0).unsqueeze(2)
                a_minus = self.params[f'{i}_gamma-'].unsqueeze(0).unsqueeze(2)
                theta_plus = self.params[f'{i}_theta+'].unsqueeze(0).unsqueeze(2)
                theta_minus = self.params[f'{i}_theta-'].unsqueeze(0).unsqueeze(2)

            if sub_index != -1:
                con = config[iid][sub_index]
            else:
                con = config[iid]

            # Applies the dReLU activation function
            zero = torch.zeros_like(con, device=self.device)
            config_plus = torch.maximum(con, zero)
            config_minus = torch.maximum(-con, zero)

            E[iid] = ((config_plus.square() * a_plus) / 2 + (config_minus.square() * a_minus) / 2 + (config_plus * theta_plus) + (config_minus * theta_minus)).sum((2, 1))
            # E[iid] *= (10/self.convolution_topology[i]["convolution_dims"][2]) # Normalize across different convolution sizes
        if E.shape[0] > 1:
            E = E.sum(0)

        return E

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
    # N_PT and nchains arguments are used only for generating data currently
    def random_init_config_h(self, zeros=False, custom_size=False):
        config = []
        for iid, i in enumerate(self.hidden_convolution_keys):
            batch, h_num, convx_num, convy_num = self.convolution_topology[i]["convolution_dims"]

            if custom_size:
                size = (*custom_size, h_num, convx_num)
            else:
                size = (self.batch_size, h_num, convx_num)

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

    # def assign_h(self, h, index, assignment):
    #     for i in h:


    ## Marginal over hidden units
    def logpartition_h(self, inputs, beta=1):
        # Input is list of matrices I_uk
        marginal = torch.zeros((len(self.hidden_convolution_keys), inputs[0].shape[0]), device=self.device)
        for iid, i in enumerate(self.hidden_convolution_keys):
            if beta == 1:
                a_plus = (self.params[f'{i}_gamma+']).unsqueeze(0).unsqueeze(2)
                a_minus = (self.params[f'{i}_gamma-']).unsqueeze(0).unsqueeze(2)
                theta_plus = (self.params[f'{i}_theta+']).unsqueeze(0).unsqueeze(2)
                theta_minus = (self.params[f'{i}_theta-']).unsqueeze(0).unsqueeze(2)
            else:
                theta_plus = (beta * self.params[f'{i}_theta+'] + (1 - beta) * self.params[f'{i}_0theta+']).unsqueeze(0).unsqueeze(2)
                theta_minus = (beta * self.params[f'{i}_theta-'] + (1 - beta) * self.params[f'{i}_0theta-']).unsqueeze(0).unsqueeze(2)
                a_plus = (beta * self.params[f'{i}_gamma+'] + (1 - beta) * self.params[f'{i}_0gamma+']).unsqueeze(0).unsqueeze(2)
                a_minus = (beta * self.params[f'{i}_gamma-'] + (1 - beta) * self.params[f'{i}_0gamma-']).unsqueeze(0).unsqueeze(2)
            y = torch.logaddexp(self.log_erf_times_gauss((-inputs[iid] + theta_plus) / torch.sqrt(a_plus)) - 0.5 * torch.log(a_plus), self.log_erf_times_gauss((inputs[iid] + theta_minus) / torch.sqrt(a_minus)) - 0.5 * torch.log(a_minus)).sum(
                    1) + 0.5 * np.log(2 * np.pi) * inputs[iid].shape[1]
            marginal[iid] = y.sum(1)  # 10 added so hidden layer has stronger effect on free energy, also in energy_h
            # marginal[iid] /= self.convolution_topology[i]["convolution_dims"][2]
        return marginal.sum(0)

    ## Marginal over visible units
    def logpartition_v(self, inputs, beta=1):
        if beta == 1:
            return torch.logsumexp(self.params['fields'][None, :, :] + inputs, 2).sum(1)
        else:
            return torch.logsumexp((beta * self.params['fields'] + (1 - beta) * self.params['fields0'])[None, :] + beta * inputs, 2).sum(1)

    ## Compute Input for Hidden Layer from Visible Potts, Uses categorical not one hot vector
    def compute_output_v(self, X): # X is the one hot vector
        outputs = []
        for i in self.hidden_convolution_keys:
            convx = self.convolution_topology[i]["convolution_dims"][2]
            outputs.append(F.conv2d(X.unsqueeze(1).double(), self.params[f"{i}_W"], stride=self.convolution_topology[i]["stride"],
                                    padding=self.convolution_topology[i]["padding"],
                                    dilation=self.convolution_topology[i]["dilation"]).squeeze(3))
            # outputs[-1] /= convx
        return outputs

    ## Compute Input for Visible Layer from Hidden dReLU
    def compute_output_h(self, Y):  # from h_uk (B, hidden_num, convx_num)
        outputs = []
        nonzero_masks = []
        for iid, i in enumerate(self.hidden_convolution_keys):
            convx = self.convolution_topology[i]["convolution_dims"][2]
            outputs.append(F.conv_transpose2d(Y[iid].unsqueeze(3), self.params[f"{i}_W"],
                                              stride=self.convolution_topology[i]["stride"],
                                              padding=self.convolution_topology[i]["padding"],
                                              dilation=self.convolution_topology[i]["dilation"],
                                              output_padding=self.convolution_topology[i]["output_padding"]).squeeze(1))
            nonzero_masks.append((outputs[-1] != 0.).double())  # Used for calculating mean of outputs, don't want zeros to influence mean
            # outputs[-1] /= convx  # multiply by 10/k to normalize by convolution dimension
        if len(outputs) > 1:
            # Returns mean output from all hidden layers
            mean_denominator = torch.sum(torch.stack(nonzero_masks), 0)
            return torch.sum(torch.stack(outputs), 0) / mean_denominator
        else:
            return outputs[0]

    ## Gibbs Sampling of Potts Visbile Layer
    def sample_from_inputs_v(self, psi, beta=1):  # Psi ohe (Batch_size, v_num, q)   fields (self.v_num, self.q)
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

        return F.one_hot(high, self.q)

    ## Gibbs Sampling of dReLU hidden layer
    def sample_from_inputs_h(self, psi, nancheck=False, beta=1):  # psi is a list of hidden Iuks
        h_uks = []
        for iid, i in enumerate(self.hidden_convolution_keys):
            if beta == 1:
                a_plus = self.params[f'{i}_gamma+'].unsqueeze(0).unsqueeze(2)
                a_minus = self.params[f'{i}_gamma-'].unsqueeze(0).unsqueeze(2)
                theta_plus = self.params[f'{i}_theta+'].unsqueeze(0).unsqueeze(2)
                theta_minus = self.params[f'{i}_theta-'].unsqueeze(0).unsqueeze(2)
            else:
                theta_plus = (beta * self.params[f'{i}_theta+'] + (1 - beta) * self.params[f'{i}_0theta+']).unsqueeze(0).unsqueeze(2)
                theta_minus = (beta * self.params[f'{i}_theta-'] + (1 - beta) * self.params[f'{i}_0theta-']).unsqueeze(0).unsqueeze(2)
                a_plus = (beta * self.params[f'{i}_gamma+'] + (1 - beta) * self.params[f'{i}_0gamma+']).unsqueeze(0).unsqueeze(2)
                a_minus = (beta * self.params[f'{i}_gamma-'] + (1 - beta) * self.params[f'{i}_0gamma-']).unsqueeze(0).unsqueeze(2)
                psi[iid] *= beta

            if nancheck:
                nans = torch.isnan(psi[iid])
                if nans.max():
                    nan_unit = torch.nonzero(nans.max(0))[0]
                    print('NAN IN INPUT')
                    print('Hidden units', nan_unit)

            psi_plus = (-psi[iid]).add(theta_plus).div(torch.sqrt(a_plus))
            psi_minus = psi[iid].add(theta_minus).div(torch.sqrt(a_minus))

            etg_plus = self.erf_times_gauss(psi_plus)  # Z+ * sqrt(a+)
            etg_minus = self.erf_times_gauss(psi_minus)  # Z- * sqrt(a-)

            p_plus = 1 / (1 + (etg_minus / torch.sqrt(a_minus)) / (etg_plus / torch.sqrt(a_plus)))  # p+ 1 / (1 +( (Z-/sqrt(a-))/(Z+/sqrt(a+))))    =   (Z+/(Z++Z-)
            nans = torch.isnan(p_plus)

            if True in nans:
                p_plus[nans] = torch.tensor(1.) * (torch.abs(psi_plus[nans]) > torch.abs(psi_minus[nans]))
            p_minus = 1 - p_plus

            is_pos = torch.rand(psi[iid].shape, device=self.device) < p_plus

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
            v[i], htmp = self.markov_step(v[i], beta=beta)
            for hid in range(self.h_layer_num):
                h[hid][i] = htmp[hid]
            e[i] = self.energy(v[i], h, hidden_sub_index=i)

        if self.record_swaps:
            particle_id = torch.arange(self.N_PT).unsqueeze(1).expand(self.N_PT, v.shape[1])

        betadiff = self.betas[1:] - self.betas[:-1]
        for i in np.arange(self.count_swaps % 2, self.N_PT - 1, 2):
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
            v[i], htmp = self.markov_step(v[i], beta=beta)
            for kid, k in self.hidden_convolution_keys:
                h[kid][i] = htmp[kid]
            e[i] = self.energy(v[i], h, hidden_sub_index=i)
        return v, h, e

    def exchange_step_PT(self, v, h, e, compute_energy=True):
        if compute_energy:
            for i in torch.arange(self.N_PT):
                e[i] = self.energy(v[i], h, hidden_sub_index=i)

        if self.record_swaps:
            particle_id = torch.arange(self.N_PT).unsqueeze(1).expand(self.N_PT, v.shape[1])

        betadiff = self.betas[1:] - self.betas[:-1]
        for i in np.arange(self.count_swaps % 2, self.N_PT - 1, 2):
            proba = torch.exp(betadiff[i] * e[i+1] - e[i]).minimum(torch.ones_like(e[i]))
            swap = torch.rand(proba.shape[0], device=self.device) < proba
            if i > 0:
                v[i:i + 2, swap] = torch.flip(v[i-1: i+1], [0])[:, swap]
                for hid in range(self.h_layer_num):
                    h[hid][i:i + 2, swap] = torch.flip(h[hid][i - 1: i + 1], [0])[:, swap]
                e[i:i + 2, swap] = torch.flip(e[i-1: i+1], [0])[:, swap]
                if self.record_swaps:
                    particle_id[i:i + 2, swap] = torch.flip(particle_id[i-1: i+1], [0])[:, swap]
            else:
                v[i:i + 2, swap] = torch.flip(v[:i+1], [0])[:, swap]
                for hid in range(self.h_layer_num):
                    h[hid][i:i + 2, swap] = torch.flip(h[hid][:i + 1], [0])[:, swap]
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
            if self.worker_num == 0:
                threads = 1
            else:
                threads = self.worker_num
            seqs, seq_read_counts, all_chars, q_data = fasta_read(self.fasta_file, self.molecule, drop_duplicates=True, threads=threads)
        except IOError:
            print(f"Provided Fasta File '{self.fasta_file}' Not Found")
            print(f"Current Directory '{os.curdir}'")
            exit()

        if self.weights == "fasta":
            self.weights = np.asarray(seq_read_counts)

        if q_data != self.q:
            print(
                f"State Number mismatch! Expected q={self.q}, in dataset q={q_data}. All observed chars: {all_chars}")
            exit(-1)

        if self.weights is None:
            data = pd.DataFrame(data={'sequence': seqs})
        else:
            data = pd.DataFrame(data={'sequence': seqs, 'seq_count': self.weights})

        train, validate = train_test_split(data, test_size=0.2, random_state=self.seed)
        self.validation_data = validate
        self.training_data = train

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
    def train_dataloader(self, init_fields=True):
        # Get Correct Weights
        if "seq_count" in self.training_data.columns:
            training_weights = self.training_data["seq_count"].tolist()
        else:
            training_weights = None

        train_reader = RBMCaterogical(self.training_data, self.q, weights=training_weights, max_length=self.v_num, shuffle=False, base_to_id=self.molecule, device=self.device, one_hot=True)

        # initialize fields from data
        if init_fields:
            with torch.no_grad():
                initial_fields = train_reader.field_init()
                self.params['fields'] += initial_fields
                self.params['fields0'] += initial_fields

        return torch.utils.data.DataLoader(
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

        val_reader = RBMCaterogical(self.validation_data, self.q, weights=validation_weights, max_length=self.v_num, shuffle=False, base_to_id=self.molecule, device=self.device, one_hot=True)

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
        elif self.loss_type == "energy":
            if self.sample_type == "gibbs":
                return self.training_step_CD_energy(batch, batch_idx)
            elif self.sample_type == "pt":
                print("Energy Loss with Parallel Tempering is currently unsupported")
                exit(1)

    def validation_step(self, batch, batch_idx):
        # Needed for PsuedoLikelihood calculation

        seqs, V_pos, one_hot, seq_weights = batch

        # psuedolikelihood = (self.psuedolikelihood(one_hot) * seq_weights).sum() / seq_weights.sum()
        free_energy_avg = (self.free_energy(one_hot) * seq_weights).sum() / seq_weights.sum()

        batch_out = {
             # "val_psuedolikelihood": psuedolikelihood.detach()
             "val_free_energy": free_energy_avg.detach()
        }

        # self.log("ptl/val_psuedolikelihood", psuedolikelihood, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/val_free_energy", free_energy_avg, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return batch_out

    def validation_epoch_end(self, outputs):
        # avg_pl = torch.stack([x['val_psuedolikelihood'] for x in outputs]).mean()
        # self.logger.experiment.add_scalar("Validation Psuedolikelihood", avg_pl, self.current_epoch)
        avg_pl = torch.stack([x['val_free_energy'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Validation Free Energy", avg_pl, self.current_epoch)

    ## On Epoch End Collects Scalar Statistics and Distributions of Parameters for the Tensorboard Logger
    def training_epoch_end(self, outputs):
        # These are detached
        avg_loss = torch.stack([x['loss'].detach() for x in outputs]).mean()
        avg_dF = torch.stack([x["free_energy_diff"] for x in outputs]).mean()
        field_reg = torch.stack([x["field_reg"] for x in outputs]).mean()
        weight_reg = torch.stack([x["weight_reg"] for x in outputs]).mean()
        distance_reg = torch.stack([x["distance_reg"] for x in outputs]).mean()
        free_energy = torch.stack([x["train_free_energy"] for x in outputs]).mean()
        # psuedolikelihood = torch.stack([x['train_psuedolikelihood'] for x in outputs]).mean()

        self.logger.experiment.add_scalars("All Scalars", {"Loss": avg_loss,
                                                           "CD_Loss": avg_dF,
                                                           "Field Reg": field_reg,
                                                           "Weight Reg": weight_reg,
                                                           "Distance Reg": distance_reg,
                                                           # "Train_Psuedolikelihood": psuedolikelihood,
                                                           "Train Free Energy": free_energy,
                                                           }, self.current_epoch)

        for name, p in self.params.items():
            self.logger.experiment.add_histogram(name, p.detach(), self.current_epoch)

    ## Not yet rewritten for crbm
    def training_step_CD_energy(self, batch, batch_idx):
        seqs, V_pos, one_hot, seq_weights = batch
        weights = seq_weights.clone.detach()
        V_pos = V_pos.clone().detach()
        V_neg, h_neg, V_pos, h_pos = self(V_pos)

        energy_pos = (self.energy(V_pos, h_pos) * weights).sum() / weights.sum()# energy of training data
        energy_neg = (self.energy(V_neg, h_neg) * weights).sum() / weights.sum() # energy of gibbs sampled visible states

        cd_loss = energy_pos - energy_neg

        # psuedolikelihood = (self.psuedolikelihood(one_hot) * weights).sum() / weights.sum()

        reg1 = self.lf / 2 * self.params['fields'].square().sum((0, 1))
        tmp = torch.sum(torch.abs(self.W), (1, 2)).square()
        reg2 = self.l1_2 / (2 * self.q * self.v_num * self.h_num) * tmp.sum()

        loss = cd_loss + reg1 + reg2

        logs = {"loss": loss.detach(),
                # "train_psuedolikelihood": psuedolikelihood.detach(),
                "free_energy_diff": cd_loss.detach(),
                "field_reg": reg1.detach(),
                "weight_reg": reg2.detach()
                }

        # self.log("ptl/train_psuedolikelihood", psuedolikelihood, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return logs

    # Not yet rewritten for CRBM
    def training_step_PT_free_energy(self, batch, batch_idx):
        seqs, V_pos, one_hot, seq_weights = batch
        V_pos = V_pos.clone().detach()
        V_neg, h_neg, V_pos, h_pos = self.forward(V_pos)
        weights = seq_weights.clone().detach()

        # psuedo likelihood actually minimized, loss sits around 0 but does it's own thing
        F_v = (self.free_energy(V_pos) * weights).sum() / weights.sum()  # free energy of training data
        F_vp = (self.free_energy(V_neg) * weights).sum() / weights.sum()  # free energy of gibbs sampled visible states
        cd_loss = F_v - F_vp  # Should Give same gradient as Tubiana Implementation minus the batch norm on the hidden unit act

        psuedolikelihood = (self.psuedolikelihood(one_hot).clone().detach() * weights).sum() / weights.sum()

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
        seqs, V_pos, one_hot, seq_weights = batch
        weights = seq_weights.clone()
        V_neg_oh, h_neg, V_pos_oh, h_pos = self(one_hot)

        # psuedo likelihood actually minimized, loss sits around 0 but does it's own thing
        F_v = (self.free_energy(V_pos_oh) * weights).sum() / weights.sum()  # free energy of training data
        F_vp = (self.free_energy(V_neg_oh) * weights).sum() / weights.sum()  # free energy of gibbs sampled visible states
        cd_loss = F_v - F_vp  # Should Give same gradient as Tubiana Implementation minus the batch norm on the hidden unit activations

        free_energy = F_v.detach()

        # psuedolikelihood = (self.psuedolikelihood(V_pos_oh) * weights).sum() / weights.sum()

        # Regularization Terms
        reg1 = self.lf/2 * self.params['fields'].square().sum((0, 1))

        reg2 = torch.zeros((1,), device=self.device)
        reg3 = torch.zeros((1,), device=self.device)
        for iid, i in enumerate(self.hidden_convolution_keys):
            # conv_shape = self.convolution_topology[i]["convolution_dims"]  # (batch, hidden u, hidden k, convy_num=1)
            # x = torch.sum(torch.abs(self.params[f"{i}_W"]), (3, 2)).square() / (conv_shape[2])
            W_shape = self.convolution_topology[i]["weight_dims"]  # (h_num,  input_channels, kernel0, kernel1)
            x = torch.sum(torch.abs(self.params[f"{i}_W"]), (3, 2, 1)).square()
            reg2 += x.sum(0) * self.l1_2 / (2*W_shape[1]*W_shape[2]*W_shape[3])

            # Size of Convolution Filters
            # weight_size = (h_num, input_channels, kernel[0], kernel[1])
            reg3 += self.ld / ((self.params[f"{i}_W"].abs() - self.params[f"{i}_W"].squeeze(1).abs()).abs().sum((1, 2, 3)).mean() + 1)

        # tmp = torch.sum(torch.abs(self.W), (1, 2)).square()

        loss = cd_loss + reg1 + reg2 + reg3

        logs = {"loss": loss,
                # "train_psuedolikelihood": psuedolikelihood.detach(),
                "free_energy_diff": cd_loss.detach(),
                "train_free_energy": free_energy,
                "field_reg": reg1.detach(),
                "weight_reg": reg2.detach(),
                "distance_reg": reg3.detach()
                }

        # self.log("ptl/train_psuedolikelihood", psuedolikelihood, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_free_energy", free_energy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return logs

    ## Gradient Clipping for poor behavior, have no need for it yet
    # def on_after_backward(self):
    #     self.grad_norm_clip_value = 10
    #     torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm_clip_value)

    def forward(self, V_pos_ohe):
        # Enforces Zero Sum Gauge on Weights
        # self.W = self.params['W_raw'] - self.params['W_raw'].sum(-1).unsqueeze(2) / self.q
        if self.sample_type == "gibbs":
            # Gibbs sampling
            # pytorch lightning handles the device
            fantasy_v = V_pos_ohe.clone()
            h_pos = self.sample_from_inputs_h(self.compute_output_v(fantasy_v))
            # with torch.no_grad() # only use last sample for gradient calculation, may be helpful but honestly not the slowest thing rn
            for _ in range(self.mc_moves-1):
                fantasy_v, fantasy_h = self.markov_step(fantasy_v)

            V_neg, fantasy_h = self.markov_step(fantasy_v)
            h_neg = self.sample_from_inputs_h(self.compute_output_v(V_neg))

            return V_neg, h_neg, V_pos_ohe, h_pos

        elif self.sample_type == "pt":
            # Parallel Tempering
            h_pos = self.sample_from_inputs_h(self.compute_output_v(V_pos_ohe))

            n_chains = V_pos_ohe.shape[0]
            fantasy_v = self.random_init_config_v(custom_size=(self.N_PT, n_chains))
            fantasy_h = self.random_init_config_h(custom_size=(self.N_PT, n_chains))
            fantasy_E = self.energy_PT(fantasy_v, fantasy_h)

            for _ in range(self.mc_moves):
                fantasy_v, fantasy_h, fantasy_E = self.markov_PT_and_exchange(fantasy_v, fantasy_h, fantasy_E)
                self.update_betas()

            V_neg = fantasy_v[0]
            h_neg = fantasy_h[0]

            return V_neg, h_neg, V_pos_ohe, h_pos

    # X must be a pandas dataframe with the sequences in string format under the column 'sequence'
    # Returns the likelihood for each sequence in an array
    def predict(self, X):
        # Read in data
        reader = RBMCaterogical(X, self.q, weights=None, max_length=self.v_num, shuffle=False, base_to_id=self.molecule, device=self.device, one_hot=True)
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
                seqs, V_pos, one_hot, seq_weights = batch
                likelihood += self.likelihood(one_hot).detach().tolist()

        return X.sequence.tolist(), likelihood

    # disabled until I figure out pseudo likelihood function
    # def predict_psuedo(self, X):
    #     reader = RBMCaterogical(X, weights=None, max_length=self.v_num, shuffle=False, base_to_id=self.molecule, device=self.device)
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
    #             seqs, V_pos, one_hot, seq_weights = batch
    #             likelihood += self.psuedolikelihood(one_hot).detach().tolist()
    #
    #     return X.sequence.tolist(), likelihood

    # Return param as a numpy array
    def get_param(self, param_name):
        try:
            tensor = self.params[param_name].clone()
            return tensor.detach().numpy()
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

    # not yet functional, needs debugging
    def cgf_from_inputs_h(self, I):
        B = I.shape[0]
        out = torch.zeros((self.h_layer_num, I.shape), device=self.device)
        for iid, i in enumerate(self.hidden_convolution_keys):
            sqrt_gamma_plus = torch.sqrt(self.params[f"{i}_gamma+"]).expand(B, -1)
            sqrt_gamma_minus = torch.sqrt(self.params[f"{i}_gamma-"]).expand(B, -1)
            log_gamma_plus = torch.log(self.params[f"{i}_gamma+"]).expand(B, -1)
            log_gamma_minus = torch.log(self.params[f"{i}_gamma-"]).expand(B, -1)

            Z_plus = -self.log_erf_times_gauss((-I[iid] + self.params['theta+'].expand(B, -1)) / sqrt_gamma_plus) - 0.5 * log_gamma_plus
            Z_minus = self.log_erf_times_gauss((I[iid] + self.params['theta-'].expand(B, -1)) / sqrt_gamma_minus) - 0.5 * log_gamma_minus
            map = Z_plus > Z_minus
            out[iid][map] = Z_plus[map] + torch.log(1 + torch.exp(Z_minus[map] - Z_plus[map]))
            out[iid][~map] = Z_minus[map] + torch.log(1 + torch.exp(Z_plus[map] - Z_minus[map]))
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
                    config_init = [config_init[0][batches * i:batches * (i + 1)],  config_init[1][batches * i:batches * (i + 1)]]
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
                if config_init != []:
                    config = config_init
                else:
                    config = [self.random_init_config_v(custom_size=(N_PT, batches)), self.random_init_config_h(custom_size=(N_PT, batches))]

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
                if record_replica:
                    data = [config[0].clone().unsqueeze(0), self.clone_h(config[1], expand_dims=[0])]
                else:
                    data = [config[0][0].clone().unsqueeze(0), self.clone_h(config[1], expand_dims=[0], sub_index=0)]
            else:
                data = [config[0].clone().unsqueeze(0), self.clone_h(config[1], expand_dims=[0])]

            if N_PT > 1:
                if Ndata > 1:
                    if record_replica:
                        data_gen_size = (Ndata-1, N_PT, batches)
                    else:
                        data_gen_size = (Ndata - 1, N_PT)
            else:
                data_gen_size = (Ndata - 1)

            if Ndata > 1 or N_PT == 1:
                data_gen_v = self.random_init_config_v(custom_size=data_gen_size, zeros=True)
                data_gen_h = self.random_init_config_h(custom_size=data_gen_size, zeros=True)

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
                        data_gen_v[n] = config[0].clone()

                        clone = self.clone_h(config[1])
                        for hid in range(self.h_layer_num):
                            data_gen_h[n][hid] = clone[hid]

                    else:
                        data_gen_v[n] = config[0][0].clone()

                        clone = self.clone_h(config[1], sub_index=0)
                        for hid in range(self.h_layer_num):
                            data_gen_h[n][hid] = clone[hid]

                else:
                    data_gen_v[n] = config[0].clone()

                    clone = self.clone_h(config[1])
                    for hid in range(self.h_layer_num):
                        data_gen_h[n][hid] = clone[hid]

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

# returns list of strings containing sequences
# optionally returns the affinities


# Get Model from checkpoint File with specified version and directory
def get_checkpoint(version, dir=""):
    checkpoint_dir = dir + "/version_" + str(version) + "/checkpoints/"

    for file in os.listdir(checkpoint_dir):
        if file.endswith(".ckpt"):
            checkpoint_file = os.path.join(checkpoint_dir, file)
    return checkpoint_file

def get_beta_and_W(crbm, Wname):
    W = crbm.get_param(Wname).squeeze(1)
    return np.sqrt((W ** 2).sum(-1).sum(-1)), W

def conv_weights(rbm, Wname, name, rows, columns, h, w, molecule='rna'):
    beta, W = get_beta_and_W(rbm, Wname)
    order = np.argsort(beta)[::-1]
    fig = Sequence_logo_all(W[order], name=name + '.pdf', nrows=rows, ncols=columns, figsize=(h,w) ,ticks_every=10,ticks_labels_size=10,title_size=12, dpi=400, molecule=molecule)


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
              "batch_size": 6000,
              "mc_moves": 4,
              "seed": 38,
              "lr": 0.0065,
              "lr_final": None,
              "decay_after": 0.75,
              "loss_type": "free_energy",
              "sample_type": "gibbs",
              "sequence_weights": None,
              "optimizer": "AdamW",
              "epochs": 150,
              "weight_decay": 0.001,  # l2 norm on all parameters
              "l1_2": 100.0,
              "lf": 0.25,
              # "data_worker_num": 10  # Optionally Set these
              }

    config = configs.lattice_default_config
    config["l1_2"] = 0.8
    config["ld"] = 20.0
    # Edit config for dataset specific hyperparameters
    config["fasta_file"] = lattice_data
    config["sequence_weights"] = None
    config["epochs"] = 100
    config["convolution_topology"] = {
        "hidden1": {"number": 5, "kernel": (9, config["q"]), "stride": (3, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
        "hidden2": {"number": 5, "kernel": (7, config["q"]), "stride": (5, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
        "hidden3": {"number": 5, "kernel": (3, config["q"]), "stride": (2, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
        "hidden4": {"number": 5, "kernel": (config["v_num"], config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
    }

    # TODO List
    # 1: Add support for non perfect convolutions (involving mismatch of input, stride, and kernel)
    #       X Non Perfect Convolutions are now supported
    #       1.1 Add function that suggest kernel/stride sizes based off input size
    # 2: Add support for sampling of conv_rbm. Be able to choose hidden units you sample from
    #       2.1 Get Likelihood/AIS/PT/Gen_Data working for CRBM
    #   X rewrote AIS/PT/Gen_data methods to work with crbm
    #   X Choosing which to sample from will require making a new crbm out of only the selected hidden layers. From there sampling should be easy and supported
    # 3: Test this out on multiple datasets
    # 4: Extend Analysis Methods for crbm
    #       4.1 Finish cgf viewing in analysis_methods
    # 5: Other stuff I'm sure

    # (kernel[0] - 1) - 1) / stride[0]
    # shapes1 = conv2d_dim((100, 1, 27, 20), config["convolution_topology"]["hidden1"])
    # shapes2 = conv2d_dim((100, 1, 27, 20), config["convolution_topology"]["hidden2"])
    # shapes3 = conv2d_dim((100, 1, 27, 20), config["convolution_topology"]["hidden3"])


    # This works program this in!!!!
    #   (27   -  1 * (kernel[0] -1 ) - 1) / stride[0]
    # h1(26   - (kernel-1) )/stride       26-6 -> 18
    # (input_sizex - dilation[0] * (kernel[0] - 1) - 1)
    # def calc_output_transpose(convolution_definition, input_size):
    #     Hout = (input_size[2] - 1) * convolution_definition["stride"][0] - 2* convolution_definition["padding"][0] + convolution_definition['dilation'][0] * (convolution_definition["kernel"][0] - 1) + convolution_definition["output_padding"][0] + 1
    #     Wout = (input_size[3] - 1) * convolution_definition["stride"][1] - 2* convolution_definition["padding"][1] + convolution_definition['dilation'][1] * (convolution_definition["kernel"][1] - 1) + convolution_definition["output_padding"][1] + 1
    #     print("size", Hout, Wout)

    # input_size = (500, 5, convx_num)
    # calc_output_transpose(config["convolution_topology"]["hidden1"], )
    # calc_output_transpose(config["convolution_topology"]["hidden2"])
    # calc_output_transpose(config["convolution_topology"]["hidden3"])



    # Training Code
    crbm = CRBM(config, debug=False)
    logger = TensorBoardLogger('tb_logs', name='conv_lattice_trial')
    plt = Trainer(max_epochs=config['epochs'], logger=logger, gpus=1)  # gpus=1,
    plt.fit(crbm)


    # Debugging Code1
    # checkpoint = "./tb_logs/conv_lattice_trial/version_82/checkpoints/epoch=83-step=167.ckpt"
    # crbm_lat = CRBM.load_from_checkpoint(checkpoint)
    #
    # crbm_lat.prepare_data()
    # crbm_lat.AIS()
    # seqs, likelis = crbm_lat.predict(crbm_lat.validation_data)
    # print("hi")
    # h1_W = crbm_lat.get_param("hidden1_W")

    # conv_weights(crbm_lat, "hidden1_W", "crbm_lattice_h1_weights_new", 4, 2, 11, 8, molecule="protein")
    # conv_weights(crbm_lat, "hidden2_W", "crbm_lattice_h2_weights_new", 4, 2, 11, 8, molecule="protein")
    # conv_weights(crbm_lat, "hidden3_W", "crbm_lattice_h3_weights_new", 4, 2, 11, 8, molecule="protein")
    # conv_weights(crbm_lat, "hidden4_W", "crbm_lattice_h4_weights_new", 4, 2, 11, 8, molecule="protein")
    # rbm_lat = CRBM(config, debug=True)
    # rbm_lat.prepare_data()
    # td = rbm_lat.train_dataloader()
    # for iid, i in enumerate(td):
    #     if iid > 0:
    #         break
    #     seq, cat, ohe, seq_weights = i
    #     rbm_lat(ohe)
    # #     v_out = rbm_lat.compute_output_v(ohe)
    # #     # h = rbm_lat.sample_from_inputs_h(v_out)
    # #     # h_out = rbm_lat.compute_output_h(h)
    # #     # nv = rbm_lat.sample_from_inputs_v(h_out)
    # #     # fe = rbm_lat.free_energy(nv)
    #     pl = rbm_lat.psuedolikelihood(ohe)
    #     print(pl.mean())




    # # Training Code
    # rbm_lat = RBM(config, debug=False)
    # logger = TensorBoardLogger('tb_logs', name='lattice_trial')
    # plt = Trainer(max_epochs=config['epochs'], logger=logger, gpus=1)  # gpus=1,
    # plt.fit(rbm_lat)


    # check weights
    # version = 0
    # # checkpoint = get_checkpoint(version, dir="./tb_logs/lattice_trial/")
    # checkpoint = "/mnt/D1/globus/rbm_hyperparam_results/train_rbm_ad5d7_00005_5_l1_2=0.3832,lf=0.00011058,lr=0.065775,weight_decay=0.086939_2022-01-18_11-02-53/checkpoints/epoch=99-step=499.ckpt"
    # rbm = RBM.load_from_checkpoint(checkpoint)
    # rbm.energy_per_state()
    # conv_weights(rbm, "./lattice_proteins_verification" + "/allweights", 5, 1, 10, 2, molecule="protein")

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
    # # plt = Trainer(max_epochs=epochs, logger=logger, gpus=0, profiler=profiler)  # gpus=1,
    # plt = Trainer(max_epochs=epochs, logger=logger, gpus=0, profiler="advanced")  # gpus=1,
    # # tic = time.perf_counter()
    # plt.fit(rbm)
    # toc = time.perf_counter()
    # tim = toc-tic
    #
    # print("Trial took", tim, "seconds")


    # plt = Trainer(gpus=1, max_epochs=10)
    # plt = Trainer(gpus=1, profiler='advanced', max_epochs=10)
    # plt = Trainer(profiler='advanced', max_epochs=10)
    # plt = Trainer(max_epochs=1)
    # plt.fit(rbm)
