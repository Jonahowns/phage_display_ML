import time
import pandas as pd
import math
import json
import numpy as np
import sys
from pytorch_lightning import LightningModule, Trainer
# from pytorch_lightning.profiler import SimpleProfiler, PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW, Adagrad, Adadelta  # Supported Optimizers
from multiprocessing import cpu_count # Just to set the worker number
from torch.autograd import Variable

from rbm_torch.utils.utils import Categorical, conv2d_dim
from torch.utils.data import WeightedRandomSampler
from rbm_torch.models.base import Base_drelu

class pool_CRBM(Base_drelu):
    def __init__(self, config, debug=False, precision="double", meminfo=False):
        super().__init__(config, debug=debug, precision=precision)

        self.mc_moves = config['mc_moves']  # Number of MC samples to take to update hidden and visible configurations

        # sample types control whether gibbs sampling, pcd, from the data or parallel tempering from random configs are used
        # Switches How the training of the RBM is performed
        self.sample_type = config['sample_type']

        assert self.sample_type in ['gibbs', 'pt', 'pcd']

        # Regularization Options #
        ###########################################
        self.l1_2 = config['l1_2']  # regularization on weights, ex. 0.25
        self.lf = config['lf']  # regularization on fields, ex. 0.001
        self.ld = config['ld']  # regularization on distance b/t aligned weights
        self.lgap = config['lgap'] # regularization on gaps
        self.lbs = config['lbs']  # regularization to promote using both sides of the weights
        self.lcorr = config['lcorr']  # regularization on correlation of weights
        ###########################################
        self.lkd = config['lkd']
        self.kl_div = torch.nn.KLDivLoss(log_target=True, reduction='none')

        self.convolution_topology = config["convolution_topology"]

        if type(self.v_num) is int:
            # Normal dist. times this value sets initial weight values
            self.weight_initial_amplitude = np.sqrt(0.001 / self.v_num)
            self.register_parameter("fields", nn.Parameter(torch.zeros((self.v_num, self.q), device=self.device)))
            self.register_parameter("fields0", nn.Parameter(torch.zeros((self.v_num, self.q), device=self.device)))
        elif type(self.v_num) is tuple:  # Normal dist. times this value sets initial weight values
            self.weight_initial_amplitude = np.sqrt(0.01 / math.prod(list(self.v_num)))
            self.register_parameter("fields", nn.Parameter(torch.zeros((*self.v_num, self.q), device=self.device)))
            self.register_parameter("fields0", nn.Parameter(torch.zeros((*self.v_num, self.q), device=self.device)))

        self.hidden_convolution_keys = list(self.convolution_topology.keys())

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

            self.pools.append(nn.MaxPool1d(pool_kernel, stride=1, return_indices=True, padding=0))
            self.unpools.append(nn.MaxUnpool1d(pool_kernel, stride=1, padding=0))

            # Convolution Weights
            self.register_parameter(f"{key}_W", nn.Parameter(self.weight_initial_amplitude * torch.randn(self.convolution_topology[key]["weight_dims"], device=self.device)))
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

        self.ind_temp_schedule = self.init_temp_schedule(config["ind_temp"])
        self.seq_temp_schedule = self.init_temp_schedule(config["seq_temp"])


        # Saves Our hyperparameter options into the checkpoint file generated for Each Run of the Model
        # i.e. Simplifies loading a model that has already been run
        self.save_hyperparameters()

        # Initialize PT members
        if self.sample_type == "pt":
            try:
                self.N_PT = config["N_PT"]
            except KeyError:
                print("No member N_PT found in provided config.")
                sys.exit(1)
            self.initialize_PT(self.N_PT, n_chains=None, record_acceptance=True, record_swaps=True)

    @property
    def h_layer_num(self):
        return len(self.hidden_convolution_keys)

    ## Used in our Loss Function
    def free_energy(self, v):
        return self.energy_v(v) - self.logpartition_h(self.compute_output_v(v))

    def free_energy_ind(self, v):
        return self.energy_v(v).unsqueeze(1) - self.logpartition_h_ind(self.compute_output_v(v))

    ## Not used but may be useful
    def free_energy_h(self, h):
        return self.energy_h(h) - self.logpartition_v(self.compute_output_h(h))

    ## Total Energy of a given visible and hidden configuration
    def energy(self, v, h, remove_init=False, hidden_sub_index=-1):
        return self.energy_v(v, remove_init=remove_init) + self.energy_h(h, sub_index=hidden_sub_index, remove_init=remove_init) - self.bidirectional_weight_term(v, h, hidden_sub_index=hidden_sub_index)

    def energy_PT(self, v, h, N_PT, remove_init=False):
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

    ## Computes -g(si) term of potential
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
            # zero = torch.zeros_like(con, device=self.device)
            config_plus = torch.clamp(con, min=0.)
            config_minus = -1 * torch.clamp(-con, min=0.)
            # config_plus = torch.maximum(con, zero)
            # config_minus = -1*torch.maximum(-con, zero)

            E[iid] = ((config_plus.square() * a_plus) / 2 + (config_minus.square() * a_minus) / 2 + (config_plus * theta_plus) + (config_minus * theta_minus)).sum(1)

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

            y = torch.logaddexp(self.log_erf_times_gauss((-inputs[iid] + theta_plus) / torch.sqrt(a_plus)) -
                                0.5 * torch.log(a_plus), self.log_erf_times_gauss((inputs[iid] + theta_minus) / torch.sqrt(a_minus)) - 0.5 * torch.log(a_minus)).sum(
                1) + 0.5 * np.log(2 * np.pi) * inputs[iid].shape[1]
            marginal[iid] = y
        return marginal.sum(0)

    def logpartition_h_ind(self, inputs, beta=1):
        # Input is list of matrices I_uk
        ys = []
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

            y = torch.logaddexp(self.log_erf_times_gauss((-inputs[iid] + theta_plus) / torch.sqrt(a_plus)) -
                                0.5 * torch.log(a_plus), self.log_erf_times_gauss(
                (inputs[iid] + theta_minus) / torch.sqrt(a_minus)) - 0.5 * torch.log(a_minus)) + 0.5 * np.log(2 * np.pi)
            ys.append(y)
        return torch.cat(ys, dim=1)

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
        self.max_inds = []
        self.min_inds = []
        for iid, i in enumerate(self.hidden_convolution_keys):
            weights = getattr(self, f"{i}_W")
            conv = F.conv2d(X.unsqueeze(1).type(torch.get_default_dtype()), weights, stride=self.convolution_topology[i]["stride"],
                            padding=self.convolution_topology[i]["padding"],
                            dilation=self.convolution_topology[i]["dilation"]).squeeze(3)

            max_pool, max_inds = self.pools[iid](conv.abs())

            flat_conv = conv.flatten(start_dim=2)
            max_conv_values = flat_conv.gather(2, index=max_inds.flatten(start_dim=2)).view_as(max_inds)
            max_pool *= max_conv_values/max_conv_values.abs()

            self.max_inds.append(max_inds)

            out = max_pool.flatten(start_dim=2)

            out.squeeze_(2)

            # out = self.normalize_inputs(out, key=i)

            if self.dr > 0.:
                out = F.dropout(out, p=self.dr, training=self.training)

            outputs.append(out)
            if True in torch.isnan(out):
                print("hi")

        return outputs

    def exp_activation(self, x, y=3):
        # assumes input is between -1 and 1
        act = (y*(-1+x.abs())).abs()
        # act = x.abs()

        # batch_normed = (act - (act.mean(0)).clamp(min=0)).clamp(min=0)

        # batch_normed = (act - y).clamp(min=0)
        # batch_normed = (batch_normed + 1e-8)/(batch_normed.sum(0)[None, :] + 1e-8)

        # h_normed = (batch_normed + 1e-8)/(batch_normed.sum(1)[:, None] + 1e-8)

        # ind_temp = self.ind_temp_schedule[self.current_epoch]
        # batch_normed = torch.softmax(act / ind_temp, -1)

        # z score
        # z_act = (act-act.mean(0)).div(act.std(0))
        # norm to 0 and 1
        # z_act = z_act + z_act.min().abs()
        # z_act = z_act.div(z_act.max())
        # return z_act

        return act

        # return (y*(-1+x.abs())).abs()


    ## Compute Input for Visible Layer from Hidden dReLU
    def compute_output_h(self, h, h_weights=None):  # from h_uk (B, hidden_num)
        outputs = []
        for iid, i in enumerate(self.hidden_convolution_keys):
            reconst = self.unpools[iid](h[iid].view_as(self.max_inds[iid]), self.max_inds[iid])

            # reconst *= h_weights[iid][:, :, None].abs()

            # reconst_total = reconst.sum(2)
            # reconst *= self.exp_activation(h_weights[iid][:, :, None], y=15)
            # reconst = reconst / reconst.sum(2).abs()[:, :, None] * reconst_total

            if reconst.ndim == 3:
                reconst.unsqueeze_(3)

            outputs.append(F.conv_transpose2d(reconst, getattr(self, f"{i}_W"),
                                              stride=self.convolution_topology[i]["stride"],
                                              padding=self.convolution_topology[i]["padding"],
                                              dilation=self.convolution_topology[i]["dilation"],
                                              output_padding=self.convolution_topology[i]["output_padding"]).squeeze(1))

        if len(outputs) > 1:
            out = torch.sum(torch.stack(outputs), 0)
        else:
            out = outputs[0]

        return out

    ## Gibbs Sampling of Potts Visbile Layer
    def sample_from_inputs_v(self, psi, beta=1):  # Psi ohe (Batch_size, v_num, q)   fields (self.v_num, self.q)
        if beta == 1:
            cum_probas = psi + getattr(self, "fields").unsqueeze(0)
        else:
            cum_probas = beta * psi + beta * getattr(self, "fields").unsqueeze(0) + (1 - beta) * getattr(self, "fields0").unsqueeze(0)

        maxi, max_indices = cum_probas.max(-1)
        maxi.unsqueeze_(2)
        cum_probas -= maxi
        cum_probas.exp_()
        cum_probas[cum_probas > 1e9] = 1e9  # For numerical stability.

        dist = torch.distributions.categorical.Categorical(probs=cum_probas)
        return F.one_hot(dist.sample(), self.q)

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

            etg_plus = self.erf_times_gauss(psi_plus)  # Z+ * sqrt(a+)
            etg_minus = self.erf_times_gauss(psi_minus)  # Z- * sqrt(a-)

            p_plus = 1 / (1 + (etg_minus / torch.sqrt(a_minus)) / (etg_plus / torch.sqrt(a_plus)))  # p+ 1 / (1 +( (Z-/sqrt(a-))/(Z+/sqrt(a+))))    =   (Z+/(Z++Z-)
            nans = torch.isnan(p_plus)

            if nans.any().item():
                p_plus[nans] = torch.tensor(1.) * (torch.abs(psi_plus[nans]) > torch.abs(psi_minus[nans]))
            # p_minus = 1 - p_plus

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

    ###################################################### Sampling Functions
    ## Samples hidden from visible and vice versa, returns newly sampled hidden and visible
    def markov_step(self, v, beta=1):
        # Gibbs Sampler
        h = self.sample_from_inputs_h(self.compute_output_v(v), beta=beta)
        return self.sample_from_inputs_v(self.compute_output_h(h), beta=beta), h

    def markov_step_with_hidden_input(self, v, beta=1):
        # Gibbs Sampler
        iv = self.compute_output_v(v)
        h = self.sample_from_inputs_h(iv, beta=beta)
        h_w = self.normalize_inputs(iv, key='all')
        h_w = [h.abs() for h in h_w]
        vn = self.sample_from_inputs_v(self.compute_output_h(h, h_weights=h_w), beta=beta)
        Ih = self.compute_output_v(vn)
        return vn, h, Ih, h_w

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

    ######################################################### Pytorch Lightning Functions
    # def on_after_backward(self):
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 10000, norm_type=2.0, error_if_nonfinite=True)


    # Clamps hidden potential values to acceptable range
    def on_before_zero_grad(self, optimizer):
        with torch.no_grad():
            for key in self.hidden_convolution_keys:
                for param in ["gamma+", "gamma-"]:
                    getattr(self, f"{key}_{param}").data.clamp_(min=0.05)
                for param in ["theta+", "theta-"]:
                    getattr(self, f"{key}_{param}").data.clamp_(min=0.0)
                getattr(self, f"{key}_W").data.clamp_(-1.0, 1.0)

    ## Calls Corresponding Training Function
    def training_step(self, batch, batch_idx):
        # All other functions use self.W for the weights
        if self.sample_type == "gibbs":
            return self.training_step_CD_free_energy(batch, batch_idx)
        elif self.sample_type == "pt":
            return self.training_step_PT_free_energy(batch, batch_idx)
        elif self.sample_type == "pcd":
            return self.training_step_PCD_free_energy(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        inds, seqs, one_hot, seq_weights = batch

        free_energy = self.free_energy(one_hot)
        # free_energy_avg = (free_energy * seq_weights).sum() / seq_weights.sum()

        batch_out = {
            "val_free_energy": free_energy.mean().detach()
        }

        # logging on step, for whatever reason allocates 512 bytes on gpu after every epoch.
        self.log("ptl/val_free_energy", batch_out["val_free_energy"], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=one_hot.shape[0])
        self.val_data_logs.append(batch_out)
        return

    def regularization_terms(self, distance_threshold=0.25):
        freg = self.lf / (2 * self.v_num * self.q) * getattr(self, "fields").square().sum((0, 1))
        wreg = torch.zeros((1,), device=self.device)
        dreg = torch.zeros((1,), device=self.device)  # discourages weights that are alike

        bs_loss = torch.zeros((1,), device=self.device)  # encourages weights to use both positive and negative contributions
        gap_loss = torch.zeros((1,), device=self.device)  # discourages high values for gaps

        for iid, i in enumerate(self.hidden_convolution_keys):
            W_shape = self.convolution_topology[i]["weight_dims"]  # (h_num,  input_channels, kernel0, kernel1)
            W = getattr(self, f"{i}_W")

            x = torch.sum(W.abs(), (3, 2, 1)).square()
            wreg += x.sum() * self.l1_2 / (2 * W_shape[1] * W_shape[2] * W_shape[3])
            # dreg += self.ld / ((getattr(self, f"{i}_W").abs() - getattr(self, f"{i}_W").squeeze(1).abs()).abs().sum((1, 2, 3)).mean() + 1)
            gap_loss += self.lgap * W[:, :, :, -1].abs().sum()

            denom = torch.sum(torch.abs(W), (3, 2, 1))
            Wpos = torch.clamp(W, min=0.)
            Wneg = torch.clamp(W, max=0.)
            bs_loss += self.lbs * torch.abs(Wpos.sum((1, 2, 3)) / denom - torch.abs(Wneg.sum((1, 2, 3))) / denom).sum()

            ws = torch.concat([Wpos, Wneg.abs()], dim=0).squeeze(1)

            with torch.no_grad():

                # compute positional differences for all pairs of weights
                pdiff = (ws.unsqueeze(0).unsqueeze(2) - ws.unsqueeze(1).unsqueeze(3)).sum(4)

                # concatenate it to itself to make full diagonals
                wdm = torch.concat([pdiff, pdiff.clone()], dim=2)

                # get stride to get matrix of all diagonals on separate row
                si, sj, v2, v = wdm.size()
                i_s, j_s, v2_s, v_s = wdm.stride()
                wdm_s = torch.as_strided(wdm, (si, sj, v, v), (i_s, j_s, v2_s, v2_s + 1))

                # get the best alignment position
                best_align = W_shape[2] - torch.argmin(wdm_s.abs().sum(3), -1)

                # get indices for all pairs of i <= j
                bat_x, bat_y = torch.triu_indices(si, sj, 1)

                # get their alignments
                bas = best_align[bat_x, bat_y]

                # create shifted weights
                vt_ind = torch.arange(len(bat_x), device=self.device)[:, None].expand(-1, v)
                v_ind = torch.arange(v, device=self.device)[None, :].expand(len(bat_y), -1)
                rolled_j = ws[bat_y][vt_ind, (v_ind + bas[:, None]) % v]

            # norms of all weights
            w_norms = torch.linalg.norm(ws, dim=(2, 1))

            # inner prod of weights i and shifted weights j
            inner_prod = torch.tensordot(ws[bat_x], rolled_j, dims=([2, 1], [2, 1]))[0]

            # angles between aligned weights
            angles = inner_prod/(w_norms[bat_x] * w_norms[bat_y] + 1e-6)

            # threshold
            angles = angles[angles > distance_threshold]

            dreg += angles.sum()

        dreg *= self.ld

        # Passed to training logger
        reg_dict = {
            "field_reg": freg.detach(),
            "weight_reg": wreg.detach(),
            "distance_reg": dreg.detach(),
            "gap_reg": gap_loss.detach(),
            "both_side_reg": bs_loss.detach()
        }

        return freg, wreg, dreg, bs_loss, gap_loss, reg_dict

    def training_step_PT_free_energy(self, batch, batch_idx):
        inds, seqs, one_hot, seq_weights = batch

        V_neg_oh, h_neg, V_pos_oh, h_pos = self(one_hot)

        # Calculate CD loss
        F_v = (self.free_energy(V_pos_oh) * seq_weights).sum() / seq_weights.sum()  # free energy of training data
        F_vp = (self.free_energy(V_neg_oh) * seq_weights).sum() / seq_weights.sum()  # free energy of gibbs sampled visible states
        cd_loss = F_v - F_vp

        # Regularization Terms
        freg, wreg, dreg, bs_loss, gap_loss, reg_dict = self.regularization_terms()

        # Calc loss
        loss = cd_loss + freg + wreg + dreg + bs_loss + gap_loss

        logs = {"loss": loss,
                "free_energy_diff": cd_loss.detach(),
                "train_free_energy": F_v.detach(),
                **reg_dict
                }

        self.log("ptl/train_free_energy", logs["train_free_energy"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.training_data_logs.append(logs)
        return logs["loss"]

    def training_step_CD_free_energy(self, batch, batch_idx):
        inds, seqs, one_hot, seq_weights = batch

        V_neg_oh, h_neg, V_pos_oh, h_pos = self(one_hot)
        F_v = (self.free_energy(V_pos_oh) * seq_weights / seq_weights.sum())  # free energy of training data
        F_vp = (self.free_energy(V_neg_oh) * seq_weights / seq_weights.sum()) # free energy of gibbs sampled visible states
        cd_loss = (F_v - F_vp).mean()

        free_energy_log = {
            "free_energy_pos": F_v.sum().detach(),
            "free_energy_neg": F_vp.sum().detach(),
            "free_energy_diff": cd_loss.sum().detach(),
            "cd_loss": cd_loss.detach(),
        }

        # Regularization Terms
        freg, wreg, dreg, bs_loss, gap_loss, reg_dict = self.regularization_terms()

        # Calculate Loss
        loss = cd_loss + freg + wreg + dreg + bs_loss + gap_loss

        logs = {"loss": loss,
                "train_free_energy": cd_loss.sum().detach(),
                **free_energy_log,
                **reg_dict
                }

        self.log("ptl/train_free_energy", logs["train_free_energy"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.training_data_logs.append(logs)
        return logs["loss"]

    def init_temp_schedule(self, temps):
        if type(temps) is float:
            end_temp, start_temp = temps, temps
        elif type(temps) is list:
            start_temp, end_temp = temps
        vary_epoch = int(self.decay_after*self.epochs)
        stationary_epochs = self.epochs - vary_epoch
        vary_temps = torch.tensor(np.geomspace(start_temp, end_temp, vary_epoch), device=self.device)
        stat_temp = torch.full((stationary_epochs,), vary_temps[-1], device=self.device)
        return torch.concat([vary_temps, stat_temp], dim=0)

    def altered_sigmoid(self, x, c=10, s=0.5):
        return 1/(1+torch.exp(-c*(x-s)))

    def training_step_PCD_free_energy(self, batch, batch_idx):
        inds, seqs, one_hot, seq_weights = batch

        if self.current_epoch == 0 and batch_idx == 0:
            self.chain = torch.zeros((self.training_data.index.__len__(), *one_hot.shape[1:]), device=self.device)

        if self.current_epoch == 0:
            self.chain[inds] = one_hot.type(torch.get_default_dtype())

        V_oh_neg, h_neg, h_neg_inputs_flat, h_pos, sparsity_penalty, h_w = self.forward_PCD(inds)


        inputs_flat = torch.concat(self.compute_output_v(one_hot), 1)
        h_inputs_flat = torch.concat([torch.clamp(inputs_flat, min=0.), torch.clamp(inputs_flat, max=0.).abs()], 1)

        ps = torch.concat(h_w, dim=1)
        # ps = self.exp_activation(ps, y=1)
        # ps = ps/ps.max()

        with torch.no_grad():
            # seq_temp = self.seq_temp_schedule[self.current_epoch]
            # sw = self.energy(one_hot, self.sample_from_inputs_h(self.compute_output_v(one_hot)))
            # sw = self.free_energy(one_hot)
            #
            # sw += sw.min().abs()  # change to positive range with highest value as highest energy values
            # sw /= sw.max()  # change values to be between 0 and 1
            # sw = torch.softmax(sw/seq_temp, -1)
            # sw = self.altered_sigmoid(sw, c=100, s=0.75)
            # sw /= sw.max()

            pm, _ = ps.max(1)
            sw = (1 - pm).clamp(min=0)
            sw = self.altered_sigmoid(sw, c=80, s=0.75).abs()
            sw /= sw.max()

        # h_pos = [h.detach() for h in h_pos]
        # for iid, i in enumerate(self.hidden_convolution_keys):
        #     dot_prods = (torch.mm(h_pos[iid].T, h_pos[iid]) - torch.eye(h_pos[iid].shape[1])) / 2
        #     orthogonal_loss = dot_prods.sum() * self.lkd
        # kld_loss = (sw*self.kl_div(h_inputs_flat, h_neg_inputs_flat).abs().sum(1)).mean() * self.lkd
        # kld_loss = (sw*(h_inputs_flat - h_neg_inputs_flat).square().sum(1)).mean() * self.lkd
        # kld_loss = sparsity_penalty

        ### Individual Free Energy
        # F_v_ind = self.free_energy_ind(one_hot)
        # F_v_vals, F_v_inds = torch.topk(F_v_ind, 1, dim=1, largest=False, sorted=False)
        # F_v = F_v_vals.sum(1)
        #
        # F_vp_ind = self.free_energy_ind(V_oh_neg)
        # F_vp = F_vp_ind.gather(1, F_v_inds).sum(1)  #F_vp_ind[torch.arange(len(F_v_inds)), F_v_inds.squeeze(1)]

        ### Normal Way
        F_v = self.free_energy(one_hot)  # free energy of training data
        F_vp = self.free_energy(V_oh_neg)

        # F_v_total = F_v.sum().abs()
        # F_vp_total = F_vp.sum().abs()
        #
        # F_v = F_v * sw
        # F_vp = F_vp * sw
        #
        # F_v = F_v/F_v.sum().abs() * F_v_total
        # F_vp = F_vp/F_vp.sum().abs() * F_vp_total


        ### Input Clustering Way
        # all_flat_ws, hidden_node_parent = [], []
        #
        # start_w_count = 0
        # for iid, i in enumerate(self.hidden_convolution_keys):
        #     W = getattr(self, f"{i}_W")
        #     Wpos = torch.clamp(W, min=0.)
        #     Wneg = torch.clamp(W, max=0.)
        #
        #     ws = torch.concat([Wpos, Wneg.abs()], dim=0).squeeze(1)
        #     all_flat_ws.append(ws)
        #     hidden_node_parent.append((torch.arange(W.shape[0], device=self.device) + start_w_count).repeat(2))
        #     start_w_count += W.shape[0]
        #
        # hidden_node_parent = torch.concat(hidden_node_parent, dim=0)
        #
        # flat_ws = torch.concat(all_flat_ws, dim=0)
        # in_max_vals, in_max_inds = flat_ws.max(2)
        #
        # matching_percentages = h_inputs_flat.div(in_max_vals.sum(1)[None, :])
        # _, matching_weights = matching_percentages.max(-1)
        #
        # mp_reg = ps.norm(dim=1, p=1).mean() * self.lkd * 0.0
        # mp2_reg = matching_percentages.norm(dim=0, p=1).mean() * 0.0 # 0.0002
        #
        # F_v_ind = self.free_energy_ind(one_hot) * ps
        # F_vp_ind = self.free_energy_ind(V_oh_neg) * ps
        #
        # F_v = F_v_ind.sum(1)
        # F_vp = F_vp_ind.sum(1)
        # bimodal_h = self.bimodality_2d(F_v_ind) + self.bimodality_2d(F_vp_ind)
        # bimodal_loss = -1*bimodal_h.mean() * 0.5




        #
        # F_v = (F_v_ind * (ps > 0.25)).sum(-1)
        # F_vp = (F_vp_ind * (ps > 0.25)).sum(-1)

        # ind_contribution = torch.full_like(F_v_ind, 0.0)
        #
        # ind_contribution.index_add_(-1, hidden_node_parent, matching_percentages)
        #
        # ind_temp = self.ind_temp_schedule[self.current_epoch]
        # ps = torch.softmax(ps/ind_temp, -1)

        # ps = self.altered_sigmoid(ps, c=50, s=0.5)
        # ps = (ps - ps.mean(0)[None, :]).clamp(min=1e-6)

        # tk_vals, tk_inds = ps.topk(1) # doesn't work very well

        # F_v = (F_v_ind.gather(1, tk_inds)*tk_vals).sum(-1)
        # F_vp = (F_vp_ind.gather(1, tk_inds)*tk_vals).sum(-1)

        # F_v_total = F_v_ind.sum(1).abs()
        # F_vp_total = F_vp_ind.sum(1).abs()
        #
        # F_v = (F_v_ind * ps).sum(-1)
        # F_vp = (F_vp_ind * ps).sum(-1)



        # matching_weights = torch.softmax(matching_weights/temp, -1)

        # F_v = F_v * sw
        # F_vp = F_vp * sw
        #
        # F_v = F_v/F_v.sum().abs() * F_v_total
        # F_vp = F_vp/F_vp.sum().abs() * F_vp_total

        # F_v = F_v * F_v_total
        # F_vp = F_vp * F_vp_total

        kld_loss = self.excess_kurtosis(F_v) * self.lkd



        cd_loss = (F_v - F_vp).mean()

        # Regularization Terms
        freg, wreg, dreg, bs_loss, gap_loss, reg_dict = self.regularization_terms()


        # Calculate Loss
        loss = cd_loss + freg + wreg + dreg + bs_loss + gap_loss + kld_loss  # + bimodal_loss  # + sparsity_penalty #  mp_reg + mp2_reg # + input_loss

        if loss.isnan():
            print("okay")

        logs = {"loss": loss,
                "free_energy_diff": cd_loss.detach(),
                "free_energy_pos": F_v.mean().detach(),
                "free_energy_neg": F_vp.mean().detach(),
                # "kl_loss": kld_loss.detach(),
                #  "input_correlation_reg": input_loss.detach(),
                **reg_dict
                }

        self.log("ptl/free_energy_diff", logs["free_energy_diff"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=one_hot.shape[0])
        self.log("ptl/train_free_energy", logs["free_energy_pos"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=one_hot.shape[0])
        self.log("ptl/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=one_hot.shape[0])

        self.training_data_logs.append(logs)
        return logs["loss"]

    def excess_kurtosis(self, x):
        mean = torch.mean(x)
        diffs = (x - mean) + 1e-12
        std = torch.std(x) + 1e-12
        zscores = diffs/std
        return torch.mean(torch.pow(zscores, 4.0)) - 3

    def bimodality_2d(self, x):
        mean = torch.mean(x, 0)
        diffs = (x - mean) + 1e-12
        std = torch.std(x, 0) + 1e-12
        zscores = diffs / std
        skew = torch.mean(torch.pow(zscores, 3.0), 0)
        kurtosis = torch.mean(torch.pow(zscores, 4.0), 0)
        return (skew.square() + 1)/kurtosis  # sarle's bimodality constant


    def mask_2d_kurtosis(self, x, mask):
        n = mask.sum(0) + 2
        mean = (x * mask).sum(0) / n
        diffs = (x - mean + 1e-12) * mask
        stds = torch.sqrt(diffs.square().sum(0) * (1 / (n - 1))) + 1e-12
        zscores = (diffs / stds) + 1e-12
        kurt = torch.pow(zscores, 4.0).sum() / n - 3
        if kurt.isnan().any().item():
            print('damn it kurt')
        return kurt

    def pearson_loss(self, Ih, k=2):
        B, H = Ih.size()
        mean = Ih.mean(dim=0).unsqueeze(0)
        diffs = (Ih - mean)
        prods = torch.matmul(diffs.T, diffs)
        bcov = prods / (B - 1)  # Unbiased estimate
        # matrix of stds
        std = Ih.std(dim=0).unsqueeze(0)
        std_mat = torch.matmul(std.T, std) + 1e-6

        pearson_mat = (bcov / std_mat).abs()
        triu_pearson_mat = pearson_mat.triu(diagonal=1)

        self_interaction_term = torch.diagonal(pearson_mat, offset=H//2).sum()

        topvals, topinds = torch.topk(triu_pearson_mat, k, dim=1)

        other_interactions = pearson_mat.triu(diagonal=1).sum() - self_interaction_term - topvals.sum()*0.9

        return other_interactions*self.lcorr

    def normalize_inputs(self, h_inputs, key="all"):
        hi = h_inputs

        if key == "all":
            for iid, i in enumerate(self.hidden_convolution_keys):
                ws = getattr(self, f"{i}_W").squeeze(1)
                # all_ws.append(W)

                in_max_val_pos, _ = ws.clamp(min=0.).max(2)
                in_max_val_neg, _ = ws.clamp(max=0.).min(2)

                in_max_val_pos = in_max_val_pos.sum(1)[None, :]
                in_max_val_neg = in_max_val_neg.sum(1)[None, :]

                hi[iid][hi[iid] > 0] /= in_max_val_pos.expand(hi[iid].shape[0], -1)[hi[iid] > 0]
                hi[iid][hi[iid] < 0] /= in_max_val_neg.expand(hi[iid].shape[0], -1).abs()[hi[iid] < 0]
        else:
            ws = getattr(self, f"{key}_W").squeeze(1)

            in_max_val_pos, _ = ws.clamp(min=0.).max(2)
            in_max_val_neg, _ = ws.clamp(max=0.).min(2)

            in_max_val_pos = in_max_val_pos.sum(1)[None, :]
            in_max_val_neg = in_max_val_neg.sum(1)[None, :]

            hi[hi > 0] /= in_max_val_pos.expand(hi.shape[0], -1)[hi > 0]
            hi[hi < 0] /= in_max_val_neg.expand(hi.shape[0], -1).abs()[hi < 0]

        return hi

    def forward_PCD(self, inds):
        # Gibbs sampling with Persistent Contrastive Divergence
        fantasy_v = self.chain[inds]  # Last sample that was saved to self.chain variable, initialized in training step
        h_pos = self.sample_from_inputs_h(self.compute_output_v(fantasy_v))
        for _ in range(self.mc_moves - 1):
            fantasy_v, fantasy_h = self.markov_step(fantasy_v)

        V_neg, fantasy_h, hidden_input, h_w = self.markov_step_with_hidden_input(fantasy_v)

        flat_hidden_input = torch.cat(hidden_input, dim=1)
        hidden_inputs = [torch.clamp(flat_hidden_input, min=0.), torch.clamp(flat_hidden_input, max=0.).abs()]

        # input_loss = self.pearson_loss(torch.cat(hidden_inputs, dim=1))

        h_neg = self.sample_from_inputs_h(self.compute_output_v(V_neg))
        #
        sparsity_penalty = torch.tensor([0.], device=self.device)
        # for kid, key in enumerate(self.hidden_convolution_keys):
        #     shw = h_w[kid]
        #     mask = shw >= 0
        #     #
        #     pos_kurt = self.mask_2d_kurtosis(shw, mask)
        #     # neg_kurt = self.mask_2d_kurtosis(shw, ~mask)
        #     #
        #     sparsity_penalty += (pos_kurt+neg_kurt).mean() * self.lkd * 0.0

            # sparsity_penalty += torch.sum(h_w[kid].abs(), (-1)).square().mean()/(2*h_w[kid].shape[1])*self.lkd
            # sparsity_penalty += torch.sum(h_w[kid].abs(), (0)).square().mean()/(2*h_w[kid].shape[0])*self.lkd
            # sparsity_penalty += torch.sum(h_w[kid].abs(), 1).mean() * self.lkd
            # sparsity_penalty += torch.linalg.norm(h_w[kid], dim=1).mean()

        self.chain[inds] = V_neg.detach().type(torch.get_default_dtype())

        return V_neg, h_neg, torch.cat(hidden_inputs, dim=1), h_pos, sparsity_penalty, h_w

    def forward_PT(self, V_pos_ohe):
        # Initialize_PT is called before the forward function is called. Therefore, N_PT will be filled
        # Parallel Tempering
        n_chains = V_pos_ohe.shape[0]

        with torch.no_grad():
            fantasy_v = self.random_init_config_v(custom_size=(self.N_PT, n_chains))
            fantasy_h = self.random_init_config_h(custom_size=(self.N_PT, n_chains))
            fantasy_E = self.energy_PT(fantasy_v, fantasy_h, self.N_PT)

            for _ in range(self.mc_moves - 1):
                fantasy_v, fantasy_h, fantasy_E = self.markov_PT_and_exchange(fantasy_v, fantasy_h, fantasy_E,self.N_PT)
                self.update_betas(self.N_PT)

        fantasy_v, fantasy_h, fantasy_E = self.markov_PT_and_exchange(fantasy_v, fantasy_h, fantasy_E, self.N_PT)
        self.update_betas(self.N_PT)

        # V_neg, h_neg, V_pos, h_pos
        return fantasy_v[0], fantasy_h[0], V_pos_ohe, self.sample_from_inputs_h(self.compute_output_v(V_pos_ohe))

    def forward(self, V_pos_ohe):
        # Gibbs sampling
        fantasy_v, first_h = self.markov_step(V_pos_ohe)
        for _ in range(self.mc_moves - 1):
            fantasy_v, fantasy_h = self.markov_step(fantasy_v)

        return fantasy_v, fantasy_h, V_pos_ohe, first_h

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
                inds, seqs, one_hot, seq_weights = batch
                likelihood += self.likelihood(one_hot).detach().tolist()

        return X.sequence.tolist(), likelihood

    # Replace gap state contribution to 0 in all weights/biases
    def fill_gaps_in_parameters(self, fill=1e-6):
        with torch.no_grad():
            fields = getattr(self, "fields")
            fields[:, -1].fill_(fill)

            for iid, i in enumerate(self.hidden_convolution_keys):
                W = getattr(self, f"{i}_W")
                W[:, :, :, -1].fill_(fill)

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
            inds, seqs, one_hot, seq_weights = batch
            one_hot_v = Variable(one_hot.type(torch.get_default_dtype()), requires_grad=True)
            V_neg, h_neg, V_pos, h_pos = self(one_hot_v)
            weights = seq_weights
            F_v = (self.free_energy(V_pos) * weights).sum() / weights.sum()  # free energy of training data
            F_vp = (self.free_energy(V_neg) * weights).sum() / weights.sum()  # free energy of gibbs sampled visible states
            cd_loss = F_v - F_vp

            # Regularization Terms
            freg, wreg, dreg, bs_loss, gap_loss, reg_dict = self.regularization_terms()
            loss = cd_loss + freg + wreg + dreg + gap_loss + bs_loss
            loss.backward()

            saliency_maps.append(one_hot_v.grad.data.detach())

        return torch.cat(saliency_maps, dim=0)

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
                sys.exit(1)

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