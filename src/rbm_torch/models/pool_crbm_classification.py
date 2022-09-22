import time
import pandas as pd
import math
import json
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.profiler import SimpleProfiler, PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW, Adagrad, Adadelta  # Supported Optimizers
from multiprocessing import cpu_count # Just to set the worker number
from torch.autograd import Variable

from rbm_torch.models.pool_crbm import pool_CRBM

from rbm_torch.utils.utils import Categorical, fasta_read, conv2d_dim, pool1d_dim, BatchNorm1D  #Sequence_logo, gen_data_lowT, gen_data_zeroT, all_weights, Sequence_logo_all,



class pool_class_CRBM(pool_CRBM):
    def __init__(self, config, debug=False, precision="double", meminfo=False):
        super().__init__(config, debug=debug, precision=precision, meminfo=meminfo)

        self.alpha = config["alpha"]

        self.classes = config["classes"]
        self.register_parameter(f"y_bias", nn.Parameter(torch.zeros(self.classes, device=self.device)))
        self.register_parameter(f"0y_bias", nn.Parameter(torch.zeros(self.classes, device=self.device)))
        for key in self.hidden_convolution_keys:
            self.register_parameter(f"{key}_M", nn.Parameter(0.05 * torch.randn((self.convolution_topology[key]["number"], self.classes), device=self.device)))

    ## Unsupervised Loss Function
    def free_energy(self, v):
        return self.energy_v(v) - self.logpartition_h(self.compute_output_v_for_h(v))

    ## Discriminative Loss Function
    def free_energy_discriminative(self, v, y):
        probs = self.compute_class_probabilities(v)
        proby = probs[y, torch.arange(probs.size(1))]
        return -torch.log(proby)

    ## Not used but may be useful
    def free_energy_h(self, h):
        return self.energy_h(h) - self.logpartition_v(self.compute_output_h(h))

    ## Total Energy of a given visible and hidden configuration
    # def energy(self, v, h, y, remove_init=False, hidden_sub_index=-1):
    #     return self.energy_v(v, remove_init=remove_init) + self.energy_h(h, sub_index=hidden_sub_index, remove_init=remove_init) - \
    #            self.bidirectional_weight_term(v, h, hidden_sub_index=hidden_sub_index) + self.energy_y(y, remove_init=remove_init) + self.classification_weight_term(h, y, hidden_sub_index=hidden_sub_index)
    #
    # def energy_PT(self, v, h, y, N_PT, remove_init=False):
    #     # if N_PT is None:
    #     #     N_PT = self.N_PT
    #     E = torch.zeros((N_PT, v.shape[1]), device=self.device)
    #     for i in range(N_PT):
    #         E[i] = self.energy(v[i], h, y[i], remove_init=remove_init, hidden_sub_index=i)
    #         # E[i] = self.energy_v(v[i], remove_init=remove_init) + self.energy_h(h, sub_index=i, remove_init=remove_init) - self.bidirectional_weight_term(v[i], h, hidden_sub_index=i)
    #     return E


    # def classification_weight_term(self, h, y, hidden_sub_index=-1):
    #     # computes h*m term
    #     if hidden_sub_index != -1:
    #         h = [subh[hidden_sub_index] for subh in h]
    #
    #     hm = self.compute_output_h_for_y(h)
    #     return hm[torch.arange(hm.shape[0]), y]


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

    ## computes b(y) term of potential
    # config is labels y
    def energy_y(self, config, remove_init=False):
        if remove_init:
            ybias = getattr(self, "y_bias") - getattr(self, "0y_bias")
        else:
            ybias = getattr(self, "y_bias")
        return torch.index_select(ybias, 0, config)

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
            marginal[iid] = y  # 10 added so hidden layer has stronger effect on free energy, also in energy_h
        return marginal.sum(0)

    ## Marginal over visible units
    def logpartition_v(self, inputs, beta=1):
        if beta == 1:
            return torch.logsumexp(getattr(self, "fields")[None, :, :] + inputs, 2).sum(1)
        else:
            return torch.logsumexp((beta * getattr(self, "fields") + (1 - beta) * getattr(self, "fields0"))[None, :] + beta * inputs, 2).sum(1)

    def logpartition_y(self, inputs, beta=1):
        if beta == 1:
            return torch.logsumexp(getattr(self, "y_bias") + inputs, 1)
        else:
            return torch.logsumexp((beta * getattr(self, "y_bias") + (1 - beta) * getattr(self, "0y_bias")) + beta * inputs, 1)

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
    def compute_output_v_for_h(self, X):  # X is the one hot vector
        outputs = []

        self.max_inds = []
        self.min_inds = []
        for iid, i in enumerate(self.hidden_convolution_keys):
            weights = getattr(self, f"{i}_W")
            conv = F.conv2d(X.unsqueeze(1).type(torch.get_default_dtype()), weights, stride=self.convolution_topology[i]["stride"],
                            padding=self.convolution_topology[i]["padding"],
                            dilation=self.convolution_topology[i]["dilation"]).squeeze(3)

            max_pool, max_inds = self.pools[iid](conv.abs())

            max_conv_values = torch.gather(conv, 2, max_inds)
            max_pool *= max_conv_values/max_conv_values.abs()

            self.max_inds.append(max_inds)

            out = max_pool.squeeze(2)

            if self.use_batch_norm:
                batch_norm = getattr(self, f"batch_norm_{i}")  # get individual batch norm
                out = batch_norm(out)  # apply batch norm

            outputs.append(out)

        return outputs

    ## Compute Input for Visible Layer from Hidden dReLU
    def compute_output_h_for_v(self, Y):  # from h_uk (B, hidden_num)
        outputs = []
        nonzero_masks = []
        for iid, i in enumerate(self.hidden_convolution_keys):
            reconst = self.unpools[iid](Y[iid].unsqueeze(2), self.max_inds[iid])


            # convx = self.convolution_topology[i]["convolution_dims"][2]
            outputs.append(F.conv_transpose2d((reconst).unsqueeze(3), getattr(self, f"{i}_W"),
                                              stride=self.convolution_topology[i]["stride"],
                                              padding=self.convolution_topology[i]["padding"],
                                              dilation=self.convolution_topology[i]["dilation"],
                                              output_padding=self.convolution_topology[i]["output_padding"]).squeeze(1))

            nonzero_masks.append((outputs[-1] != 0.).type(torch.get_default_dtype()))  # * getattr(self, "hidden_layer_W")[iid])  # Used for calculating mean of outputs, don't want zeros to influence mean

        if len(outputs) > 1:
            # Returns mean output from all hidden layers, zeros are ignored
            mean_denominator = torch.sum(torch.stack(nonzero_masks), 0) + 1e-6
            return torch.sum(torch.stack(outputs), 0) / mean_denominator
        else:
            return outputs[0]

    def compute_output_h_for_y(self, H):
        # h is list of tensors with shapes (batch_size, h_num)
        outputs = []
        for kid, key in enumerate(self.hidden_convolution_keys):
            outputs.append(torch.matmul(H[kid], getattr(self, f"{key}_M")))

        return torch.stack(outputs, dim=1).sum(1)

    ## Computes h M term
    def compute_output_y_for_h(self, Y, all_classes=False):
        # from y our class labels  or regression values?
        # calculates M*Y
        outputs = []
        for key in self.hidden_convolution_keys:
            if all_classes:
                outputs.append(torch.swapaxes(getattr(self, f"{key}_M"), 1, 0).expand(Y.shape[0], -1, -1))
            else:
                outputs.append(torch.index_select(torch.swapaxes(getattr(self, f"{key}_M"), 1, 0), 0, Y))
        return outputs

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

    def sample_from_inputs_y(self, psi):
        # input psi is shape (batch_size, classes), should be equal to h*M
        probs = F.softmax(psi + getattr(self, "y_bias").unsqueeze(0), dim=1)
        return probs.argmax(1)

    ###################################################### Sampling Functions
    ## Samples hidden from visible and vice versa, returns newly sampled hidden and visible
    # def markov_step(self, v, y, beta=1):
    #     # Gibbs Sampler
    #     h_input_v = self.compute_output_v_for_h(v)
    #     h_input_y = self.compute_output_y_for_h(y)
    #     h_input = [torch.add(h_input_v[i], h_input_y[i]) for i in range(len(h_input_v))]
    #     h = self.sample_from_inputs_h(h_input, beta=beta)
    #     return self.sample_from_inputs_v(self.compute_output_h_for_v(h), beta=beta), h, self.sample_from_inputs_y(self.compute_output_h_for_y(h))

    # def markov_PT_and_exchange(self, v, h, y, e, N_PT):
    #     for i, beta in zip(torch.arange(N_PT), self.betas):
    #         v[i], htmp, y[i] = self.markov_step(v[i], y[i], beta=beta)
    #         for hid in range(len(self.hidden_convolution_keys)):
    #             h[hid][i] = htmp[hid]
    #         e[i] = self.energy(v[i], h, y[i], hidden_sub_index=i)
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
    #             y[i:i + 2, swap] = torch.flip(y[i - 1: i + 1], [0])[:, swap]
    #             for hid in range(len(self.hidden_convolution_keys)):
    #                 h[hid][i:i + 2, swap] = torch.flip(h[hid][i - 1: i + 1], [0])[:, swap]
    #             # h[i:i + 2, swap] = torch.flip(h[i - 1: i + 1], [0])[:, swap]
    #             e[i:i + 2, swap] = torch.flip(e[i - 1: i + 1], [0])[:, swap]
    #             if self.record_swaps:
    #                 particle_id[i:i + 2, swap] = torch.flip(particle_id[i - 1: i + 1], [0])[:, swap]
    #         else:
    #             v[i:i + 2, swap] = torch.flip(v[:i + 1], [0])[:, swap]
    #             y[i:i + 2, swap] = torch.flip(y[:i + 1], [0])[:, swap]
    #             for hid in range(len(self.hidden_convolution_keys)):
    #                 h[hid][i:i + 2, swap] = torch.flip(h[hid][:i + 1], [0])[:, swap]
    #             e[i:i + 2, swap] = torch.flip(e[:i + 1], [0])[:, swap]
    #             if self.record_swaps:
    #                 particle_id[i:i + 2, swap] = torch.flip(particle_id[:i + 1], [0])[:, swap]
    #
    #         if self.record_acceptance:
    #             self.acceptance_rates[i] = swap.type(torch.get_default_dtype()).mean()
    #             self.mav_acceptance_rates[i] = self.mavar_gamma * self.mav_acceptance_rates[i] + self.acceptance_rates[
    #                 i] * (1 - self.mavar_gamma)
    #
    #     if self.record_swaps:
    #         self.particle_id.append(particle_id)
    #
    #     self.count_swaps += 1
    #     return v, h, y, e

    ######################################################### Pytorch Lightning Functions
    ## Not yet rewritten for crbm

    def training_step_CD_free_energy(self, batch, batch_idx):
        seqs, one_hot, seq_weights, labels = batch

        # if self.meminfo:
        #     print("GPU Allocated Training Step Start:", torch.cuda.memory_allocated(0))
        # V_neg_oh, h_neg, y_neg, V_pos_oh, h_pos, y_pos = self(one_hot, labels)
        V_neg_oh, h_neg, V_pos_oh, h_pos = self(one_hot)

        # F_v = (self.free_energy(V_pos_oh) * weights).sum()  # free energy of training data
        # free_energy = self.free_energy(V_pos_oh)
        F_v = (self.free_energy(V_pos_oh) * seq_weights).sum() / seq_weights.sum()  # free energy of training data
        # F_vp = (self.free_energy(V_neg_oh) * weights.abs()).sum() # free energy of gibbs sampled visible states
        F_vp = (self.free_energy(V_neg_oh) * seq_weights.abs()).sum() / seq_weights.sum()  # free energy of gibbs sampled visible states
        free_energy_diff = F_v - F_vp
        # cd_loss = (free_energy_diff/torch.abs(free_energy_diff)) * torch.log(torch.abs(free_energy_diff) + 10)
        unsupervised_cd_loss = free_energy_diff

        D_v = self.free_energy_discriminative(V_pos_oh, labels).mean()
        # D_vp = self.free_energy_discriminative(V_neg_oh, y_neg).mean()
        discriminative_cd_loss = D_v

        hybrid_loss_function = 20*((1 + self.alpha) * discriminative_cd_loss + self.alpha * unsupervised_cd_loss)

        # if self.meminfo:
        #     print("GPU Allocated After CD_Loss:", torch.cuda.memory_allocated(0))

        # Regularization Terms
        reg1 = self.lf / (2 * self.v_num * self.q) * getattr(self, "fields").square().sum((0, 1))
        reg2 = torch.zeros((1,), device=self.device)
        reg3 = torch.zeros((1,), device=self.device)

        bs_loss = torch.zeros((1,), device=self.device)  # encourages weights to use both positive and negative contributions
        gap_loss = torch.zeros((1,), device=self.device)  # discourages high values for gaps
        div_loss = torch.zeros((1,), device=self.device)  # discourages weights with long range similarities

        for iid, i in enumerate(self.hidden_convolution_keys):
            W_shape = self.convolution_topology[i]["weight_dims"]  # (h_num,  input_channels, kernel0, kernel1)
            W = getattr(self, f"{i}_W")
            x = torch.sum(torch.abs(W), (3, 2, 1)).square()
            reg2 += x.mean() * self.l1_2 / (2 * W_shape[1] * W_shape[2] * W_shape[3])
            reg3 += self.ld / ((getattr(self, f"{i}_W").abs() - getattr(self, f"{i}_W").squeeze(1).abs()).abs().sum((1, 2, 3)).mean() + 1)
            gap_loss += self.lgap * W[:, :, :, -1].abs().sum()

            denom = torch.sum(torch.abs(W), (3, 2, 1))
            zeroW = torch.zeros_like(W, device=self.device)
            Wpos = torch.maximum(W, zeroW)
            Wneg = torch.minimum(W, zeroW)
            bs_loss += self.lbs * torch.abs(Wpos.sum((1, 2, 3)) / denom - torch.abs(Wneg.sum((1, 2, 3))) / denom).mean()


        label_probs = self.compute_class_probabilities(V_pos_oh)
        predicted_labels = label_probs.argmax(0)
        # class_loss = self.closs(torch.swapaxes(label_probs, 0, 1), labels)


        # Calculate Loss
        loss = hybrid_loss_function + reg1 + reg2 + reg3 + gap_loss + bs_loss  # + class_loss

        # Debugging
        # nancheck = torch.isnan(torch.tensor([cd_loss, F_v, F_vp, reg1, reg2, reg3], device=self.device))
        # if True in nancheck:
        #     print(nancheck)
        #     torch.save(V_pos_oh, "vpos_err.pt")
        #     torch.save(V_neg_oh, "vneg_err.pt")
        #     torch.save(one_hot, "oh_err.pt")
        #     torch.save(seq_weights, "seq_weights_err.pt")

        # Calculate Loss
        # loss = cd_loss + reg1 + reg2 + reg3

        logs = {"loss": loss,
                "free_energy_diff": hybrid_loss_function.detach(),
                "train_free_energy": F_v.detach(),
                "field_reg": reg1.detach(),
                "weight_reg": reg2.detach(),
                "distance_reg": reg3.detach()
                }

        # y_input = self.compute_output_h_for_y(h_pos)
        # predicted_labels = self.sample_from_inputs_y(y_input)

        acc = (predicted_labels == labels).double().mean()
        logs["acc"] = acc.detach()
        self.log("ptl/train_acc", logs["acc"], on_step=True, on_epoch=True, prog_bar=True, logger=True)


        self.log("ptl/train_free_energy", logs["train_free_energy"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return logs

    def validation_step(self, batch, batch_idx):

        seqs, one_hot, seq_weights, labels = batch


        # pseudo_likelihood = (self.pseudo_likelihood(one_hot) * seq_weights).sum() / seq_weights.sum()
        free_energy = self.free_energy(one_hot)
        free_energy_avg = (free_energy * seq_weights).sum() / seq_weights.abs().sum()


        batch_out = {
            # "val_pseudo_likelihood": pseudo_likelihood.detach()
            "val_free_energy": free_energy_avg.detach()
        }

        predicted_labels = self.label_prediction(one_hot)
        acc = (predicted_labels == labels).double().mean()
        batch_out["acc"] = acc.detach()
        self.log("ptl/val_acc", batch_out["acc"], on_step=True, on_epoch=True, prog_bar=True, logger=True)


        # logging on step, for whatever reason allocates 512 bytes on gpu after every epoch.
        self.log("ptl/val_free_energy", batch_out["val_free_energy"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #
        return batch_out

        ## Loads Training Data

    def train_dataloader(self, init_fields=True):
        # Get Correct Weights
        if "seq_count" in self.training_data.columns:
            training_weights = self.training_data["seq_count"].tolist()
        else:
            training_weights = None

        labels = True

        train_reader = Categorical(self.training_data, self.q, weights=training_weights, max_length=self.v_num,
                                   molecule=self.molecule, device=self.device, one_hot=True, labels=labels)

        # initialize fields from data
        if init_fields:
            with torch.no_grad():
                initial_fields = train_reader.field_init()
                self.fields += initial_fields
                self.fields0 += initial_fields

            # Performance was almost identical whether shuffling or not
        if self.sample_type == "pcd":
            shuffle = False
        else:
            shuffle = True

        return torch.utils.data.DataLoader(
            train_reader,
            batch_size=self.batch_size,
            num_workers=self.worker_num,  # Set to 0 if debug = True
            pin_memory=self.pin_mem,
            shuffle=shuffle
        )

    def val_dataloader(self):
        # Get Correct Validation weights
        if "seq_count" in self.validation_data.columns:
            validation_weights = self.validation_data["seq_count"].tolist()
        else:
            validation_weights = None

        labels = True

        val_reader = Categorical(self.validation_data, self.q, weights=validation_weights, max_length=self.v_num,
                                 molecule=self.molecule, device=self.device, one_hot=True, labels=labels)

        return torch.utils.data.DataLoader(
            val_reader,
            batch_size=self.batch_size,
            num_workers=self.worker_num,  # Set to 0 to view tensors while debugging
            pin_memory=self.pin_mem,
            shuffle=False
        )

    # def forward(self, V_pos_ohe, y_pos):
    #     if self.sample_type == "gibbs":
    #         # Gibbs sampling
    #         # pytorch lightning handles the device
    #         with torch.no_grad():  # only use last sample for gradient calculation, Enabled to minimize memory usage, hopefully won't have much effect on performance
    #             first_v, first_h, first_y = self.markov_step(V_pos_ohe, y_pos)
    #             V_neg = first_v.clone()
    #             fantasy_y = first_y.clone()
    #             if self.mc_moves - 2 > 0:
    #                 for _ in range(self.mc_moves - 2):
    #                     V_neg, fantasy_h, fantasy_y = self.markov_step(V_neg, fantasy_y)
    #
    #         V_neg, fantasy_h, fantasy_y = self.markov_step(V_neg, fantasy_y)
    #
    #         # V_neg, h_neg, V_pos, h_pos
    #         return V_neg, fantasy_h, fantasy_y, V_pos_ohe, first_h, y_pos
    #
    #     elif self.sample_type == "pt":
    #         # Initialize_PT is called before the forward function is called. Therefore, N_PT will be filled
    #
    #         # Parallel Tempering
    #         n_chains = V_pos_ohe.shape[0]
    #
    #         with torch.no_grad():
    #             fantasy_v = self.random_init_config_v(custom_size=(self.N_PT, n_chains))
    #             fantasy_h = self.random_init_config_h(custom_size=(self.N_PT, n_chains))
    #             fantasy_E = self.energy_PT(fantasy_v, fantasy_h, self.N_PT)
    #
    #             for _ in range(self.mc_moves - 1):
    #                 fantasy_v, fantasy_h, fantasy_E = self.markov_PT_and_exchange(fantasy_v, fantasy_h, fantasy_E, self.N_PT)
    #                 self.update_betas(self.N_PT)
    #
    #         fantasy_v, fantasy_h, fantasy_E = self.markov_PT_and_exchange(fantasy_v, fantasy_h, fantasy_E, self.N_PT)
    #         self.update_betas(self.N_PT)
    #
    #         # V_neg, h_neg, V_pos, h_pos
    #         return fantasy_v[0], fantasy_h[0], V_pos_ohe, self.sample_from_inputs_h(self.compute_output_v(V_pos_ohe))

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
            likelihood_preds, label_preds = [], []
            for i, batch in enumerate(data_loader):
                seqs, one_hot, seq_weights = batch
                predicted_labels = self.label_prediction(one_hot).detach().tolist()
                label_preds += predicted_labels
                likelihood_preds += self.likelihood(one_hot).detach().tolist()

        return X.sequence.tolist(), likelihood_preds, label_preds

    def compute_class_probabilities(self, v):
        h_input_v = self.compute_output_v_for_h(v)
        y = torch.zeros((v.shape[0]))
        h_input_all_classes = self.compute_output_y_for_h(y, all_classes=True)
        numerator = torch.zeros((self.classes, v.shape[0]), device=self.device)
        for c in range(self.classes):
            h_in = [torch.add(h_input_v[i], h_input_all_classes[i][:, c]) for i in range(len(h_input_v))]
            numerator[c] = torch.index_select(getattr(self, "y_bias"), 0, torch.tensor(c, device=self.device).expand(h_in[0].shape[0])) + self.logpartition_h(h_in)
        return F.softmax(numerator, dim=0)

    def label_prediction(self, v):
        probs = self.compute_class_probabilities(v)
        return probs.argmax(0)

    def random_init_config_y(self, custom_size=False, zeros=False):
        if custom_size:
            size = custom_size
        else:
            size = self.batch_size

        if zeros:
            return torch.zeros(size, device=self.device)
        else:
            return torch.rand((self.classes, *size), device=self.device)

    # def AIS(self, M=10, n_betas=10000, batches=None, verbose=0, beta_type='adaptive'):
    #     with torch.no_grad():
    #         if beta_type == 'linear':
    #             betas = torch.arange(n_betas, device=self.device) / torch.tensor(n_betas - 1, dtype=torch.float64, device=self.device)
    #         elif beta_type == 'root':
    #             betas = torch.sqrt(torch.arange(n_betas, device=self.device)) / torch.tensor(n_betas - 1, dtype=torch.float64, device=self.device)
    #         elif beta_type == 'adaptive':
    #             Nthermalize = 200
    #             Nchains = 20
    #             N_PT = 11
    #             self.adaptive_PT_lr = 0.05
    #             self.adaptive_PT_decay = True
    #             self.adaptive_PT_lr_decay = 10 ** (-1 / float(Nthermalize))
    #             if verbose:
    #                 t = time.time()
    #                 print('Learning betas...')
    #             self.gen_data(N_PT=N_PT, Nchains=Nchains, Lchains=1, Nthermalize=Nthermalize, update_betas=True)
    #             if verbose:
    #                 print('Elapsed time: %s, Acceptance rates: %s' % (time.time() - t, self.mav_acceptance_rates))
    #             betas = []
    #             sparse_betas = self.betas.flip(0)
    #             for i in range(N_PT - 1):
    #                 betas += list(sparse_betas[i] + (sparse_betas[i + 1] - sparse_betas[i]) * torch.arange(n_betas / (N_PT - 1), device=self.device) / (n_betas / (N_PT - 1) - 1))
    #             betas = torch.tensor(betas, device=self.device)
    #             # if verbose:
    #             # import matplotlib.pyplot as plt
    #             # plt.plot(betas); plt.title('Interpolating temperatures');plt.show()
    #
    #         # Initialization.
    #         log_weights = torch.zeros(M, device=self.device)
    #         # config = self.gen_data(Nchains=M, Lchains=1, Nthermalize=0, beta=0)
    #
    #         config = [self.sample_from_inputs_v(self.random_init_config_v(custom_size=(M,))),
    #                   self.sample_from_inputs_h(self.random_init_config_h(custom_size=(M,))),
    #                   self.sample_from_inputs_y(self.random_init_config_y(custom_size=(M,)))]
    #
    #         log_Z_init = torch.zeros(1, device=self.device)
    #
    #         log_Z_init += self.logpartition_h(self.random_init_config_h(custom_size=(1,), zeros=True), beta=0)
    #         log_Z_init += self.logpartition_v(self.random_init_config_v(custom_size=(1,), zeros=True), beta=0)
    #         log_Z_init += self.logpartition_y(self.random_init_config_y(custom_size=(1,), zeros=True), beta=0)
    #
    #         if verbose:
    #             print(f'Initial evaluation: log(Z) = {log_Z_init.data}')
    #
    #         for i in range(1, n_betas):
    #             if verbose:
    #                 if (i % 2000 == 0):
    #                     print(f'Iteration {i}, beta: {betas[i]}')
    #                     print('Current evaluation: log(Z)= %s +- %s' % ((log_Z_init + log_weights).mean(), (log_Z_init + log_weights).std() / np.sqrt(M)))
    #
    #             config[0], config[1], config[2] = self.markov_step(config[0], config[2])
    #             energy = self.energy(config[0], config[1], config[2])
    #             log_weights += -(betas[i] - betas[i - 1]) * energy
    #         self.log_Z_AIS = (log_Z_init + log_weights).mean()
    #         self.log_Z_AIS_std = (log_Z_init + log_weights).std() / np.sqrt(M)
    #         if verbose:
    #             print('Final evaluation: log(Z)= %s +- %s' % (self.log_Z_AIS, self.log_Z_AIS_std))
    #         return self.log_Z_AIS, self.log_Z_AIS_std

    # def likelihood(self, data, labels, recompute_Z=False):
    #     if (not hasattr(self, 'log_Z_AIS')) | recompute_Z:
    #         self.AIS()
    #     return -((1+self.alpha) * self.free_energy_discriminative(data, labels) + self.alpha*self.free_energy(data)) - self.log_Z_AIS

    # def gen_data(self, Nchains=10, Lchains=100, Nthermalize=0, Nstep=1, N_PT=1, config_init=[], beta=1, batches=None, reshape=True, record_replica=False, record_acceptance=None, update_betas=None, record_swaps=False):
    #     """
    #     Generate Monte Carlo samples from the RBM. Starting from random initial conditions, Gibbs updates are performed to sample from equilibrium.
    #     Inputs :
    #         Nchains (10): Number of Markov chains
    #         Lchains (100): Length of each chain
    #         Nthermalize (0): Number of Gibbs sampling steps to perform before the first sample of a chain.
    #         Nstep (1): Number of Gibbs sampling steps between each sample of a chain
    #         N_PT (1): Number of Monte Carlo Exchange replicas to use. This==useful if the mixing rate==slow. Watch self.acceptance_rates_g to check that it==useful (acceptance rates about 0==useless)
    #         batches (10): Number of batches. Must divide Nchains. higher==better for speed (more vectorized) but more RAM consuming.
    #         reshape (True): If True, the output==(Nchains x Lchains, n_visibles/ n_hiddens) (chains mixed). Else, the output==(Nchains, Lchains, n_visibles/ n_hiddens)
    #         config_init ([]). If not [], a Nchains X n_visibles numpy array that specifies initial conditions for the Markov chains.
    #         beta (1): The inverse temperature of the model.
    #     """
    #     with torch.no_grad():
    #         self.initialize_PT(N_PT, n_chains=Nchains, record_acceptance=record_acceptance, record_swaps=record_swaps)
    #
    #         if batches == None:
    #             batches = Nchains
    #         n_iter = int(Nchains / batches)
    #         Ndata = Lchains * batches
    #         if record_replica:
    #             reshape = False
    #
    #         if (N_PT > 1):
    #             if record_acceptance == None:
    #                 record_acceptance = True
    #
    #             if update_betas == None:
    #                 update_betas = False
    #
    #             # if record_acceptance:
    #             #     self.mavar_gamma = 0.95
    #
    #             if update_betas:
    #                 record_acceptance = True
    #                 # self.update_betas_lr = 0.1
    #                 # self.update_betas_lr_decay = 1
    #         else:
    #             record_acceptance = False
    #             update_betas = False
    #
    #         if (N_PT > 1) and record_replica:
    #             visible_data = self.random_init_config_v(custom_size=(Nchains, N_PT, Lchains), zeros=True)
    #             hidden_data = self.random_init_config_h(custom_size=(Nchains, N_PT, Lchains), zeros=True)
    #             y_data = self.random_init_config_y(custom_size=(Nchains, N_PT, Lchains), zeros=True)
    #             data = [visible_data, hidden_data, y_data]
    #         else:
    #             visible_data = self.random_init_config_v(custom_size=(Nchains, Lchains), zeros=True)
    #             hidden_data = self.random_init_config_h(custom_size=(Nchains, Lchains), zeros=True)
    #             y_data = self.random_init_config_y(custom_size=(Nchains, Lchains), zeros=True)
    #             data = [visible_data, hidden_data, y_data]
    #
    #         if config_init is not []:
    #             if type(config_init) == torch.tensor:
    #                 h_layer = self.random_init_config_h()
    #                 y_layer = self.random_init_config_y()
    #                 config_init = [config_init, h_layer, y_layer]
    #
    #         for i in range(n_iter):
    #             if config_init == []:
    #                 config = self._gen_data(Nthermalize, Ndata, Nstep, N_PT=N_PT, batches=batches, reshape=False, beta=beta,
    #                                         record_replica=record_replica, record_acceptance=record_acceptance, update_betas=update_betas, record_swaps=record_swaps)
    #             else:
    #                 config_init = [config_init[0][batches * i:batches * (i + 1)], config_init[1][batches * i:batches * (i + 1)]]
    #                 config = self._gen_data(Nthermalize, Ndata, Nstep, N_PT=N_PT, batches=batches, reshape=False, beta=beta,
    #                                         record_replica=record_replica, config_init=config_init, record_acceptance=record_acceptance,
    #                                         update_betas=update_betas, record_swaps=record_swaps)
    #
    #             if (N_PT > 1) & record_replica:
    #                 data[0][batches * i:batches * (i + 1)] = torch.swapaxes(config[0], 0, 2).clone()
    #                 data[2][batches * i:batches * (i + 1)] = torch.swapaxes(config[2], 0, 2).clone()
    #                 for hid in range(len(self.hidden_convolution_keys)):
    #                     data[1][hid][batches * i:batches * (i + 1)] = torch.swapaxes(config[1][hid], 0, 2).clone()
    #             else:
    #                 data[0][batches * i:batches * (i + 1)] = torch.swapaxes(config[0], 0, 1).clone()
    #                 data[2][batches * i:batches * (i + 1)] = torch.swapaxes(config[2], 0, 1).clone()
    #                 for hid in range(len(self.hidden_convolution_keys)):
    #                     data[1][hid][batches * i:batches * (i + 1)] = torch.swapaxes(config[1][hid], 0, 1).clone()
    #
    #         if reshape:
    #             return [data[0].flatten(0, -3), [hd.flatten(0, -3) for hd in data[1]], data[2]]
    #         else:
    #             return data
    #
    # def _gen_data(self, Nthermalize, Ndata, Nstep, N_PT=1, batches=1, reshape=True, config_init=[], beta=1, record_replica=False, record_acceptance=True, update_betas=False, record_swaps=False):
    #     with torch.no_grad():
    #
    #         if N_PT > 1:
    #             if update_betas or len(self.betas) != N_PT:
    #                 self.betas = torch.flip(torch.arange(N_PT) / (N_PT - 1) * beta, [0])
    #
    #             self.acceptance_rates = torch.zeros(N_PT - 1)
    #             self.mav_acceptance_rates = torch.zeros(N_PT - 1)
    #
    #         self.count_swaps = 0
    #         self.record_swaps = record_swaps
    #
    #         if self.record_swaps:
    #             self.particle_id = [torch.arange(N_PT)[:, None].repeat(batches, dim=1)]
    #
    #         Ndata /= batches
    #         Ndata = int(Ndata)
    #
    #         if config_init != []:
    #             config = config_init
    #         else:
    #             if N_PT > 1:
    #                 config = [self.random_init_config_v(custom_size=(N_PT, batches)), self.random_init_config_h(custom_size=(N_PT, batches)), self.random_init_config_y(custom_size=(N_PT, batches))]
    #             else:
    #                 config = [self.random_init_config_v(custom_size=(batches,)), self.random_init_config_h(custom_size=(batches,)), self.random_init_config_y(custom_size=(batches))]
    #
    #         for _ in range(Nthermalize):
    #             if N_PT > 1:
    #                 energy = self.energy_PT(config[0], config[1], config[2], N_PT)
    #                 config[0], config[1], config[2], energy = self.markov_PT_and_exchange(config[0], config[1], config[2], energy, N_PT)
    #                 if update_betas:
    #                     self.update_betas(N_PT, beta=beta)
    #             else:
    #                 config[0], config[1], config[2] = self.markov_step(config[0], config[2], beta=beta)
    #
    #         if N_PT > 1:
    #             if record_replica:
    #                 data = [config[0].clone().unsqueeze(0), self.clone_h(config[1], expand_dims=[0]), config[2].clone().unsqueeze(0)]
    #             else:
    #                 data = [config[0][0].clone().unsqueeze(0), self.clone_h(config[1], expand_dims=[0], sub_index=0), config[2][0].clone().unsqueeze(0)]
    #         else:
    #             data = [config[0].clone().unsqueeze(0), self.clone_h(config[1], expand_dims=[0]), config[2].clone().unsqueeze(0)]
    #
    #         if N_PT > 1:
    #             if Ndata > 1:
    #                 if record_replica:
    #                     data_gen_v = self.random_init_config_v(custom_size=(Ndata, N_PT, batches), zeros=True)
    #                     data_gen_h = self.random_init_config_h(custom_size=(Ndata, N_PT, batches), zeros=True)
    #                     data_gen_y = self.random_init_config_y(custom_size=(Ndata, N_PT, batches), zeros=True)
    #                     data_gen_v[0] = config[0].clone()
    #                     data_gen_y[0] = config[2].clone()
    #
    #                     clone = self.clone_h(config[1])
    #                     for hid in range(len(self.hidden_convolution_keys)):
    #                         data_gen_h[hid][0] = clone[hid]
    #                 else:
    #                     data_gen_v = self.random_init_config_v(custom_size=(Ndata, batches), zeros=True)
    #                     data_gen_h = self.random_init_config_h(custom_size=(Ndata, batches), zeros=True)
    #                     data_gen_y = self.random_init_config_y(custom_size=(Ndata, batches), zeros=True)
    #                     data_gen_v[0] = config[0][0].clone()
    #                     data_gen_y[0] = config[2][0].clone()
    #
    #                     clone = self.clone_h(config[1], sub_index=0)
    #                     for hid in range(len(self.hidden_convolution_keys)):
    #                         data_gen_h[hid][0] = clone[hid]
    #         else:
    #             data_gen_v = self.random_init_config_v(custom_size=(Ndata, batches), zeros=True)
    #             data_gen_h = self.random_init_config_h(custom_size=(Ndata, batches), zeros=True)
    #             data_gen_y = self.random_init_config_y(custom_size=(Ndata, batches), zeros=True)
    #             data_gen_v[0] = config[0].clone()
    #             data_gen_y[0] = config[2].clone()
    #
    #             clone = self.clone_h(config[1])
    #             for hid in range(len(self.hidden_convolution_keys)):
    #                 data_gen_h[hid][0] = clone[hid]
    #
    #         for n in range(Ndata - 1):
    #             for _ in range(Nstep):
    #                 if N_PT > 1:
    #                     energy = self.energy_PT(config[0], config[1], config[2], N_PT)
    #                     config[0], config[1], config[2], energy = self.markov_PT_and_exchange(config[0], config[1], config[2], energy, N_PT)
    #                     if update_betas:
    #                         self.update_betas(N_PT, beta=beta)
    #                 else:
    #                     config[0], config[1], config[2] = self.markov_step(config[0], config[2], beta=beta)
    #
    #             if N_PT > 1 and Ndata > 1:
    #                 if record_replica:
    #                     data_gen_v[n + 1] = config[0].clone()
    #                     data_gen_y[n + 1] = config[2].clone()
    #
    #                     clone = self.clone_h(config[1])
    #                     for hid in range(len(self.hidden_convolution_keys)):
    #                         data_gen_h[hid][n + 1] = clone[hid]
    #
    #                 else:
    #                     data_gen_v[n + 1] = config[0][0].clone()
    #                     data_gen_y[n + 1] = config[2][0].clone()
    #
    #                     clone = self.clone_h(config[1], sub_index=0)
    #                     for hid in range(len(self.hidden_convolution_keys)):
    #                         data_gen_h[hid][n + 1] = clone[hid]
    #
    #             else:
    #                 data_gen_v[n + 1] = config[0].clone()
    #                 data_gen_y[n + 1] = config[2].clone()
    #
    #                 clone = self.clone_h(config[1])
    #                 for hid in range(len(self.hidden_convolution_keys)):
    #                     data_gen_h[hid][n + 1] = clone[hid]
    #
    #         if Ndata > 1:
    #             data = [data_gen_v, data_gen_h, data_gen_y]
    #
    #         if self.record_swaps:
    #             print('cleaning particle trajectories')
    #             positions = torch.tensor(self.particle_id)
    #             invert = torch.zeros([batches, Ndata, N_PT])
    #             for b in range(batches):
    #                 for i in range(Ndata):
    #                     for k in range(N_PT):
    #                         invert[b, i, k] = torch.nonzero(positions[i, :, b] == k)[0]
    #             self.particle_id = invert
    #             self.last_at_zero = torch.zeros([batches, Ndata, N_PT])
    #             for b in range(batches):
    #                 for i in range(Ndata):
    #                     for k in range(N_PT):
    #                         tmp = torch.nonzero(self.particle_id[b, :i, k] == 0)[0]
    #                         if len(tmp) > 0:
    #                             self.last_at_zero[b, i, k] = i - 1 - tmp.max()
    #                         else:
    #                             self.last_at_zero[b, i, k] = 1000
    #             self.last_at_zero[:, 0, 0] = 0
    #
    #             self.trip_duration = torch.zeros([batches, Ndata])
    #             for b in range(batches):
    #                 for i in range(Ndata):
    #                     self.trip_duration[b, i] = self.last_at_zero[b, i, np.nonzero(invert[b, i, :] == 9)[0]]
    #
    #         if reshape:
    #             data[0] = data[0].flatten(0, -3)
    #             data[1] = [hd.flatten(0, -3) for hd in data[1]]
    #             data[2] = data[2].flatten(0, -3)
    #         else:
    #             data[0] = data[0]
    #             data[1] = data[1]
    #             data[2] = data[2]
    #
    #         return data

    def on_before_zero_grad(self, optimizer):
        super().on_before_zero_grad(optimizer)
        with torch.no_grad():
            for key in self.hidden_convolution_keys:
                getattr(self, f"{key}_M").data.clamp_(0.0, 1.0)
