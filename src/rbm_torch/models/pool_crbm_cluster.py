import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from rbm_torch.utils.utils import Categorical, fasta_read, conv2d_dim, pool1d_dim, BatchNorm1D, label_samples, process_weights, configure_optimizer, StratifiedBatchSampler, WeightedSubsetRandomSampler  #Sequence_logo, gen_data_lowT, gen_data_zeroT, all_weights, Sequence_logo_all,
from rbm_torch.utils.kmeans import kmeans
from torch.utils.data import WeightedRandomSampler
from rbm_torch.models.base import Base


class pcrbm_cluster(Base):
    def __init__(self, config, debug=False, precision="single"):
        super().__init__(config, debug=debug, precision=precision)

        self.mc_moves = config['mc_moves']  # Number of MC samples to take to update hidden and visible configurations

        # Batch sampling strategy, can be random or stratified
        try:
            self.sampling_strategy = config["sampling_strategy"]
        except KeyError:
            self.sampling_strategy = "random"
        assert self.sampling_strategy in ["random", "stratified", "weighted", "stratified_weighted", "polar"]

        # Only used is sampling strategy is weighted
        try:
            self.sampling_weights = config["sampling_weights"]
        except KeyError:
            self.sampling_weights = None

        # Stratify the datasets, training, validationa, and test
        try:
            self.stratify = config["stratify_datasets"]
        except KeyError:
            self.stratify = False

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
        self.lcorr = config['lcorr']
        ###########################################

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

        # duplicate convolution keys and convolution topology
        self.clusters = config["clusters"]
        cluster_convolution_topology = {}
        for i in range(self.clusters):
            cluster_convolution_topology[i] = {f"{k}_c{i}": v for k, v in self.convolution_topology.items()}

        self.convolution_topology = cluster_convolution_topology
        self.hidden_convolution_keys = [[x+f"_c{i}" for x in self.hidden_convolution_keys] for i in range(self.clusters)]

        self.max_inds = [[] for i in range(self.clusters)]


        # Should get rid of pearson crap soon
        self.use_pearson = config["use_pearson"]
        self.pearson_xvar = "none"
        if self.use_pearson:
            self.pearson_xvar = config["pearson_xvar"]  # label or fitness_value

            assert self.pearson_xvar in ["values", "labels"]

        # if self.pearson_xvar == "labels" or self.stratify or self.sampling_strategy == "stratified":
        try:
            self.group_fraction = config["group_fraction"]
        except KeyError:
            self.group_fraction = [1 / self.label_groups for i in self.label_groups]

        if self.sampling_strategy == "polar":
            assert self.batch_size % 2 == 0
            assert self.label_groups == 2
            assert self.group_fraction == [0.5, 0.5]

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

        for cid, cluster in enumerate(self.hidden_convolution_keys):
            cluster_pools = []
            cluster_unpools = []
            for key in cluster:
                # Set information about the convolutions that will be useful
                dims = conv2d_dim([self.batch_size, 1, self.v_num, self.q], self.convolution_topology[cid][key])
                self.convolution_topology[cid][key]["weight_dims"] = dims["weight_shape"]
                self.convolution_topology[cid][key]["convolution_dims"] = dims["conv_shape"]
                self.convolution_topology[cid][key]["output_padding"] = dims["output_padding"]

                # deal with pool and unpool initialization
                pool_input_size = dims["conv_shape"][:-1]

                pool_kernel = pool_input_size[2]
                pool_stride = 1
                pool_padding = 0

                cluster_pools.append(nn.MaxPool1d(pool_kernel, stride=pool_stride, return_indices=True, padding=pool_padding))
                cluster_unpools.append(nn.MaxUnpool1d(pool_kernel, stride=pool_stride, padding=pool_padding))

                # Convolution Weights
                self.register_parameter(f"{key}_W", nn.Parameter(self.weight_initial_amplitude * torch.randn(self.convolution_topology[cid][key]["weight_dims"], device=self.device)))
                # hidden layer parameters
                self.register_parameter(f"{key}_theta+", nn.Parameter(torch.zeros(self.convolution_topology[cid][key]["number"], device=self.device)))
                self.register_parameter(f"{key}_theta-", nn.Parameter(torch.zeros(self.convolution_topology[cid][key]["number"], device=self.device)))
                self.register_parameter(f"{key}_gamma+", nn.Parameter(torch.ones(self.convolution_topology[cid][key]["number"], device=self.device)))
                self.register_parameter(f"{key}_gamma-", nn.Parameter(torch.ones(self.convolution_topology[cid][key]["number"], device=self.device)))
                # Used in PT Sampling / AIS
                self.register_parameter(f"{key}_0theta+", nn.Parameter(torch.zeros(self.convolution_topology[cid][key]["number"], device=self.device), requires_grad=False))
                self.register_parameter(f"{key}_0theta-", nn.Parameter(torch.zeros(self.convolution_topology[cid][key]["number"], device=self.device), requires_grad=False))
                self.register_parameter(f"{key}_0gamma+", nn.Parameter(torch.ones(self.convolution_topology[cid][key]["number"], device=self.device), requires_grad=False))
                self.register_parameter(f"{key}_0gamma-", nn.Parameter(torch.ones(self.convolution_topology[cid][key]["number"], device=self.device), requires_grad=False))

            self.pools.append(cluster_pools)
            self.unpools.append(cluster_unpools)

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

        self.meminfo = False
        self.initial_run = 50



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

    def free_energy_cluster(self, v, cluster_indx):
        return self.energy_v(v) - self.logpartition_h_cluster(self.compute_output_v_cluster(v, cluster_indx), cluster_indx)

    ## Used in our Loss Function
    def free_energy(self, v):
        # logpart = torch.stack([self.logpartition_h_cluster(v, i) for i in range(self.clusters)], dim=0).sum(0)
        # visible enrgy  - logpartition over all clusters
        return self.energy_v(v) - torch.stack([self.logpartition_h_cluster(self.compute_output_v_cluster(v, i), i) for i in range(self.clusters)], dim=0).sum(0)

    # ## Not used but may be useful
    # def free_energy_h(self, h):
    #     return self.energy_h(h) - self.logpartition_v(self.compute_output_h(h))

    ## Total Energy of a given visible and hidden configuration
    # def energy(self, v, h, remove_init=False, hidden_sub_index=-1):
    #     return self.energy_v(v, remove_init=remove_init) + self.energy_h(h, sub_index=hidden_sub_index, remove_init=remove_init) - self.bidirectional_weight_term(v, h, hidden_sub_index=hidden_sub_index)

    # def energy_PT(self, v, h, N_PT, remove_init=False):
    #     # if N_PT is None:
    #     #     N_PT = self.N_PT
    #     E = torch.zeros((N_PT, v.shape[1]), device=self.device)
    #     for i in range(N_PT):
    #         E[i] = self.energy_v(v[i], remove_init=remove_init) + self.energy_h(h, sub_index=i, remove_init=remove_init) - self.bidirectional_weight_term(v[i], h, hidden_sub_index=i)
    #     return E

    # def bidirectional_weight_term(self, v, h, hidden_sub_index=-1):
    #     conv = self.compute_output_v(v)
    #     E = torch.zeros((len(self.hidden_convolution_keys), conv[0].shape[0]), device=self.device)
    #     for iid, i in enumerate(self.hidden_convolution_keys):
    #         if hidden_sub_index != -1:
    #             h_uk = h[iid][hidden_sub_index]
    #         else:
    #             h_uk = h[iid]
    #         E[iid] = h_uk.mul(conv[iid]).sum(1)
    #
    #     if E.shape[0] > 1:
    #         return E.sum(0)
    #     else:
    #         return E.squeeze(0)

    ############################################################# Individual Layer Functions
    def transform_v(self, I):
        return F.one_hot(torch.argmax(I + getattr(self, "fields").unsqueeze(0), dim=-1), self.q)
        # return self.one_hot_tmp.scatter(2, torch.argmax(I + getattr(self, "fields").unsqueeze(0), dim=-1).unsqueeze(-1), 1.)

    # def transform_h(self, I):
    #     output = []
    #     for kid, key in enumerate(self.hidden_convolution_keys):
    #         a_plus = (getattr(self, f'{key}_gamma+')).unsqueeze(0).unsqueeze(2)
    #         a_minus = (getattr(self, f'{key}_gamma-')).unsqueeze(0).unsqueeze(2)
    #         theta_plus = (getattr(self, f'{key}_theta+')).unsqueeze(0).unsqueeze(2)
    #         theta_minus = (getattr(self, f'{key}_theta-')).unsqueeze(0).unsqueeze(2)
    #         tmp = ((I[kid] + theta_minus) * (I[kid] <= torch.minimum(-theta_minus, (theta_plus / torch.sqrt(a_plus) -
    #                                                                                 theta_minus / torch.sqrt(a_minus)) / (1 / torch.sqrt(a_plus) + 1 / torch.sqrt(a_minus))))) / \
    #               a_minus + ((I[kid] - theta_plus) * (I[kid] >= torch.maximum(theta_plus, (theta_plus / torch.sqrt(a_plus) -
    #                                                                                        theta_minus / torch.sqrt(a_minus)) / (1 / torch.sqrt(a_plus) + 1 / torch.sqrt(a_minus))))) / a_plus
    #         output.append(tmp)
    #     return output

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
    def energy_h_cluster(self, config, cluster_indx, remove_init=False, sub_index=-1):
        # config is list of h_uks
        if sub_index != -1:
            E = torch.zeros((len(self.hidden_convolution_keys[cluster_indx]), config[0].shape[1]), device=self.device)
        else:
            E = torch.zeros((len(self.hidden_convolution_keys[cluster_indx]), config[0].shape[0]), device=self.device)

        for iid, i in enumerate(self.hidden_convolution_keys[cluster_indx]):
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
    def logpartition_h_cluster(self, inputs, cluster_indx, beta=1):
        # Input is list of matrices I_uk for specified cluster

        marginal = torch.zeros((len(self.hidden_convolution_keys[cluster_indx]), inputs[0].shape[0]), device=self.device)
        for iid, i in enumerate(self.hidden_convolution_keys[cluster_indx]):
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
    # def mean_h(self, psi, hidden_key=None, beta=1):
    #     if hidden_key is None:
    #         means = []
    #         for kid, key in enumerate(self.hidden_convolution_keys):
    #             if beta == 1:
    #                 a_plus = (getattr(self, f'{key}_gamma+')).unsqueeze(0)
    #                 a_minus = (getattr(self, f'{key}_gamma-')).unsqueeze(0)
    #                 theta_plus = (getattr(self, f'{key}_theta+')).unsqueeze(0)
    #                 theta_minus = (getattr(self, f'{key}_theta-')).unsqueeze(0)
    #             else:
    #                 theta_plus = (beta * getattr(self, f'{key}_theta+') + (1 - beta) * getattr(self, f'{key}_0theta+')).unsqueeze(0)
    #                 theta_minus = (beta * getattr(self, f'{key}_theta-') + (1 - beta) * getattr(self, f'{key}_0theta-')).unsqueeze(0)
    #                 a_plus = (beta * getattr(self, f'{key}_gamma+') + (1 - beta) * getattr(self, f'{key}_0gamma+')).unsqueeze(0)
    #                 a_minus = (beta * getattr(self, f'{key}_gamma-') + (1 - beta) * getattr(self, f'{key}_0gamma-')).unsqueeze(0)
    #                 psi[kid] *= beta
    #
    #             # if psi[kid].dim() == 3:
    #             #     a_plus = a_plus.unsqueeze(2)
    #             #     a_minus = a_minus.unsqueeze(2)
    #             #     theta_plus = theta_plus.unsqueeze(2)
    #             #     theta_minus = theta_minus.unsqueeze(2)
    #
    #             psi_plus = (-psi[kid] + theta_plus) / torch.sqrt(a_plus)
    #             psi_minus = (psi[kid] + theta_minus) / torch.sqrt(a_minus)
    #
    #             etg_plus = self.erf_times_gauss(psi_plus)
    #             etg_minus = self.erf_times_gauss(psi_minus)
    #
    #             p_plus = 1 / (1 + (etg_minus / torch.sqrt(a_minus)) / (etg_plus / torch.sqrt(a_plus)))
    #             nans = torch.isnan(p_plus)
    #             p_plus[nans] = 1.0 * (torch.abs(psi_plus[nans]) > torch.abs(psi_minus[nans]))
    #             p_minus = 1 - p_plus
    #
    #             mean_pos = (-psi_plus + 1 / etg_plus) / torch.sqrt(a_plus)
    #             mean_neg = (psi_minus - 1 / etg_minus) / torch.sqrt(a_minus)
    #             means.append(mean_pos * p_plus + mean_neg * p_minus)
    #         return means
    #     else:
    #         if beta == 1:
    #             a_plus = (getattr(self, f'{hidden_key}_gamma+')).unsqueeze(0)
    #             a_minus = (getattr(self, f'{hidden_key}_gamma-')).unsqueeze(0)
    #             theta_plus = (getattr(self, f'{hidden_key}_theta+')).unsqueeze(0)
    #             theta_minus = (getattr(self, f'{hidden_key}_theta-')).unsqueeze(0)
    #         else:
    #             theta_plus = (beta * getattr(self, f'{hidden_key}_theta+') + (1 - beta) * getattr(self, f'{hidden_key}_0theta+')).unsqueeze(0)
    #             theta_minus = (beta * getattr(self, f'{hidden_key}_theta-') + (1 - beta) * getattr(self, f'{hidden_key}_0theta-')).unsqueeze(0)
    #             a_plus = (beta * getattr(self, f'{hidden_key}_gamma+') + (1 - beta) * getattr(self, f'{hidden_key}_0gamma+')).unsqueeze(0)
    #             a_minus = (beta * getattr(self, f'{hidden_key}_gamma-') + (1 - beta) * getattr(self, f'{hidden_key}_0gamma-')).unsqueeze(0)
    #             psi *= beta
    #
    #         # if psi.dim() == 3:
    #         #     a_plus = a_plus.unsqueeze(2)
    #         #     a_minus = a_minus.unsqueeze(2)
    #         #     theta_plus = theta_plus.unsqueeze(2)
    #         #     theta_minus = theta_minus.unsqueeze(2)
    #
    #         psi_plus = (psi + theta_plus) / torch.sqrt(a_plus)  #  min pool
    #         psi_minus = (psi + theta_minus) / torch.sqrt(a_minus)  # max pool
    #
    #         etg_plus = self.erf_times_gauss(psi_plus)
    #         etg_minus = self.erf_times_gauss(psi_minus)
    #
    #         p_plus = 1 / (1 + (etg_minus / torch.sqrt(a_minus)) / (etg_plus / torch.sqrt(a_plus)))
    #         nans = torch.isnan(p_plus)
    #         p_plus[nans] = 1.0 * (torch.abs(psi_plus[nans]) > torch.abs(psi_minus[nans]))
    #         p_minus = 1 - p_plus
    #
    #         mean_pos = (-psi_plus + 1 / etg_plus) / torch.sqrt(a_plus)
    #         mean_neg = (psi_minus - 1 / etg_minus) / torch.sqrt(a_minus)
    #         return mean_pos * p_plus + mean_neg * p_minus


    def compute_output_v_cluster(self, X, cluster_indx):
        outputs = []
        self.max_inds[cluster_indx] = []
        for kid, key in enumerate(self.hidden_convolution_keys[cluster_indx]):
            weights = getattr(self, f"{key}_W")
            conv = F.conv2d(X.unsqueeze(1).type(torch.get_default_dtype()), weights,
                            stride=self.convolution_topology[cluster_indx][key]["stride"],
                            padding=self.convolution_topology[cluster_indx][key]["padding"],
                            dilation=self.convolution_topology[cluster_indx][key]["dilation"]).squeeze(3)

            max_pool, max_inds = self.pools[cluster_indx][kid](conv.abs())

            flat_conv = conv.flatten(start_dim=2)
            max_conv_values = flat_conv.gather(2, index=max_inds.flatten(start_dim=2)).view_as(max_inds)

            # max_conv_values = torch.gather(conv, 2, max_inds)
            max_pool *= max_conv_values / max_conv_values.abs()

            self.max_inds[cluster_indx].append(max_inds)

            out = max_pool.flatten(start_dim=2)

            out.squeeze_(2)

            if self.dr > 0.:
                # dropout_mask = F.dropout(torch.ones((out.shape[1]), device=self.device))
                # out = out * dropout_mask
                out = F.dropout(out, p=self.dr, training=self.training)

            outputs.append(out)
            if True in torch.isnan(out):
                print("hi")

        return outputs


    ## Compute Input for Hidden Layer from Visible Potts, Uses one hot vector
    def compute_output_v(self, X):  # X is the one hot vector
        outputs = []
        for i in range(self.clusters):
            outputs.append(self.compute_output_v_cluster(X, i))

        # flatten list to tensor
        return outputs

    ## Compute Input for Visible Layer from Hidden dReLU
    def compute_output_h_cluster(self, h, cluster_indx):  # from h_uk (B, hidden_num)
        outputs = []
        for kid, key in enumerate(self.hidden_convolution_keys[cluster_indx]):
            size = self.convolution_topology[cluster_indx][key]["number"]

            reconst = self.unpools[cluster_indx][kid](h[kid].view_as(self.max_inds[cluster_indx][kid]), self.max_inds[cluster_indx][kid])

            if reconst.ndim == 3:
                reconst.unsqueeze_(3)

            # convx = self.convolution_topology[i]["convolution_dims"][2]
            outputs.append(F.conv_transpose2d(reconst, getattr(self, f"{key}_W"),
                                              stride=self.convolution_topology[cluster_indx][key]["stride"],
                                              padding=self.convolution_topology[cluster_indx][key]["padding"],
                                              dilation=self.convolution_topology[cluster_indx][key]["dilation"],
                                              output_padding=self.convolution_topology[cluster_indx][key]["output_padding"]).squeeze(1))

        if len(outputs) > 1:
            # Returns mean output from all hidden layers, zeros are ignored
            # mean_denominator = torch.sum(torch.stack(nonzero_masks), 0) + 1e-6
            out = torch.sum(torch.stack(outputs), 0)
            if True in torch.isnan(out):
                print("hi")
            return out  # / mean_denominator
        else:
            if True in torch.isnan(outputs[0]):
                print("hi")
            return outputs[0]


    ## Compute Input for Visible Layer from Hidden dReLU
    def compute_output_h(self, h):  # from h_uk (B, hidden_num)
        outputs = []
        for i in range(self.clusters):
            self.compute_output_h_cluster(h[i], i)

        if len(outputs) > 1:
            # Returns mean output from all hidden layers, zeros are ignored
            # mean_denominator = torch.sum(torch.stack(nonzero_masks), 0) + 1e-6
            out = torch.sum(torch.stack(outputs), 0)
            if True in torch.isnan(out):
                print("hi")
            return out  # / mean_denominator
        else:
            if True in torch.isnan(outputs[0]):
                print("hi")
            return outputs[0]

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
    def sample_from_inputs_h_cluster(self, psi, cluster_indx, nancheck=False, beta=1):  # psi is a list of hidden [input]
        h_uks = []
        for iid, i in enumerate(self.hidden_convolution_keys[cluster_indx]):
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

            p_plus = 1 / (1 + (etg_minus / torch.sqrt(a_minus)) / (etg_plus / torch.sqrt(
                a_plus)))  # p+ 1 / (1 +( (Z-/sqrt(a-))/(Z+/sqrt(a+))))    =   (Z+/(Z++Z-)
            nans = torch.isnan(p_plus)

            if True in nans:
                p_plus[nans] = torch.tensor(1., device=self.device) * (torch.abs(psi_plus[nans]) > torch.abs(psi_minus[nans]))
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

    ## Gibbs Sampling of dReLU hidden layer
    def sample_from_inputs_h(self, psi, nancheck=False, beta=1):  # psi is a list of hidden [input]
        h_cluster_uks = []
        for i in range(self.clusters):
            h_cluster_uks.append(self.sample_from_inputs_h_cluster(psi[i], i, nancheck=nancheck, beta=beta))
        return h_cluster_uks

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

    def markov_step_cluster(self, v, cluster_indx, beta=1):
        # Gibbs Sampler
        h = self.sample_from_inputs_h_cluster(self.compute_output_v_cluster(v, cluster_indx), cluster_indx, beta=beta)
        return self.sample_from_inputs_v(self.compute_output_h_cluster(h, cluster_indx), beta=beta), h

    def markov_step_with_hidden_input(self, v, beta=1):
        # Gibbs Sampler
        h = self.sample_from_inputs_h(self.compute_output_v(v), beta=beta)
        vn = self.sample_from_inputs_v(self.compute_output_h(h), beta=beta)
        Ih = self.compute_output_v(vn)
        return vn, h, Ih

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
        # for key in self.hidden_convolution_keys:
        #     for param in ["gamma+", "gamma-", "theta+", "theta-", "W"]:
        #         par = getattr(self, f"{key}_{param}")
        #         grad = par.grad
        #         if True in torch.isnan(par) or True in torch.isnan(grad):
        #             print("hi")
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 10000, norm_type=2.0, error_if_nonfinite=True)

    # Clamps hidden potential values to acceptable range
    def on_before_zero_grad(self, optimizer):
        # for key in self.hidden_convolution_keys:
        #     for param in ["gamma+", "gamma-", "theta+", "theta-", "W"]:1
        #         if True in torch.isnan(par) or True in torch.isnan(grad):
        #             print("hi")
        with torch.no_grad():
            for cluster in self.hidden_convolution_keys:
                for key in cluster:
                    for param in ["gamma+", "gamma-"]:
                        getattr(self, f"{key}_{param}").data.clamp_(min=0.05)
                    for param in ["theta+", "theta-"]:
                        getattr(self, f"{key}_{param}").data.clamp_(min=0.0)
                    getattr(self, f"{key}_W").data.clamp_(-1.0, 1.0)

    ## Loads Training Data
    def train_dataloader(self, init_fields=True):
        # Get Correct Weights
        training_weights = None
        if "seq_count" in self.training_data.columns:
            training_weights = self.training_data["seq_count"].tolist()

        # additional_data = None
        # if "additional_data" in self.training_data.columns:
        #     additional_data = self.training_data["additional_data"].tolist()

        training_stds = None
        if "sample_std" in self.training_data.columns:
            training_stds = self.training_data["sample_std"].tolist()

        labels = False
        if self.pearson_xvar == "labels":
            labels = True

        train_reader = Categorical(self.training_data, self.q, weights=training_weights, max_length=self.v_num,
                                   molecule=self.molecule, device=self.device, one_hot=True, labels=labels, additional_data=training_stds)

        # initialize fields from data
        if init_fields:
            with torch.no_grad():
                initial_fields = train_reader.field_init()
                self.fields += initial_fields
                self.fields0 += initial_fields

        shuffle = True
        # if self.sample_type == "pcd":
        #     shuffle = False

        if self.sampling_strategy == "stratified":
            return torch.utils.data.DataLoader(
                train_reader,
                batch_sampler=StratifiedBatchSampler(self.training_data["label"].to_numpy(), batch_size=self.batch_size, shuffle=shuffle, seed=self.seed),
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
        elif self.sampling_strategy == "stratified_weighted" or self.sampling_strategy == "polar":
            return torch.utils.data.DataLoader(
                train_reader,
                batch_sampler=WeightedSubsetRandomSampler(self.sampling_weights, self.training_data["label"].to_numpy(), self.group_fraction, self.batch_size, self.sample_multiplier),
                num_workers=self.worker_num,  # Set to 0 if debug = True
                pin_memory=self.pin_mem
            )
        else:
            self.sampling_strategy = "random"
            self.cluster_filters = torch.full((self.clusters, len(train_reader),), True, device=self.device)
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

        validation_stds = None
        if "sample_std" in self.validation_data.columns:
            validation_stds = self.validation_data["sample_std"].tolist()

        labels = False
        if self.pearson_xvar == "labels":
            labels = True

        val_reader = Categorical(self.validation_data, self.q, weights=validation_weights, max_length=self.v_num,
                                 molecule=self.molecule, device=self.device, one_hot=True, labels=labels, additional_data=validation_stds)

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
            inds, seqs, one_hot, seq_weights, labels, additional_data = batch
        if self.pearson_xvar == "labels":
            inds, seqs, one_hot, seq_weights, labels = batch
        if self.additional_data:
            inds, seqs, one_hot, seq_weights, additional_data = batch
        else:
            inds, seqs, one_hot, seq_weights = batch

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


    ## On Epoch End Collects Scalar Statistics and Distributions of Parameters for the Tensorboard Logger
    def training_epoch_end(self, outputs):
        super().training_epoch_end(outputs)

        if self.meminfo:
            # print("GPU Reserved:", torch.cuda.memory_reserved(0))
            print(f"GPU Allocated Mem Epcoh {self.current_epoch}:", torch.cuda.memory_allocated(0))

    ## Not yet rewritten for crbm
    def training_step_CD_energy(self, batch, batch_idx):
        seqs, one_hot, seq_weights = batch

        # Regularization Terms
        reg1, reg2, reg3, bs_loss, gap_loss, reg_dict = self.regularization_terms()

        if self.sampling_strategy == "polar":
            half_batch = self.batch_size // 2
            V_neg_oh = one_hot[: half_batch]  # first half is sequences we don't like
            V_neg_weights = seq_weights[: half_batch]
            V_neg_weights = 1. / V_neg_weights
            V_neg_weights = F.softmax(V_neg_weights, dim=0)

            # shuffle around the negative tensor
            shuffle_tensor = torch.randperm(half_batch)
            V_neg_oh = V_neg_oh[shuffle_tensor]
            V_neg_oh = V_neg_oh[shuffle_tensor]

            V_pos_oh = one_hot[half_batch:]  # second half is sequences we do like
            V_pos_weights = seq_weights[half_batch:]

            # gibbs sampling
            V_gs_neg_oh, h_gs_neg, V_pos_oh, h_pos = self(V_pos_oh)
            h_neg = self.sample_from_inputs_h(self.compute_output_v(V_neg_oh))

            E_gs = (self.energy(V_gs_neg_oh, h_gs_neg) * V_pos_weights / V_pos_weights.sum())
            E_n = (self.energy(V_neg_oh, h_neg) * V_neg_weights / V_neg_weights.sum())
            E_p = (self.energy(V_pos_oh, h_pos) * V_pos_weights / V_pos_weights.sum())


            # free_energy_diff = (2 * E_p - E_n * 1.1 - E_gs * 0.9).sum()
            energy_diff = (2*E_p - E_gs - E_n).sum()
            cd_loss = energy_diff

            energy_log = {
                "energy_pos": E_p.sum().detach(),
                "energy_neg": E_n.sum().detach(),
                "energy_gibbs": E_gs.sum().detach(),
                # "cd_gibbs": free_energy_diff_gibbs.detach(),
                # "cd_polar": energy_diff.detach(),
                "cd_loss": cd_loss.detach(),
            }

        else:
            V_gs_neg_oh, h_gs_neg, V_pos_oh, h_pos = self(one_hot)
            E_gs = (self.energy(V_gs_neg_oh, h_gs_neg) * seq_weights / seq_weights.sum())
            E_p = (self.energy(V_pos_oh, h_pos) * seq_weights / seq_weights.sum())
            cd_loss = (E_p - E_gs).sum()

            energy_log = {
                "energy_pos": E_p.sum().detach(),
                "energy_neg": E_gs.sum().detach(),
                "cd_loss": cd_loss.detach(),
            }


        # Calculate Loss
        loss = cd_loss + reg1 + reg2 + reg3 + bs_loss + gap_loss

        logs = {"loss": loss,
                "train_energy": E_p.sum().detach(),
                **energy_log,
                **reg_dict
                }

        self.log("ptl/train_energy", logs["train_energy"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return logs

    def regularization_terms_cluster(self, cluster_indx):
        reg2 = torch.zeros((1,), device=self.device)
        reg3 = torch.zeros((1,), device=self.device)

        bs_loss = torch.zeros((1,), device=self.device)  # encourages weights to use both positive and negative contributions
        gap_loss = torch.zeros((1,), device=self.device)  # discourages high values for gaps

        for iid, i in enumerate(self.hidden_convolution_keys[cluster_indx]):
            W_shape = self.convolution_topology[cluster_indx][i]["weight_dims"]  # (h_num,  input_channels, kernel0, kernel1)
            W = getattr(self, f"{i}_W")
            x = torch.sum(torch.abs(getattr(self, f"{i}_W")), (3, 2, 1)).square()
            reg2 += x.sum() * self.l1_2 / (2 * W_shape[1] * W_shape[2] * W_shape[3])
            reg3 += self.ld / ((getattr(self, f"{i}_W").abs() - getattr(self, f"{i}_W").squeeze(1).abs()).abs().sum((1, 2, 3)).mean() + 1)
            gap_loss += self.lgap * W[:, :, :, -1].abs().sum()

            denom = torch.sum(torch.abs(W), (3, 2, 1))
            # zeroW = torch.zeros_like(W, device=self.device)
            Wpos = torch.clamp(W, min=0.)
            Wneg = torch.clamp(W, max=0.)
            # Wpos = torch.maximum(W, zeroW)
            # Wneg = torch.minimum(W, zeroW)
            bs_loss += self.lbs * torch.abs(Wpos.sum((1, 2, 3)) / denom - torch.abs(Wneg.sum((1, 2, 3))) / denom).sum()

        # Passed to training logger
        reg_dict = {
            "weight_reg": reg2.detach(),
            "distance_reg": reg3.detach(),
            "gap_reg": gap_loss.detach(),
            "both_side_reg": bs_loss.detach()
        }

        return reg2, reg3, bs_loss, gap_loss, reg_dict

    # Not yet rewritten for CRBM
    def training_step_PT_free_energy(self, batch, batch_idx):
        inds, seqs, one_hot, seq_weights = batch

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
        self.log("ptl/train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return logs

    def training_step_CD_free_energy(self, batch, batch_idx):
        if self.pearson_xvar == "labels":
            inds, seqs, one_hot, seq_weights, labels = batch
        else:
            inds, seqs, one_hot, seq_weights = batch
        # if self.meminfo:
        #     print("GPU Allocated Training Step Start:", torch.cuda.memory_allocated(0))

        if self.sampling_strategy == "polar":
            half_batch = self.batch_size // 2
            V_neg_oh = one_hot[: half_batch] # first half is sequences we don't like
            V_neg_weights = seq_weights[: half_batch]
            V_neg_weights = 1. / V_neg_weights
            V_neg_weights = F.softmax(V_neg_weights, dim=0)

            # shuffle around the negative tensor
            shuffle_tensor = torch.randperm(half_batch)
            V_neg_oh = V_neg_oh[shuffle_tensor]
            V_neg_oh = V_neg_oh[shuffle_tensor]


            V_pos_oh = one_hot[half_batch:] # second half is sequences we do like
            V_pos_weights = seq_weights[half_batch:]

            #gibbs sampling
            V_gs_neg_oh, h_gs_neg, V_pos_oh, h_pos = self(V_pos_oh)
            # h_neg = self.sample_from_inputs_h(self.compute_output_v(V_neg_oh))

            reconstruction_error = 1 - (V_pos_oh.argmax(-1) == V_gs_neg_oh.argmax(-1)).double().mean(-1)
            V_pos_weights *= reconstruction_error

            F_gs = self.free_energy(V_gs_neg_oh) * V_pos_weights / V_pos_weights.sum()

            F_v = self.free_energy(V_pos_oh) * V_pos_weights / V_pos_weights.sum()  # free energy of training data
            F_vp = self.free_energy(V_neg_oh) * V_neg_weights / V_neg_weights.sum() # free energy of gibbs sampled visible states



            # # Average activities of hidden units
            # hidden_exp_pos = torch.cat(self.mean_h(h_pos), dim=1)
            # hidden_exp_neg = torch.cat(self.mean_h(h_neg), dim=1)
            # hidden_exp_gs_neg = torch.cat(self.mean_h(h_gs_neg), dim=1)
            #
            # if True in torch.isnan(hidden_exp_pos) or True in torch.isnan(hidden_exp_neg) or True in torch.isnan(hidden_exp_gs_neg):
            #     print("hi")
            #
            # # Make activities opposite of one another, we'll see how this works
            # activity_distance_neg = (hidden_exp_pos + hidden_exp_neg).sum()
            # activity_distance_gs = (hidden_exp_pos + hidden_exp_gs_neg).sum()
            #
            # # activity_loss = (activity_distance_neg + activity_distance_gs).abs() * 0.05
            #
            # activity_loss = activity_distance_gs.abs() * 0.05



            #
            # F_gs = self.free_energy(V_gs_neg_oh) * V_pos_weights
            #
            # F_v = self.free_energy(V_pos_oh) * V_pos_weights # free energy of training data
            # F_vp = self.free_energy(V_neg_oh) * V_pos_weights  # free energy of gibbs sampled visible states



            # F_gs = (self.free_energy(V_gs_neg_oh) * V_pos_weights / V_pos_weights.sum()).sum()
            #
            # F_v = (self.free_energy(V_pos_oh) * V_pos_weights / V_pos_weights.sum()).sum()  # free energy of training data
            # F_vp = (self.free_energy(V_neg_oh) * V_neg_weights / V_neg_weights.sum()).sum()  # free energy of gibbs sampled visible states

            epoch_fraction = self.current_epoch/self.epochs

            free_energy_diff = (2*F_v - F_vp*1.1 - F_gs*0.9).sum()
            # free_energy_diff = F_v - F_gs


            # cd_loss = (0.2+epoch_fraction)*free_energy_diff + (1.4-epoch_fraction)*free_energy_diff_gibbs
            cd_loss = free_energy_diff

            free_energy_log = {
                "free_energy_pos": F_v.sum().detach(),
                "free_energy_neg": F_vp.sum().detach(),
                "free_energy_gibbs": F_gs.sum().detach(),
                # "cd_gibbs": free_energy_diff_gibbs.detach(),
                "cd_polar": free_energy_diff.detach(),
                "cd_loss": cd_loss.detach(),
            }

        else:
            V_neg_oh, h_neg, V_pos_oh, h_pos = self(one_hot)
            free_energy = self.free_energy(V_pos_oh)
            F_v = self.free_energy(V_pos_oh)
            F_vp = self.free_energy(V_neg_oh)
            # F_v = (self.free_energy(V_pos_oh) * seq_weights / seq_weights.sum()).sum()  # free energy of training data
            # F_vp = (self.free_energy(V_neg_oh) * seq_weights / seq_weights.sum()).sum()  # free energy of gibbs sampled visible states
            # free_energy_diff = F_v - F_vp
            # free_energy_adj = ((self.free_energy(V_pos_oh) * seq_weights - self.free_energy(V_pos_oh)) / seq_weights.sum()).sum()

            targets = (seq_weights/seq_weights.max()).type(torch.get_default_dtype())
            free_energy_term = F_v/F_v.min()

            # minimum = (targets - free_energy_term).min()
            adaptive_weights = 5*(targets-free_energy_term).exp()  # * (self.current_epoch/self.epochs * 2 + 0.5)
            # adaptive_weights = torch.maximum(adaptive_weights, torch.zeros_like(adaptive_weights, device=self.device))
            free_energy_diff = F_v*adaptive_weights - F_vp*adaptive_weights.abs()
            cd_loss = free_energy_diff.sum()

            # free_energy_kd = F.kl_div((free_energy/free_energy.sum()).log(), (seq_weights/seq_weights.sum()).type(torch.get_default_dtype()), reduction="batchmean")

            # cd_loss = free_energy_diff + free_energy_kd*10000

            # cd_loss = free_energy_kd *100000

            free_energy_log = {
                "free_energy_pos": F_v.sum().detach(),
                "free_energy_neg": F_vp.sum().detach(),
                "free_energy_diff": free_energy_diff.sum().detach(),
                # "free_energy_kd": free_energy_kd.detach(),
                "cd_loss": cd_loss.detach(),
            }

            # self.log("ptl/train_kd", free_energy_kd.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)


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
                "train_free_energy": free_energy_diff.sum().detach(),
                **free_energy_log,
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
            inds, seqs, one_hot, seq_weights, labels, additional_data = batch
        if self.pearson_xvar == "labels":
            inds, seqs, one_hot, seq_weights, labels = batch
        if self.additional_data:
            inds, seqs, one_hot, seq_weights, additional_data = batch
        else:
            inds, seqs, one_hot, seq_weights = batch

        if self.current_epoch == 0 and batch_idx == 0:
            self.chain = torch.zeros((self.clusters, self.training_data.index.__len__(), *one_hot.shape[1:]), device=self.device)

        if self.current_epoch == 0:
            for cluster_indx in range(self.clusters):
                self.chain[cluster_indx][inds] = one_hot.type(torch.get_default_dtype())


        # Regularization Terms
        reg1 = self.lf / (2 * self.v_num * self.q) * getattr(self, "fields").square().sum((0, 1))

        cluster_logs = {"loss": reg1}

        # cluster_indx = self.get_cluster_indx(self.current_epoch)
        #
        # # Apply filter to learn poorly rated sequences
        # if cluster_indx > 0:
        #     # apply filters to inds, data, and weights
        #     overall_filter = torch.clamp(torch.sum(self.cluster_filters[:cluster_indx, inds], dim=0) - (cluster_indx - 1), min=0).bool()
        #     # overall_filter = torch.clamp(torch.stack(filters, dim=0).sum(0) , min=0).bool()
        #     filtered_inds = inds[overall_filter]
        #     cdata = cdata[overall_filter]
        #     weights = weights[overall_filter]
        #
        # # Typical free energy calculations
        # F_v = self.free_energy_cluster(cdata, cluster_indx)
        # Vc_neg, hc_neg = self.forward_PCD(filtered_inds, cluster_indx)
        # F_vp = self.free_energy_cluster(Vc_neg, cluster_indx)
        #
        # # Make new filter for next cluster if more than 5% of the seqs have not been picked up by a cluster
        # # Otherwise focus on that 5% of sequences
        # if overall_filter.sum() > .05 * inds.shape[0]:
        #     cluster_ids, means = kmeans(F_v, 2, tol=1e-3, iter_limit=40)
        #     unrealized_seqs = cluster_ids == torch.argmax(means)
        #
        #     self.cluster_filters[cluster_indx][filtered_inds] = unrealized_seqs
        #
        # cluster_cd_loss = (weights * (F_v - F_vp)).mean()
        #
        # reg2, reg3, bs_loss, gap_loss, reg_dict = self.regularization_terms_cluster(cluster_indx)
        #
        # cluster_loss = cluster_cd_loss + reg2 + reg3 + bs_loss + gap_loss
        #
        # cluster_logs["loss"] = cluster_logs["loss"].add(cluster_loss)
        # cluster_logs["cluster"] = cluster_indx
        # cluster_logs[f"Cluster free_energy"] = self.free_energy_cluster(cdata, cluster_indx).detach().mean()
        # # cluster_logs[f"Cluster {cluster_indx} Loss"] = cluster_loss.detach()
        # cluster_logs[f"Cluster CD Loss"] = cluster_cd_loss.detach()
        # cluster_logs.update(reg_dict)

        if self.current_epoch < self.initial_run:
            F_v = self.free_energy_cluster(one_hot, 0)
            Vc_neg, hc_neg = self.forward_PCD(inds, 0)
            F_vp = self.free_energy_cluster(Vc_neg, 0)

            reg2, reg3, bs_loss, gap_loss, reg_dict = self.regularization_terms_cluster(0)

            cluster_cd_loss = (seq_weights * (F_v - F_vp)).mean()

            cluster_loss = cluster_cd_loss + reg2 + reg3 + bs_loss + gap_loss

            cluster_logs["loss"] = cluster_logs["loss"].add(cluster_loss)
            cluster_logs[f"Cluster {0} free_energy"] = self.free_energy_cluster(one_hot, 0).detach().mean()
            cluster_logs[f"Cluster {0} Loss"] = cluster_loss.detach()
            cluster_logs[f"Cluster {0} CD Loss"] = cluster_cd_loss.detach()
            cluster_logs.update({f"Cluster_{0} " + k: v.detach() for k, v in reg_dict.items()})

            if self.current_epoch == self.initial_run-1:
                if batch_idx == 0:
                    self.all_Fv = [F_v]
                    self.all_Inds = [inds]
                else:
                    self.all_Fv.append(F_v)
                    self.all_Inds.append(inds)

        elif self.current_epoch == self.initial_run:
            if batch_idx == 0:
                #Initialize Clusters
                full_Fv = torch.cat(self.all_Fv)
                bins = 30
                min, max = full_Fv.min(), full_Fv.max()
                counts = torch.histc(full_Fv, bins, min=min.data, max=max.data)
                boundaries = torch.linspace(min.data, max.data, bins + 1)

                original_counts = counts.clone()
                peaks = []
                while True:
                     if torch.max(counts) < 100:
                         break
                     l_indx, r_indx = self.find_peak(original_counts, torch.argmax(counts))
                     peaks.append((l_indx, r_indx))
                     counts[l_indx:r_indx] = 0.

                self.all_Inds = torch.cat(self.all_Inds)
                self.cluster_assignments = torch.full((len(self.all_Inds),), 0, device=self.device)
                for peak_indx, bounds in enumerate(peaks):
                    peak_members = torch.logical_and(full_Fv > boundaries[bounds[0]], full_Fv <= boundaries[bounds[1]])
                    self.cluster_assignments[self.all_Inds[peak_members.bool()]] = peak_indx + 1

                self.clusters = len(peaks) + 1

        else:

            clust_assign = self.cluster_assignments[inds]
            for cluster_indx in range(1, self.clusters):
                filter = clust_assign == cluster_indx
                cdata = one_hot[filter]
                weights = seq_weights[filter]
                filtered_inds = inds[filter]

                # Typical free energy calculations
                F_v = self.free_energy_cluster(cdata, cluster_indx)
                Vc_neg, hc_neg = self.forward_PCD(filtered_inds, cluster_indx)
                F_vp = self.free_energy_cluster(Vc_neg, cluster_indx)

                cluster_cd_loss = (weights * (F_v - F_vp)).mean()

                reg2, reg3, bs_loss, gap_loss, reg_dict = self.regularization_terms_cluster(cluster_indx)

                cluster_loss = cluster_cd_loss + reg2 + reg3 + bs_loss + gap_loss

                cluster_logs["loss"] = cluster_logs["loss"].add(cluster_loss)
                cluster_logs[f"Cluster {cluster_indx} free_energy"] = self.free_energy_cluster(cdata, cluster_indx).detach().mean()
                cluster_logs[f"Cluster {cluster_indx} Loss"] = cluster_loss.detach()
                cluster_logs[f"Cluster {cluster_indx} CD Loss"] = cluster_cd_loss.detach()
                cluster_logs.update({f"Cluster_{cluster_indx} " + k: v.detach() for k, v in reg_dict.items()})




        # for cluster_indx in range(self.clusters):
        #     filtered_inds = inds
        #     cdata = one_hot
        #     weights = seq_weights
        #     # Filter that is applied to full batch to get the current clusters
        #     full_filter = torch.full((one_hot.shape[0],), False, device=self.device) # modified later
        #     overall_filter = ~full_filter  # modified below
        #
        #     # Apply filter to learn poorly rated sequences
        #     if cluster_indx > 0:
        #         # apply filters to inds, data, and weights
        #         overall_filter = torch.clamp(torch.sum(self.cluster_filters[:cluster_indx, inds], dim=0) - (cluster_indx - 1), min=0).bool()
        #         # overall_filter = torch.clamp(torch.stack(filters, dim=0).sum(0) , min=0).bool()
        #         filtered_inds = inds[overall_filter]
        #         cdata = cdata[overall_filter]
        #         weights = weights[overall_filter]
        #
        #     # Typical free energy calculations
        #     F_v = self.free_energy_cluster(cdata, cluster_indx)
        #     Vc_neg, hc_neg = self.forward_PCD(filtered_inds, cluster_indx)
        #     F_vp = self.free_energy_cluster(Vc_neg, cluster_indx)
        #
        #     # Make new filter for next cluster if more than 5% of the seqs have not been picked up by a cluster
        #     # Otherwise focus on that 5% of sequences
        #     if overall_filter.sum() > .05 * inds.shape[0]:
        #         # cluster_ids, means = kmeans(F_v, 2, tol=1e-3, iter_limit=40)
        #         # unrealized_seqs = cluster_ids == torch.argmax(means)
        #
        #         bins = 30
        #         min, max = F_v.min(), F_v.max()
        #         counts = torch.histc(F_v, bins, min=min.data, max=max.data)
        #         boundaries = torch.linspace(min.data, max.data, bins + 1)
        #
        #         min_indx = self.find_closest_minima(counts)
        #
        #         unrealized_seqs = F_v > boundaries[min_indx+1]
        #
        #         self.cluster_filters[cluster_indx][filtered_inds] = unrealized_seqs
        #
        #     cluster_cd_loss = (weights * (F_v - F_vp)).mean()
        #
        #     reg2, reg3, bs_loss, gap_loss, reg_dict = self.regularization_terms_cluster(cluster_indx)
        #
        #     cluster_loss = cluster_cd_loss + reg2 + reg3 + bs_loss + gap_loss
        #
        #     cluster_logs["loss"] = cluster_logs["loss"].add(cluster_loss)
        #     cluster_logs[f"Cluster {cluster_indx} free_energy"] = self.free_energy_cluster(cdata, cluster_indx).detach().mean()
        #     cluster_logs[f"Cluster {cluster_indx} Loss"] = cluster_loss.detach()
        #     cluster_logs[f"Cluster {cluster_indx} CD Loss"] = cluster_cd_loss.detach()
        #     cluster_logs.update({f"Cluster_{cluster_indx} " + k: v.detach() for k, v in reg_dict.items()})

        # Free Energy of Sequences from all clusters
        cluster_logs["free_energy"] = self.free_energy(one_hot).detach().mean()

        self.log("ptl/train_free_energy", cluster_logs["free_energy"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", cluster_logs["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return cluster_logs

    def find_closest_minima(self, counts):
        i = 0
        prev_count = 0
        while True:
            if counts[i] < prev_count:
                break
            else:
                prev_count = counts[i]
                i += 1
        return i

    # get indices of largest peak
    def find_peak(self, counts, peak_indx):
        # finds local minima surrounding given peak index and returns their indices
        i = peak_indx
        # right side
        right_prev_count = counts[peak_indx]
        j = i + 1  # right side index
        while True:
            if j >= len(counts):
                j -= 1
                break
            if counts[j] >= right_prev_count:
                j -= 1
                break
            else:
                right_prev_count = counts[j]
                j += 1
        # left side
        k = i - 1  # left side index
        left_prev_count = counts[peak_indx]
        while True:
            if k < 0:
                k = 0
                break
            if counts[k] >= left_prev_count:
                k += 1
                break
            else:
                left_prev_count = counts[k]
                k -= 1
        return k, j

    def forward_PCD(self, inds, cluster_indx):
        # Gibbs sampling with Persistent Contrastive Divergence
        # pytorch lightning handles the device
        fantasy_v = self.chain[cluster_indx][inds]  # Last sample that was saved to self.chain variable, initialized in training step
        # h_pos = self.sample_from_inputs_h(self.compute_output_v(fantasy_v))
        # with torch.no_grad() # only use last sample for gradient calculation, may be helpful but honestly not the slowest thing rn
        for _ in range(self.mc_moves - 1):
            fantasy_v, fantasy_h = self.markov_step_cluster(fantasy_v, cluster_indx)

        V_neg, fantasy_h = self.markov_step_cluster(fantasy_v, cluster_indx)

        h_neg = self.sample_from_inputs_h(self.compute_output_v(V_neg))

        return V_neg, h_neg

    def forward(self, V_pos_ohe):
        if self.sample_type == "gibbs":
            # Gibbs sampling
            # pytorch lightning handles the device
            fantasy_v, first_h = self.markov_step(V_pos_ohe)
            # with torch.no_grad():  # only use last sample for gradient calculation, Enabled to minimize memory usage, hopefully won't have much effect on performance
            for _ in range(self.mc_moves - 1):
                fantasy_v, fantasy_h = self.markov_step(fantasy_v)

            # V_neg, fantasy_h = self.markov_step(fantasy_v)
            # V_neg, h_neg, V_pos, h_pos
            return fantasy_v, fantasy_h, V_pos_ohe, first_h

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
                inds, seqs, one_hot, seq_weights = batch
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
            inds, seqs, one_hot, seq_weights = batch
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





