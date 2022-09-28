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


# using approach from https://github.com/fregu856/ebms_regression/blob/master/1dregression/1/nce%2B_train.py

#### Should rewrite to use this newer approach
# https://github.com/fregu856/ebms_proposals/blob/main/1dregression_1/ebmdn4_train_K1.py


def gauss_density_centered(x, std):
    return torch.exp(-0.5*(x / std)**2) / (math.sqrt(2*math.pi)*std)

def gmm_density_centered(x, std):
    """
    Assumes dim=-1 is the component dimension and dim=-2 is feature dimension. Rest are sample dimension.
    """
    if x.dim() == std.dim() - 1:
        x = x.unsqueeze(-1)
    elif not (x.dim() == std.dim() and x.shape[-1] == 1):
        raise ValueError('Last dimension must be the gmm stds.')
    return gauss_density_centered(x, std).prod(-2).mean(-1)

def sample_gmm_centered(std, num_samples=1):
    num_components = std.shape[-1]
    num_dims = std.numel() // num_components

    std = std.view(1, num_dims, num_components)

    # Sample component ids
    k = torch.randint(num_components, (num_samples,), dtype=torch.int64)
    std_samp = std[0,:,k].t()

    # Sample
    x_centered = std_samp * torch.randn(num_samples, num_dims)
    prob_dens = gmm_density_centered(x_centered, std)

    prob_dens_zero = gmm_density_centered(torch.zeros_like(x_centered), std)

    return x_centered, prob_dens, prob_dens_zero

def sample_gmm_centered2(beta, std, num_samples=1):
    num_components = std.shape[-1]
    num_dims = std.numel() // num_components

    std = std.view(1, num_dims, num_components)

    # Sample component ids
    k = torch.randint(num_components, (num_samples,), dtype=torch.int64)
    std_samp = std[0,:,k].t()

    # Sample
    x_centered = beta*std_samp * torch.randn(num_samples, num_dims)
    prob_dens = gmm_density_centered(x_centered, std)

    prob_dens_zero = gmm_density_centered(torch.zeros_like(x_centered), std)

    return x_centered, prob_dens, prob_dens_zero


class pool_regression_CRBM(pool_CRBM):
    def __init__(self, config, debug=False, precision="double", meminfo=False):
        super().__init__(config, debug=debug, precision=precision, meminfo=meminfo)
        self.beta = config["beta"]
        self.stds = torch.zeros((1, 2), device=self.device)
        self.num_samples = config["num_samples"]
        self.stds[0, 0] = 0.1
        self.stds[0, 1] = 0.3

        self.linear = nn.Linear(1, 1, bias=False)


        # self.alpha = config["alpha"]
        #
        # self.classes = config["classes"]
        # self.register_parameter(f"y_bias", nn.Parameter(torch.zeros(self.classes, device=self.device)))
        # self.register_parameter(f"0y_bias", nn.Parameter(torch.zeros(self.classes, device=self.device)))
        for key in self.hidden_convolution_keys:
            self.register_parameter(f"{key}_M", nn.Parameter(0.05 * torch.randn((self.convolution_topology[key]["number"]), device=self.device)))

    ## Unsupervised Loss Function
    def free_energy_v_y(self, v, y):
        h_input_v = self.compute_output_v(v)
        y_in = self.compute_output_y_for_h(y)
        h_in = [torch.add(h_input_v[i], y_in[i]) for i in range(len(h_input_v))]
        return self.energy_v(v) - self.logpartition_h(h_in)
        # return (self.energy_v(v) - self.logpartition_h(h_in)).div(100)


    ## Computes h M term
    def compute_output_y_for_h(self, Y):
        # from y our regression values?
        # calculates M*Y
        outputs = []
        for key in self.hidden_convolution_keys:
            # if all_classes:
            #     outputs.append(torch.swapaxes(getattr(self, f"{key}_M"), 1, 0).expand(Y.shape[0], -1, -1))
            # else:
            # outputs.append(torch.index_select(torch.swapaxes(getattr(self, f"{key}_M"), 1, 0), 0, Y))
            outputs.append(Y.unsqueeze(1) * getattr(self, f"{key}_M").unsqueeze(0))
        return outputs


    ######################################################### Pytorch Lightning Functions
    ## Not yet rewritten for crbm

    def training_step_CD_free_energy(self, batch, batch_idx):
        seqs, one_hot, seq_weights = batch

        ys = seq_weights
        xs = one_hot

        y_samples_0_zero, q_y_samples_0, _ = sample_gmm_centered2(self.beta, self.stds, num_samples=1)
        y_samples_0 = ys + y_samples_0_zero.squeeze(1)  # (shape: (batch_size, 1))

        # x_features = network.feature_net(xs)  # (shape: (batch_size, hidden_dim))    ### should this be p_v?
        # x_features = self.free_energy(seqs)
        # scores_samples_0 = network.predictor_net(x_features, y_samples_0)  # (shape: (batch_size, 1))
        scores_samples_0 = self.free_energy_v_y(xs, y_samples_0)  # (shape: (batch_size, 1))
        #scores_samples_0 = scores_samples_0.squeeze(1)  # (shape: (batch_size))

        y_samples_zero, q_y_samples, _ = sample_gmm_centered(self.stds, num_samples=self.num_samples)
        y_samples_zero = y_samples_zero.squeeze(1)  # (shape: (num_samples))

        y_samples = ys.unsqueeze(1) + y_samples_zero.unsqueeze(0)  # (shape: (batch_size, num_samples))
        q_y_samples = q_y_samples.unsqueeze(0) * torch.ones(y_samples.size())  # (shape: (batch_size, num_samples))

        scores_samples = []
        for i in range(self.num_samples):
            scores_samples.append(self.free_energy_v_y(xs, y_samples[:, i]).unsqueeze(1))
        # scores_samples = self.predict_y(y_samples)  # (shape: (batch_size, num_samples))
        scores_samples = torch.cat(scores_samples, 1)

        ########################################################################
        # compute loss:
        ########################################################################
        loss = -torch.mean(scores_samples_0 - torch.log(q_y_samples_0) - torch.log(torch.exp(scores_samples_0 - torch.log(q_y_samples_0)) + torch.sum(torch.exp(scores_samples - torch.log(q_y_samples)), dim=1)))
        # # Calculate Loss
        # loss = hybrid_loss_function + reg1 + reg2 + reg3 + gap_loss + bs_loss  # + class_loss
        #
        # # Calculate Loss
        # # loss = cd_loss + reg1 + reg2 + reg3
        #
        predy = self.predict_y(xs)

        mse = torch.mean((predy - ys)**2)

        logs = {"loss": loss,
                "train_mse": mse.detach(),
                # "train_free_energy": F_v.detach(),
                # "field_reg": reg1.detach(),
                # "weight_reg": reg2.detach(),
                # "distance_reg": reg3.detach()
                }
        #
        # acc = (predicted_labels == labels).double().mean()
        # logs["acc"] = acc.detach()
        # self.log("ptl/train_acc", logs["acc"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #
        #
        self.log("ptl/train_mse", logs["train_mse"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #
        return logs

    def training_epoch_end(self, outputs):
        mse = torch.stack([x["train_mse"] for x in outputs]).mean()
        loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        # pseudo_likelihood = torch.stack([x['train_pseudo_likelihood'] for x in outputs]).mean()

        all_scalars = {"Loss": loss,
                       "Train MSE": mse,
                       # "Field Reg": field_reg,
                       # "Weight Reg": weight_reg,
                       # "Distance Reg": distance_reg,
                       # # "Train_pseudo_likelihood": pseudo_likelihood,
                       # "Train Free Energy": free_energy,
                       }
        self.logger.experiment.add_scalars("All Scalars", all_scalars, self.current_epoch)


    def validation_step(self, batch, batch_idx):

        seqs, one_hot, seq_weights = batch

        predy = self.predict_y(one_hot)

        mse = torch.mean((predy - seq_weights) ** 2)

        batch_out = {}

        batch_out["val_mse"] = mse.detach()

        return batch_out

        ## Loads Training Data

    def validation_epoch_end(self, outputs):
        mse = torch.stack([x['val_mse'] for x in outputs]).mean()

        scalars = {"Validation MSE": mse}

        self.logger.experiment.add_scalars("Val Scalars", scalars, self.current_epoch)

    def train_dataloader(self, init_fields=True):
        # Get Correct Weights
        if "seq_count" in self.training_data.columns:
            training_weights = self.training_data["seq_count"].tolist()
        else:
            training_weights = None

        train_reader = Categorical(self.training_data, self.q, weights=training_weights, max_length=self.v_num,
                                   molecule=self.molecule, device=self.device, one_hot=True, labels=False)

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


        val_reader = Categorical(self.validation_data, self.q, weights=validation_weights, max_length=self.v_num,
                                 molecule=self.molecule, device=self.device, one_hot=True, labels=False)

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

    def y_input(self):
        y_input = []
        for key in self.hidden_convolution_keys:
            y_input.append(getattr(self, f"{key}_M"))
        return y_input

    def predict_y(self, v):
        h_input_v = self.compute_output_v(v)
        y_in = self.y_input()
        h_in = [torch.add(h_input_v[i], y_in[i]) for i in range(len(h_input_v))]
        return self.linear(self.logpartition_h(h_in))
