import time
import pandas as pd
import math
import json
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.profiler import SimpleProfiler, PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from pytorch_lightning import LightningModule, Trainer

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW, Adagrad, Adadelta  # Supported Optimizers
from multiprocessing import cpu_count # Just to set the worker number
from torch.autograd import Variable

from rbm_torch.models.pool_crbm_base import pool_CRBM

from rbm_torch.utils.utils import Categorical, fasta_read, conv2d_dim, pool1d_dim, BatchNorm1D  #Sequence_logo, gen_data_lowT, gen_data_zeroT, all_weights, Sequence_logo_all,


# using approach from https://github.com/fregu856/ebms_regression/blob/master/1dregression/1/nce%2B_train.py

#### Should rewrite to use this newer approach
# https://github.com/fregu856/ebms_proposals/blob/main/1dregression_1/ebmdn4_train_K1.py


class NoiseNet(nn.Module):
    def __init__(self, hidden_dim=10, K=1):
        super().__init__()

        self.K = K

        self.fc1_mean = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, self.K)

        self.fc1_sigma = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_sigma = nn.Linear(hidden_dim, self.K)

        self.fc1_weight = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_weight = nn.Linear(hidden_dim, self.K)

    def forward(self, x_feature):
        # (x_feature has shape: (batch_size, hidden_dim))

        means = F.relu(self.fc1_mean(x_feature))  # (shape: (batch_size, hidden_dim))
        means = self.fc2_mean(means)  # (shape: batch_size, K))

        log_sigma2s = F.relu(self.fc1_sigma(x_feature))  # (shape: (batch_size, hidden_dim))
        log_sigma2s = self.fc2_sigma(log_sigma2s)  # (shape: batch_size, K))

        weight_logits = F.relu(self.fc1_weight(x_feature))  # (shape: (batch_size, hidden_dim))
        weight_logits = self.fc2_weight(weight_logits)  # (shape: batch_size, K))
        weights = torch.softmax(weight_logits, dim=1) # (shape: batch_size, K))

        return means, log_sigma2s, weights


class PredictorNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=10):
        super().__init__()

        self.fc1_y = nn.Linear(input_dim, hidden_dim)
        # self.fc2_y = nn.Linear(hidden_dim, hidden_dim)

        self.fc1_xy = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc2_xy = nn.Linear(hidden_dim, 1)
        # self.fc3_xy = nn.Linear(hidden_dim, 1)

    def forward(self, x_feature, y):
        # (x_feature has shape: (batch_size, hidden_dim))
        # (y has shape (batch_size, num_samples)) (num_sampes==1 when running on (x_i, y_i))

        if y.dim() == 1:
            y = y.view(-1,1)

        batch_size, num_samples = y.shape

        # Replicate
        x_feature = x_feature.view(batch_size, 1, -1).expand(-1, num_samples, -1) # (shape: (batch_size, num_samples, hidden_dim))

        # resize to batch dimension
        x_feature = x_feature.reshape(batch_size*num_samples, -1) # (shape: (batch_size*num_samples, hidden_dim))
        y = y.reshape(batch_size*num_samples, -1) # (shape: (batch_size*num_samples, 1))

        y_feature = F.tanh(self.fc1_y(y)) # (shape: (batch_size*num_samples, hidden_dim))
        # y_feature = F.relu(self.fc2_y(y_feature)) # (shape: (batch_size*num_samples, hidden_dim))

        xy_feature = torch.cat([x_feature, y_feature], 1) # (shape: (batch_size*num_samples, 2*hidden_dim))

        xy_feature = F.tanh(self.fc1_xy(xy_feature)) # (shape: (batch_size*num_samples, hidden_dim))
        # xy_feature = F.relu(self.fc2_xy(xy_feature)) + xy_feature # (shape: (batch_size*num_samples, hidden_dim))
        score = F.tanh(self.fc2_xy(xy_feature)) # (shape: (batch_size*num_samples, hidden_dim))
        # score = self.fc3_xy(xy_feature) # (shape: (batch_size*num_samples, 1))

        score = score.view(batch_size, num_samples) # (shape: (batch_size, num_samples))

        return score


class FeatureNet(nn.Module):
    def __init__(self, config, debug=False, precision="single"):
        super().__init__()
        self.pcrbm = pool_CRBM(config, debug=True, precision=precision)

        self.hidden_dim = 0
        for key in self.pcrbm.hidden_convolution_keys:
            self.hidden_dim += self.pcrbm.convolution_topology[key]["number"]

        self.batch_norm = nn.BatchNorm1d(self.hidden_dim, affine=False)

    def logpartition_h_ind(self, inputs, beta=1):
        # Input is list of matrices I_uk
        ys = []
        for iid, i in enumerate(self.pcrbm.hidden_convolution_keys):
            if beta == 1:
                a_plus = (getattr(self.pcrbm, f'{i}_gamma+')).unsqueeze(0)
                a_minus = (getattr(self.pcrbm, f'{i}_gamma-')).unsqueeze(0)
                theta_plus = (getattr(self.pcrbm, f'{i}_theta+')).unsqueeze(0)
                theta_minus = (getattr(self.pcrbm, f'{i}_theta-')).unsqueeze(0)
            else:
                theta_plus = (beta * getattr(self.pcrbm, f'{i}_theta+') + (1 - beta) * getattr(self.pcrbm, f'{i}_0theta+')).unsqueeze(0)
                theta_minus = (beta * getattr(self.pcrbm, f'{i}_theta-') + (1 - beta) * getattr(self.pcrbm, f'{i}_0theta-')).unsqueeze(0)
                a_plus = (beta * getattr(self.pcrbm, f'{i}_gamma+') + (1 - beta) * getattr(self.pcrbm, f'{i}_0gamma+')).unsqueeze(0)
                a_minus = (beta * getattr(self.pcrbm, f'{i}_gamma-') + (1 - beta) * getattr(self.pcrbm, f'{i}_0gamma-')).unsqueeze(0)

            # in_neg = inputs[iid][:, :, 1]
            # in_pos = inputs[iid][:, :, 0]
            y = torch.logaddexp(self.pcrbm.log_erf_times_gauss((-inputs[iid] + theta_plus) / torch.sqrt(a_plus)) -
                                0.5 * torch.log(a_plus), self.pcrbm.log_erf_times_gauss((inputs[iid] + theta_minus) / torch.sqrt(a_minus)) - 0.5 * torch.log(a_minus)) + 0.5 * np.log(2 * np.pi) * inputs[iid].shape[1]
            ys.append(y)  # 10 added so hidden layer has stronger effect on free energy, also in energy_h
            # marginal[iid] /= self.convolution_topology[i]["convolution_dims"][2]
        return torch.cat(ys, dim=1)

    def forward(self, x):
        # (x has shape (batch_size, v_num, q))

        # x_feature = self.pcrbm.free_energy(x)
        # hidden = self.pcrbm.sample_from_inputs_h(x_out)
        free_energy_ind = self.pcrbm.energy_v(x).unsqueeze(1) - self.logpartition_h_ind(self.pcrbm.compute_output_v(x))

        # x_feature = torch.cat(crbm_energy, 1)
        # x_feature = self.batchnorm(hidden/self.hidden_dim)
        # x_feature = hidden/self.hidden_dim

        return self.batch_norm(free_energy_ind)


class RegressionNet(LightningModule):
    def __init__(self, config, debug=False, precision="single"):
        super().__init__()

        # self.model_id = model_id
        # self.project_dir = project_dir
        # self.create_model_dirs()

        input_dim = 1
        self.sample_num = 256

        self.feature_net = FeatureNet(config, debug=debug, precision=precision)
        hidden_dim = self.feature_net.hidden_dim

        self.noise_net = NoiseNet(hidden_dim)

        self.predictor_net = PredictorNet(input_dim, hidden_dim)

    def forward(self, x, y):
        # (x has shape (batch_size, 1))
        # (y has shape (batch_size, num_samples)) (num_sampes==1 when running on (x_i, y_i))

        x_feature = self.feature_net(x)  # (shape: (batch_size, hidden_dim))
        return self.noise_net(x_feature)

    def train_dataloader(self):
        return self.feature_net.pcrbm.train_dataloader()

    def val_dataloader(self):
        return self.feature_net.pcrbm.val_dataloader()

    def setup(self,  stage=None):
        return self.feature_net.pcrbm.setup()

    def training_step(self, batch, batch_idx):
        inds, seq, one_hot, seq_weights = batch
        xs = one_hot # (shape: (batch_size, 1))
        ys = seq_weights.unsqueeze(1).to(torch.get_default_dtype())  # (shape: (batch_size, 1))

        x_features = self.feature_net(xs)  # (shape: (batch_size, hidden_dim))
        # if self.current_epoch < 50:
        #     x_features = x_features.detach()

        means, log_sigma2s, weights = self.noise_net(x_features.detach())  # (all have shape: (batch_size, K))
        sigmas = torch.exp(log_sigma2s / 2.0)  # (shape: (batch_size, K))

        q_distr = torch.distributions.normal.Normal(loc=means, scale=sigmas)
        q_ys_K = torch.exp(q_distr.log_prob(torch.transpose(ys, 1, 0).unsqueeze(2)))  # (shape: (1, batch_size, K))
        q_ys = torch.sum(weights.unsqueeze(0) * q_ys_K, dim=2)  # (shape: (1, batch_size))
        q_ys = q_ys.squeeze(0)  # (shape: (batch_size))

        y_samples_K = q_distr.sample(sample_shape=torch.Size([self.sample_num]))  # (shape: (num_samples, batch_size, K))
        inds = torch.multinomial(weights, num_samples=self.sample_num, replacement=True).unsqueeze(2)  # (shape: (batch_size, num_samples, 1))
        inds = torch.transpose(inds, 1, 0)  # (shape: (num_samples, batch_size, 1))
        y_samples = y_samples_K.gather(2, inds).squeeze(2)  # (shape: (num_samples, batch_size))
        y_samples = y_samples.detach()
        q_y_samples_K = torch.exp(q_distr.log_prob(y_samples.unsqueeze(2)))  # (shape: (num_samples, batch_size, K))
        q_y_samples = torch.sum(weights.unsqueeze(0) * q_y_samples_K, dim=2)  # (shape: (num_samples, batch_size))
        y_samples = torch.transpose(y_samples, 1, 0)  # (shape: (batch_size, num_samples))
        q_y_samples = torch.transpose(q_y_samples, 1, 0)  # (shape: (batch_size, num_samples))

        scores_gt = self.predictor_net(x_features, ys)  # (shape: (batch_size, 1))
        scores_gt = scores_gt.squeeze(1)  # (shape: (batch_size))

        scores_samples = self.predictor_net(x_features, y_samples)  # (shape: (batch_size, num_samples))

        ########################################################################
        # compute loss:
        ########################################################################
        f_samples = scores_samples
        p_N_samples = q_y_samples.detach()
        f_0 = scores_gt
        p_N_0 = q_ys.detach()
        exp_vals_0 = f_0 - torch.log(p_N_0 + 0.0)
        exp_vals_samples = f_samples - torch.log(p_N_samples + 0.0)
        exp_vals = torch.cat([exp_vals_0.unsqueeze(1), exp_vals_samples], dim=1)
        loss_ebm_nce = -torch.mean(exp_vals_0 - torch.logsumexp(exp_vals, dim=1))

        log_Z = torch.logsumexp(scores_samples.detach() - torch.log(q_y_samples), dim=1) - math.log(self.sample_num)  # (shape: (batch_size))
        loss_mdn_kl = torch.mean(log_Z)

        loss_mdn_nll = torch.mean(-torch.log(q_ys))

        loss = loss_ebm_nce + loss_mdn_nll

        self.log("loss_nce", loss_ebm_nce.detach(), prog_bar=True, on_epoch=True)
        self.log("loss_kl", loss_mdn_kl.detach(), prog_bar=True, on_epoch=True)
        self.log("loss_nll", loss_mdn_nll.detach(), prog_bar=True, on_epoch=True)
        return loss

    def predict_y(self, x):
        y_samples = torch.linspace(0.0, 1.01, self.sample_num, device=self.device)

        x_features = self.feature_net(x)
        scores = self.predictor_net(x_features, y_samples.expand(x.shape[0], -1))  # (shape: (batch_size, num_samples))
        scores = torch.exp(scores)
        denom = torch.sum(scores, dim=1)

        prob = scores / denom.unsqueeze(1)

        pred = y_samples[prob.argmax(1)]
        return pred

    def validation_step(self, batch, batch_idx):
        inds, seqs, one_hot, seq_weights = batch

        x = one_hot  # (shape: (batch_size, 1))

        # y_samples = np.linspace(0.0, 1.0, self.sample_num)  # (shape: (num_samples, ))
        # y_samples = y_samples.astype(np.float32)
        # y_samples = torch.from_numpy(y_samples).cuda()
        pred = self.predict_y(x)

        mse = torch.mean((pred - seq_weights)**2)

        self.log("val_mse", mse.detach(), prog_bar=True, on_epoch=True)
        return

    def configure_optimizers(self):
        optim = self.feature_net.pcrbm.optimizer(self.parameters(), lr=self.feature_net.pcrbm.lr, weight_decay=self.feature_net.pcrbm.wd)
        # Exponential Weight Decay after set amount of epochs (set by decay_after)
        decay_gamma = (self.feature_net.pcrbm.lrf / self.feature_net.pcrbm.lr) ** (1 / (self.feature_net.pcrbm.epochs * (1 - self.feature_net.pcrbm.decay_after)))
        decay_milestone = math.floor(self.feature_net.pcrbm.decay_after * self.feature_net.pcrbm.epochs)
        my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=[decay_milestone], gamma=decay_gamma)
        # my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=decay_gamma)
        optim_dict = {"lr_scheduler": my_lr_scheduler,
                      "optimizer": optim}
        return optim_dict

    def on_before_zero_grad(self, optimizer):
        with torch.no_grad():
            for key in self.feature_net.pcrbm.hidden_convolution_keys:
                for param in ["gamma+", "gamma-"]:
                    getattr(self.feature_net.pcrbm, f"{key}_{param}").data.clamp_(0.05, 1.0)
                for param in ["theta+", "theta-"]:
                    getattr(self.feature_net.pcrbm, f"{key}_{param}").data.clamp_(0.0, 1.0)
                getattr(self.feature_net.pcrbm, f"{key}_W").data.clamp_(-1.0, 1.0)

if __name__ == "__main__":
    from rbm_torch.utils.utils import load_run_file

    run_file_dir = "/home/jonah/PycharmProjects/phage_display_ML/example_run_files/"
    run_file = run_file_dir + "example_pool_crbm_regression.json"

    run_data, config = load_run_file(run_file)
    config["fasta_file"] = "/home/jonah/PycharmProjects/phage_display_ML/regression_model_comparison/cov/cov_z_full_norm.fasta"

    debug = False
    model = RegressionNet(config, debug=debug, precision="single")

    logger = TensorBoardLogger("/home/jonah/PycharmProjects/phage_display_ML/regression_test/", name="crbm_regression")

    if run_data["gpus"] == 0:
        plt = Trainer(max_epochs=config['epochs'], logger=logger, accelerator="cpu")
    else:
        plt = Trainer(max_epochs=config['epochs'], logger=logger, accelerator="cuda", devices=run_data["gpus"])

    plt.fit(model)

    train_data = model.train_dataloader()
    actual, pred = [], []
    model.eval()
    for batch_idx, batch in enumerate(train_data):
        seqs, one_hot, seq_weights = batch

        predy = model.predict_y(one_hot)
        actual += seq_weights
        pred += predy

    import matplotlib.pyplot as plt

    plt.scatter(actual, pred, alpha=0.1)
    plt.show()


# def gauss_density_centered(x, std):
#     return torch.exp(-0.5*(x / std)**2) / (math.sqrt(2*math.pi)*std)
#
# def gmm_density_centered(x, std):
#     """
#     Assumes dim=-1 is the component dimension and dim=-2 is feature dimension. Rest are sample dimension.
#     """
#     if x.dim() == std.dim() - 1:
#         x = x.unsqueeze(-1)
#     elif not (x.dim() == std.dim() and x.shape[-1] == 1):
#         raise ValueError('Last dimension must be the gmm stds.')
#     return gauss_density_centered(x, std).prod(-2).mean(-1)
#
# def sample_gmm_centered(std, num_samples=1):
#     num_components = std.shape[-1]
#     num_dims = std.numel() // num_components
#
#     std = std.view(1, num_dims, num_components)
#
#     # Sample component ids
#     k = torch.randint(num_components, (num_samples,), dtype=torch.int64)
#     std_samp = std[0,:,k].t()
#
#     # Sample
#     x_centered = std_samp * torch.randn(num_samples, num_dims)
#     prob_dens = gmm_density_centered(x_centered, std)
#
#     prob_dens_zero = gmm_density_centered(torch.zeros_like(x_centered), std)
#
#     return x_centered, prob_dens, prob_dens_zero
#
# def sample_gmm_centered2(beta, std, num_samples=1):
#     num_components = std.shape[-1]
#     num_dims = std.numel() // num_components
#
#     std = std.view(1, num_dims, num_components)
#
#     # Sample component ids
#     k = torch.randint(num_components, (num_samples,), dtype=torch.int64)
#     std_samp = std[0,:,k].t()
#
#     # Sample
#     x_centered = beta*std_samp * torch.randn(num_samples, num_dims)
#     prob_dens = gmm_density_centered(x_centered, std)
#
#     prob_dens_zero = gmm_density_centered(torch.zeros_like(x_centered), std)
#
#     return x_centered, prob_dens, prob_dens_zero
#
#
# class pool_regression_CRBM(pool_CRBM):
#     def __init__(self, config, debug=False, precision="double", meminfo=False):
#         super().__init__(config, debug=debug, precision=precision, meminfo=meminfo)
#         self.beta = config["beta"]
#         self.stds = torch.zeros((1, 2), device=self.device)
#         self.num_samples = config["num_samples"]
#         self.stds[0, 0] = 0.1
#         self.stds[0, 1] = 0.3
#
#         # self.linear = nn.Linear(1, 1, bias=False)
#
#
#         # self.alpha = config["alpha"]
#         #
#         # self.classes = config["classes"]
#         # self.register_parameter(f"y_bias", nn.Parameter(torch.zeros(self.classes, device=self.device)))
#         # self.register_parameter(f"0y_bias", nn.Parameter(torch.zeros(self.classes, device=self.device)))
#         for key in self.hidden_convolution_keys:
#             self.register_parameter(f"{key}_M", nn.Parameter(0.05 * torch.randn((self.convolution_topology[key]["number"]), device=self.device)))
#
#     ## Unsupervised Loss Function
#     def free_energy_v_y(self, v, y):
#         h_input_v = self.compute_output_v(v)
#         y_in = self.compute_output_y_for_h(y)
#         h_in = [torch.add(h_input_v[i], y_in[i]) for i in range(len(h_input_v))]
#         return self.energy_v(v) - self.logpartition_h(h_in)
#         # return (self.energy_v(v) - self.logpartition_h(h_in)).div(100)
#
#
#     ## Computes h M term
#     def compute_output_y_for_h(self, Y):
#         # from y our regression values?
#         # calculates M*Y
#         outputs = []
#         for key in self.hidden_convolution_keys:
#             # if all_classes:
#             #     outputs.append(torch.swapaxes(getattr(self, f"{key}_M"), 1, 0).expand(Y.shape[0], -1, -1))
#             # else:
#             # outputs.append(torch.index_select(torch.swapaxes(getattr(self, f"{key}_M"), 1, 0), 0, Y))
#             outputs.append(Y.unsqueeze(1) * getattr(self, f"{key}_M").unsqueeze(0))
#         return outputs
#
#
#     ######################################################### Pytorch Lightning Functions
#     ## Not yet rewritten for crbm
#
#     def training_step_CD_free_energy(self, batch, batch_idx):
#         seqs, one_hot, seq_weights = batch
#
#         ys = seq_weights
#         xs = one_hot
#
#         y_samples_0_zero, q_y_samples_0, _ = sample_gmm_centered2(self.beta, self.stds, num_samples=1)
#         y_samples_0 = ys + y_samples_0_zero.squeeze(1)  # (shape: (batch_size, 1))
#
#         # x_features = network.feature_net(xs)  # (shape: (batch_size, hidden_dim))    ### should this be p_v?
#         # x_features = self.free_energy(seqs)
#         # scores_samples_0 = network.predictor_net(x_features, y_samples_0)  # (shape: (batch_size, 1))
#         scores_samples_0 = self.free_energy_v_y(xs, y_samples_0)  # (shape: (batch_size, 1))
#         #scores_samples_0 = scores_samples_0.squeeze(1)  # (shape: (batch_size))
#
#         y_samples_zero, q_y_samples, _ = sample_gmm_centered(self.stds, num_samples=self.num_samples)
#         y_samples_zero = y_samples_zero.squeeze(1)  # (shape: (num_samples))
#
#         y_samples = ys.unsqueeze(1) + y_samples_zero.unsqueeze(0)  # (shape: (batch_size, num_samples))
#         q_y_samples = q_y_samples.unsqueeze(0) * torch.ones(y_samples.size())  # (shape: (batch_size, num_samples))
#
#         scores_samples = []
#         for i in range(self.num_samples):
#             scores_samples.append(self.free_energy_v_y(xs, y_samples[:, i]).unsqueeze(1))
#         # scores_samples = self.predict_y(y_samples)  # (shape: (batch_size, num_samples))
#         scores_samples = torch.cat(scores_samples, 1)
#
#         ########################################################################
#         # compute loss:
#         ########################################################################
#         loss = -torch.mean(scores_samples_0 - torch.log(q_y_samples_0) - torch.log(torch.exp(scores_samples_0 - torch.log(q_y_samples_0)) + torch.sum(torch.exp(scores_samples - torch.log(q_y_samples)), dim=1)))
#         # # Calculate Loss
#         # loss = hybrid_loss_function + reg1 + reg2 + reg3 + gap_loss + bs_loss  # + class_loss
#         #
#         # # Calculate Loss
#         # # loss = cd_loss + reg1 + reg2 + reg3
#         #
#         predy = self.predict_y(xs)
#
#         mse = torch.mean((predy - ys)**2)
#
#         logs = {"loss": loss,
#                 "train_mse": mse.detach(),
#                 # "train_free_energy": F_v.detach(),
#                 # "field_reg": reg1.detach(),
#                 # "weight_reg": reg2.detach(),
#                 # "distance_reg": reg3.detach()
#                 }
#         #
#         # acc = (predicted_labels == labels).double().mean()
#         # logs["acc"] = acc.detach()
#         # self.log("ptl/train_acc", logs["acc"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         #
#         #
#         self.log("ptl/train_mse", logs["train_mse"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         self.log("ptl/train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         #
#         return logs
#
#     def training_epoch_end(self, outputs):
#         mse = torch.stack([x["train_mse"] for x in outputs]).mean()
#         loss = torch.stack([x["train_loss"] for x in outputs]).mean()
#         # pseudo_likelihood = torch.stack([x['train_pseudo_likelihood'] for x in outputs]).mean()
#
#         all_scalars = {"Loss": loss,
#                        "Train MSE": mse,
#                        # "Field Reg": field_reg,
#                        # "Weight Reg": weight_reg,
#                        # "Distance Reg": distance_reg,
#                        # # "Train_pseudo_likelihood": pseudo_likelihood,
#                        # "Train Free Energy": free_energy,
#                        }
#         self.logger.experiment.add_scalars("All Scalars", all_scalars, self.current_epoch)
#
#
#     def validation_step(self, batch, batch_idx):
#
#         seqs, one_hot, seq_weights = batch
#
#         predy = self.predict_y(one_hot)
#
#         mse = torch.mean((predy - seq_weights) ** 2)
#
#         batch_out = {}
#
#         batch_out["val_mse"] = mse.detach()
#
#         return batch_out
#
#         ## Loads Training Data
#
#     def validation_epoch_end(self, outputs):
#         mse = torch.stack([x['val_mse'] for x in outputs]).mean()
#
#         scalars = {"Validation MSE": mse}
#
#         self.logger.experiment.add_scalars("Val Scalars", scalars, self.current_epoch)
#
#     def train_dataloader(self, init_fields=True):
#         # Get Correct Weights
#         if "seq_count" in self.training_data.columns:
#             training_weights = self.training_data["seq_count"].tolist()
#         else:
#             training_weights = None
#
#         train_reader = Categorical(self.training_data, self.q, weights=training_weights, max_length=self.v_num,
#                                    molecule=self.molecule, device=self.device, one_hot=True, labels=False)
#
#         # initialize fields from data
#         if init_fields:
#             with torch.no_grad():
#                 initial_fields = train_reader.field_init()
#                 self.fields += initial_fields
#                 self.fields0 += initial_fields
#
#             # Performance was almost identical whether shuffling or not
#         if self.sample_type == "pcd":
#             shuffle = False
#         else:
#             shuffle = True
#
#         return torch.utils.data.DataLoader(
#             train_reader,
#             batch_size=self.batch_size,
#             num_workers=self.worker_num,  # Set to 0 if debug = True
#             pin_memory=self.pin_mem,
#             shuffle=shuffle
#         )
#
#     def val_dataloader(self):
#         # Get Correct Validation weights
#         if "seq_count" in self.validation_data.columns:
#             validation_weights = self.validation_data["seq_count"].tolist()
#         else:
#             validation_weights = None
#
#
#         val_reader = Categorical(self.validation_data, self.q, weights=validation_weights, max_length=self.v_num,
#                                  molecule=self.molecule, device=self.device, one_hot=True, labels=False)
#
#         return torch.utils.data.DataLoader(
#             val_reader,
#             batch_size=self.batch_size,
#             num_workers=self.worker_num,  # Set to 0 to view tensors while debugging
#             pin_memory=self.pin_mem,
#             shuffle=False
#         )
#
#     # def forward(self, V_pos_ohe, y_pos):
#     #     if self.sample_type == "gibbs":
#     #         # Gibbs sampling
#     #         # pytorch lightning handles the device
#     #         with torch.no_grad():  # only use last sample for gradient calculation, Enabled to minimize memory usage, hopefully won't have much effect on performance
#     #             first_v, first_h, first_y = self.markov_step(V_pos_ohe, y_pos)
#     #             V_neg = first_v.clone()
#     #             fantasy_y = first_y.clone()
#     #             if self.mc_moves - 2 > 0:
#     #                 for _ in range(self.mc_moves - 2):
#     #                     V_neg, fantasy_h, fantasy_y = self.markov_step(V_neg, fantasy_y)
#     #
#     #         V_neg, fantasy_h, fantasy_y = self.markov_step(V_neg, fantasy_y)
#     #
#     #         # V_neg, h_neg, V_pos, h_pos
#     #         return V_neg, fantasy_h, fantasy_y, V_pos_ohe, first_h, y_pos
#     #
#     #     elif self.sample_type == "pt":
#     #         # Initialize_PT is called before the forward function is called. Therefore, N_PT will be filled
#     #
#     #         # Parallel Tempering
#     #         n_chains = V_pos_ohe.shape[0]
#     #
#     #         with torch.no_grad():
#     #             fantasy_v = self.random_init_config_v(custom_size=(self.N_PT, n_chains))
#     #             fantasy_h = self.random_init_config_h(custom_size=(self.N_PT, n_chains))
#     #             fantasy_E = self.energy_PT(fantasy_v, fantasy_h, self.N_PT)
#     #
#     #             for _ in range(self.mc_moves - 1):
#     #                 fantasy_v, fantasy_h, fantasy_E = self.markov_PT_and_exchange(fantasy_v, fantasy_h, fantasy_E, self.N_PT)
#     #                 self.update_betas(self.N_PT)
#     #
#     #         fantasy_v, fantasy_h, fantasy_E = self.markov_PT_and_exchange(fantasy_v, fantasy_h, fantasy_E, self.N_PT)
#     #         self.update_betas(self.N_PT)
#     #
#     #         # V_neg, h_neg, V_pos, h_pos
#     #         return fantasy_v[0], fantasy_h[0], V_pos_ohe, self.sample_from_inputs_h(self.compute_output_v(V_pos_ohe))
#
#     # X must be a pandas dataframe with the sequences in string format under the column 'sequence'
#     # Returns the likelihood for each sequence in an array
#     def predict(self, X):
#         # Read in data
#         reader = Categorical(X, self.q, weights=None, max_length=self.v_num, molecule=self.molecule, device=self.device, one_hot=True)
#         data_loader = torch.utils.data.DataLoader(
#             reader,
#             batch_size=self.batch_size,
#             num_workers=self.worker_num,  # Set to 0 if debug = True
#             pin_memory=self.pin_mem,
#             shuffle=False
#         )
#         self.eval()
#         with torch.no_grad():
#             likelihood_preds, label_preds = [], []
#             for i, batch in enumerate(data_loader):
#                 seqs, one_hot, seq_weights = batch
#                 predicted_labels = self.label_prediction(one_hot).detach().tolist()
#                 label_preds += predicted_labels
#                 likelihood_preds += self.likelihood(one_hot).detach().tolist()
#
#         return X.sequence.tolist(), likelihood_preds, label_preds
#
#     def y_input(self):
#         y_input = []
#         for key in self.hidden_convolution_keys:
#             y_input.append(getattr(self, f"{key}_M"))
#         return y_input
#
#     def predict_y(self, v):
#         h_input_v = self.compute_output_v(v)
#         y_in = self.y_input()
#         h_in = [torch.add(h_input_v[i], y_in[i]) for i in range(len(h_input_v))]
#         return self.linear(self.logpartition_h(h_in))


