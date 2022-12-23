from rbm_torch.models.base import Base

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch_geometric as tg
from torch_geometric.nn import GATv2Conv

import json
import numpy as np


class SCGCN(Base):
    def __init__(self, config, debug="False", precision="single"):
        super().__init__(config, debug=debug, precision=precision)

        self.dr = config["dr"]

        self.max_node_size = config["max_node_size"]


        self.latent_dim = config["latent_dim"]
        self.node_depth = config["node_depth"]
        self.adj_depth = config["adj_depth"]
        self.predictor_depth = config["predictor_depth"]

        # Input will be of size (B, 1, v_num, q)
        # cnet, self.feature_num = self.make_node_net(depth=self.node_depth)
        # self.node_conv_net = nn.Sequential(*cnet)
        self.node_conv_net = nn.Sequential(
                nn.Conv2d(1, self.max_node_size, kernel_size=(7, self.q)),
                nn.BatchNorm2d(self.max_node_size),
                nn.LeakyReLU())

        self.feature_num = self.v_num - 7 + 1


        self.adjacency_net = nn.Sequential(*self.make_adj_net(depth=self.adj_depth))

        self.predictor_net = nn.Sequential(*self.make_predictor_net(self.latent_dim, depth=self.predictor_depth))
        # self.regression_loss = nn.MSELoss()

        # GCN Components
        # self.c1 = GATv2Conv(-1, self.latent_dim)
        self.c1 = GATv2Conv(-1, 3 * self.latent_dim)
        self.c2 = GATv2Conv(3 * self.latent_dim, 2 * self.latent_dim)
        self.c3 = GATv2Conv(2 * self.latent_dim, self.latent_dim)
        self.c4 = GATv2Conv(self.latent_dim, self.latent_dim)
        self.c5 = GATv2Conv(self.latent_dim, self.latent_dim)
        self.c6 = GATv2Conv(self.latent_dim, self.latent_dim)

        # self.bn_c1 = torch.nn.BatchNorm1d(self.latent_dim)
        self.bn_c1 = torch.nn.BatchNorm1d(3 * self.latent_dim)
        self.bn_c2 = torch.nn.BatchNorm1d(2 * self.latent_dim)
        self.bn_c3 = torch.nn.BatchNorm1d(self.latent_dim)
        self.bn_c4 = torch.nn.BatchNorm1d(self.latent_dim)
        self.bn_c5 = torch.nn.BatchNorm1d(self.latent_dim)
        self.bn_c6 = torch.nn.BatchNorm1d(self.latent_dim)

        # self.aggregator = tg.nn.PowerMeanAggregation(1.0, learn=False)
        self.aggregator = tg.nn.MeanAggregation()
        # self.aggregator = tg.nn.MedianAggregation()

        self.combined_embedding_net = nn.Sequential(
            nn.Linear(int(self.max_node_size*(self.max_node_size-1)/2), self.feature_num*self.max_node_size),
            nn.BatchNorm1d(self.feature_num*self.max_node_size),
            nn.ReLU()
        )


        self.save_hyperparameters()

    def make_node_net(self, depth=3):
        conv_start_channels = 1
        conv_end_channels = self.max_node_size

        kernel_sizes, output_sizes, channels = [], [], []

        channels = [x for x in range(conv_start_channels, conv_end_channels - 1, -int((conv_start_channels - conv_end_channels) // (depth-1)))]
        channels.append(conv_end_channels)

        input_size = self.v_num
        for i in range(depth):
            kernel_sizes.append(int(input_size // 4))
            output_sizes.append(input_size - kernel_sizes[-1] + 1)
            input_size = output_sizes[-1]

        network = []
        for i in range(depth):
            if i == 0:
                network.append(nn.Sequential(
                    nn.Conv2d(1, channels[i], kernel_size=(kernel_sizes[i], self.q)),
                    nn.BatchNorm2d(channels[i]),
                    nn.LeakyReLU()))
                    # nn.Tanh()))
            else:
                network.append(nn.Sequential(
                    nn.Conv2d(channels[i - 1], channels[i], kernel_size=(kernel_sizes[i], 1)),
                    nn.BatchNorm2d(channels[i]),
                    nn.LeakyReLU()))

        feature_num = output_sizes[-1]

        return network, feature_num

    def make_adj_net(self, depth=3):
        fcn_start_size = self.max_node_size*self.feature_num
        fcn_end_size = int((self.max_node_size*(self.max_node_size - 1)) / 2)

        fcn_size = [fcn_start_size]
        # if self.network_layers > 1:
        for i in range(depth - 1):
            in_size = fcn_size[-1]
            fcn_size.append(in_size // 2)

        fcn_size.append(fcn_end_size)

        network = []
        for i in range(depth - 1):
            network.append(nn.Dropout(self.dr))
            network.append(nn.Linear(fcn_size[i], fcn_size[i + 1], dtype=torch.get_default_dtype()))
            network.append(nn.BatchNorm1d(fcn_size[i + 1]))
            network.append(nn.LeakyReLU())

        network.append(nn.Dropout(self.dr))
        network.append(nn.Linear(fcn_size[-2], fcn_size[-1], dtype=torch.get_default_dtype()))
        network.append(nn.Sigmoid())  # nice values b/t 0 and 1
        return network

    def make_predictor_net(self, in_features, dropout=0.1, depth=1):
        net = []
        for i in range(depth - 1):  # build many linear layers halving the number of units each time
            out_features = in_features // 2
            net.append(nn.Linear(in_features, out_features))
            in_features = out_features

            net.append(nn.LeakyReLU())  # activation function
            net.append(nn.Dropout(p=self.dr))  # dropout

        # Last layer, since its regression, 1 output feature
        net.append(nn.Linear(in_features, 1))
        return net

    def make_adjacency_matrix(self, adj_encoding):
        """Converts probabilities of edges and converts to a full edge list"""
        # Returns Tuple of edge tensors


        # location of all edges as sampled above
        global_edges = torch.argwhere(adj_encoding)

        # decode flattened upper triangular matrix into linear i and j indices
        # from: https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
        global_edges_batch = global_edges[:, 0].clone()
        global_edges[:, 0] = global_edges[:, 1]
        global_edges[:, 0] = self.max_node_size - 2 - (torch.sqrt(-8 * global_edges[:, 0] + 4 * self.max_node_size * (self.max_node_size - 1) - 7) / 2.0 - 0.5).long()
        global_edges[:, 1] = global_edges[:, 1] + global_edges[:, 0] + 1 - self.max_node_size * (self.max_node_size - 1) / 2 + (self.max_node_size - global_edges[:, 0]) * ((self.max_node_size - global_edges[:, 0]) - 1) / 2

        global_edges.add_(global_edges_batch.unsqueeze(1) * self.max_node_size)

        # for splitting into subtensors
        # vals, counts = torch.unique(global_edges[:, 0], return_counts=True)
        # edges = torch.tensor_split(global_edges.T, torch.cumsum(counts, 0), dim=1)

        return global_edges.T.long()

    def forward(self, X):
        nodes = self.node_conv_net(X.unsqueeze(1)).squeeze(3)
        flat_nodes = nodes.view(nodes.shape[0]*self.max_node_size, -1)

        adj_probs = self.adjacency_net(nodes.view(-1, self.max_node_size*self.feature_num))
        adj_encoding = torch.bernoulli(adj_probs)

        edges = self.make_adjacency_matrix(adj_encoding)

        edge_embedding = self.combined_embedding_net(adj_encoding).reshape(-1, self.feature_num)
        combined_embedding = edge_embedding*flat_nodes

        x = self.c1(combined_embedding, edges)
        x = F.leaky_relu(x)
        x = F.dropout(x)
        x = self.bn_c1(x)
        x = self.c2(x, edges)
        x = F.leaky_relu(x)
        x = F.dropout(x)
        x = self.bn_c2(x)
        x = self.c3(x, edges)
        x = F.leaky_relu(x)
        x = F.dropout(x)
        x = self.bn_c3(x)
        x = self.c4(x, edges)
        x = F.leaky_relu(x)
        x = F.dropout(x)
        x = self.bn_c4(x)
        x = self.c5(x, edges)
        x = F.leaky_relu(x)
        x = F.dropout(x)
        x = self.bn_c5(x)
        x = self.c6(x, edges)
        x = F.leaky_relu(x)
        x = F.dropout(x)
        x = self.bn_c6(x)

        x = self.aggregator(x, torch.arange(0,  nodes.shape[0], 1/(self.max_node_size), device=self.device).long())

        # Finally Our predictor network
        return self.predictor_net(x).squeeze(1)

    def regression_loss(self, preds, targets):
        return torch.pow(preds-targets, 2)

    def robust_loss(self, preds, targets, alpha, c):
        # from: https://openaccess.thecvf.com/content_CVPR_2019/papers/Barron_A_General_and_Adaptive_Robust_Loss_Function_CVPR_2019_paper.pdf
        # Author: Jonathan T Barton
        x = preds - targets
        ab_alpha = abs(alpha - 2)
        return ab_alpha/alpha * (torch.pow(torch.pow(x/c, 2)/ab_alpha + 1, alpha/2) - 1)

    def training_step(self, batch, batch_idx):
        if self.sample_stds is not None:
            inds, seqs, one_hot, enrichment, stds = batch

        else:
            inds, seqs, one_hot, enrichment = batch

        preds = self(one_hot.type(torch.get_default_dtype()))

        batch_out = {}

        adj_start_epoch = 15
        y_steps = 100
        if self.sample_stds is not None and self.current_epoch >= adj_start_epoch:
            if self.current_epoch == adj_start_epoch:
                ys = self.adjust_labels(preds, enrichment, enrichment, stds, steps=y_steps)
                if batch_idx == 0:
                    self.ystar = torch.zeros((self.training_data.index.__len__(),), device=self.device)
                self.ystar[inds] = ys
                # else:
                #     self.ystar = torch.cat((self.ystar, ys), dim=0)
                if True in torch.isnan(ys):
                    print("Label Adjustment Produced Nan")
                    exit(1)
                # loss_per_seq = self.robust_loss(preds, ys, -1, 0.5)
                loss_per_seq = self.regression_loss(preds, ys)
            elif self.current_epoch > adj_start_epoch:
                current_y = self.ystar[inds]
                ys = self.adjust_labels(preds, current_y, enrichment, stds, steps=y_steps)
                if True in torch.isnan(ys):
                    print("Label Adjustment Produced Nan")
                    exit(1)
                self.ystar[inds] = ys
                # loss_per_seq = self.robust_loss(preds, ys, -1, 0.5)
                loss_per_seq = self.regression_loss(preds, ys)

                batch_out["diff_y_l2"] = torch.pow(ys - enrichment, 2).mean()
        else:
            loss_per_seq = self.regression_loss(preds, enrichment)
            # loss_per_seq = self.robust_loss(preds, enrichment, -1, 0.5)

        # itlm
        if self.current_epoch > 0:
            threshold = torch.quantile(loss_per_seq, 1-self.itlm_alpha)
            loss = loss_per_seq[loss_per_seq < threshold].mean()
        else:
            loss = loss_per_seq.mean()

        batch_out["loss"] = loss

        self.log("train_loss", batch_out["loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)

        return batch_out

    def validation_step(self, batch, batch_idx):
        if self.sample_stds is not None:
            inds, seqs, one_hot, enrichment, stds = batch

        else:
            inds, seqs, one_hot, enrichment = batch

        preds = self(one_hot.type(torch.get_default_dtype()))

        batch_out = {}

        loss_per_seq = self.regression_loss(preds, enrichment)

        batch_out["loss"] = loss_per_seq.mean()

        self.log("val_loss", batch_out["loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)

        return batch_out

    def adjust_labels(self, preds, current_labels, original_labels, stds, steps=2, lr=0.1):
        ystar = torch.nn.Parameter(current_labels.detach().clone())
        p = preds.detach().clone()
        ol = original_labels.detach().clone()
        cl = current_labels.detach().clone()

        max_dev = ol + stds
        min_dev = ol - stds
        zero = torch.zeros_like(max_dev, device=self.device)

        optimizer = torch.optim.SGD(lr=lr, params=[ystar])
        loss = p - ystar
        for _ in range(steps):
            likeli = 1 / stds * math.sqrt(2 * math.pi) * torch.exp(-torch.pow((ystar - cl) / (2 * stds), 2)) * (1 + torch.erf(loss * (ystar - cl) / (stds * math.sqrt(2))))
            log_likeli = torch.log(likeli) * stds
            nll_loss = -log_likeli.mean()
            optimizer.zero_grad()
            nll_loss.backward()
            optimizer.step()

        return torch.clip(ystar, min_dev, max_dev).detach()


if __name__ == '__main__':

    config = {
        "seed": 69,
        "precision": "double",
        "data_worker_num": 6,
        "gpus": 1,
        "epochs": 500,
        "batch_size": 500,
        "optimizer": "Adam",
        "lr": 0.005,
        "lr_final": None,
        "weight_decay": 0.001,
        "decay_after": 0.75,
        "label_spacing": [0.0, 6.0, 16],
        "label_groups": 2,
        # "fasta_file": "../../../regression_model_comparison/cov/cov_r12_v_r10_all.fasta",
        # "weights": "../../../regression_model_comparison/cov/cov_r12_v_r10_all_uniform_weights.json",
        # "fasta_file": "../../../regression_model_comparison/cov/cov_z_avg_enriched.fasta",
        "fasta_file": "../../../regression_model_comparison/cov/cov_z_full.fasta",
        # "weights": "../../../regression_model_comparison/cov/cov_z_avg_scores_std.json",
        # "weights": "../../../regression_model_comparison/cov/cov_z_avg_enriched_normal_weights.json",
        # "weights": "fasta",
        "weights": "../../../regression_model_comparison/cov/cov_z_full_scores_std.json",
        "molecule": "dna",
        "v_num": 40,
        "q": 5,
        "validation_set_size": 0.1,
        "test_set_size": 0.0,
        "sampling_weights": None,
        "dr": 0.05,
        "max_node_size": 24,
        "latent_dim": 50,
        "node_depth": 2,
        "adj_depth": 2,
        "predictor_depth": 2,
        "alpha": 0.
    }

    # Deal with weights
    weights = None
    config["sampling_weights"] = None
    config["sample_stds"] = None
    if "fasta" in config["weights"]:
        weights = config["weights"]  # All weights are already in the processed fasta files
    elif config["weights"] is None or config["weights"] in ["None", "none", "equal"]:
        pass
    else:
        ## Assumes weight file to be in same directory as our data files.
        try:
            with open(config["weights"]) as f:
                data = json.load(f)
            weights = np.asarray(data["weights"])

            # Deal with Sampling Weights
            try:
                sampling_weights = np.asarray(data["sampling_weights"])
            except KeyError:
                sampling_weights = None
            config["sampling_weights"] = sampling_weights

            # Deal with Sample Stds
            try:
                sample_stds = np.asarray(data["sample_stds"])
            except KeyError:
                sample_stds = None
            config["sample_stds"] = sample_stds

        except IOError:
            print(f"Could not load provided weight file {config['weights']}")
            exit(-1)

    config["sequence_weights"] = weights



    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning import Trainer

    debug = False
    if debug:
        config["worker_num"] = 0

    model = SCGCN(config, debug=debug, precision=config["precision"])

    logger = TensorBoardLogger("./", name="SCGCN")

    if config["gpus"]:
        tr = Trainer(max_epochs=config['epochs'], logger=logger, accelerator="cuda", devices=1)
    else:
        tr = Trainer(max_epochs=config['epochs'], logger=logger, accelerator="cpu")

    tr.fit(model)

