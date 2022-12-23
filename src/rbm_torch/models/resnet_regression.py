from rbm_torch.models.base import Base
import pytorch_lightning as pl
import torchvision.models as models
from torch.optim import SGD, Adam
import torch.nn as nn
import torch
import numpy as np
import math
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import json
from rbm_torch.utils.utils import Categorical, fasta_read, conv2d_dim, pool1d_dim, BatchNorm1D, label_samples, process_weights
import os
import pandas as pd


class ResNet(Base):
    def __init__(self, config, debug=False, precision="double"):
        super().__init__(config, debug=debug, precision=precision)

        self.__dict__.update(locals())
        resnets = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }

        self.resnet_version = config["resnet_version"]

        self.criterion = torch.nn.MSELoss(reduction="mean")

        # Using a pretrained ResNet backbone
        self.resnet_model = resnets[self.resnet_version](pretrained=True)
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features

        fcn = [
            nn.Dropout(config['dr']),
            nn.Linear(linear_size, linear_size // 2),
            nn.ReLU(),
            nn.Linear(linear_size // 2, 1)
        ]

        self.fcn = nn.Sequential(*fcn)
        self.resnet_model.conv1 = torch.nn.Conv2d(1, 64, (7, self.q), (2, 2), bias=True)

        modules = list(self.resnet_model.children())[:-1]  # delete the last fc layer.
        self.resnet_model = nn.Sequential(*modules)
        self.save_hyperparameters()

    def forward(self, X):
        x = self.resnet_model(X)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fcn(x)
        return x

    def training_step(self, batch, batch_idx):
        inds, seq, x, y = batch
        pred = self(x.double().unsqueeze(1))
        train_loss = self.criterion(pred.squeeze(1), y)

        # # Convert to labels
        # preds = torch.argmax(softmax, 1).clone().double() # convert to torch float 64
        #
        # predcpu = list(preds.detach().cpu().numpy())
        # ycpu = list(y.detach().cpu().numpy())
        # train_acc = sklearn.metrics.balanced_accuracy_score(ycpu, predcpu)

        # perform logging
        self.log("ptl/train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return train_loss


    def validation_step(self, batch, batch_idx):
        inds, seq, x, y = batch
        pred = self(x.double().unsqueeze(1))
        val_loss = self.criterion(pred.squeeze(1), y)

        # # Convert to labels
        # preds = torch.argmax(softmax, 1).clone().double()  # convert to torch float 64
        #
        # predcpu = list(preds.detach().cpu().numpy())
        # ycpu = list(y.detach().cpu().numpy())
        # val_acc = sklearn.metrics.balanced_accuracy_score(ycpu, predcpu)

        # perform logging
        self.log("ptl/val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        # self.log("ptl/val_accuracy", val_acc, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": val_loss}


if __name__ == '__main__':
    config = {
        "seed": 69,
        "precision": "double",
        "data_worker_num": 6,
        "gpus": 1,
        "batch_size": 500,
        "epochs": 500,
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
        "weights": "fasta",
        "molecule": "dna",
        "v_num": 40,
        "q": 5,
        "validation_set_size": 0.1,
        "test_set_size": 0.0,
        "sampling_weights": None,
        "dr": 0.05,
        "alpha": 0.0,
        "resnet_version": 18,
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

    model = ResNet(config, debug=debug, precision=config["precision"])

    logger = TensorBoardLogger("./", name="RESNET")

    if config["gpus"]:
        tr = Trainer(max_epochs=config['epochs'], logger=logger, accelerator="cuda", devices=1)
    else:
        tr = Trainer(max_epochs=config['epochs'], logger=logger, accelerator="cpu")

    tr.fit(model)