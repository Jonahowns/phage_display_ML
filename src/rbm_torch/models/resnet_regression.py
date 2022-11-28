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

class ResNet(pl.LightningModule):
    def __init__(self, config, num_classes, resnet_version,
                test_path=None,
                 optimizer='adam',
                 transfer=True):
        super().__init__()

        self.__dict__.update(locals())
        resnets = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        # hyperparameters
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        # for importing different versions of the data
        # self.datatype = config['datatype']

        self.training_data = None
        self.validation_data = None

        self.criterion = torch.nn.MSELoss(reduction="sum")

        # Using a pretrained ResNet backbone
        self.resnet_model = resnets[resnet_version](pretrained=transfer)
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features

        # replace final layer for fine tuning
        fcn = [
            nn.Dropout(config['dr']),
            nn.Linear(linear_size, 1),
            nn.Sigmoid()
        ]

        self.fcn = nn.Sequential(*fcn)
        self.resnet_model.conv1 = torch.nn.Conv1d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)

        modules = list(self.resnet_model.children())[:-1]  # delete the last fc layer.
        self.resnet_model = nn.Sequential(*modules)

    def forward(self, X):
        x = self.resnet_model(X)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fcn(x)
        return x

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def train_dataloader(self, init_fields=True):
        # Get Correct Weights
        training_weights = None
        if "seq_count" in self.training_data.columns:
            training_weights = self.training_data["seq_count"].tolist()

        train_reader = Categorical(self.training_data, self.q, weights=training_weights, max_length=self.v_num,
                                   molecule=self.molecule, device=self.device, one_hot=True, labels=False, additional_data=None)

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
        validation_weights = None
        if "seq_count" in self.validation_data.columns:
            validation_weights = self.validation_data["seq_count"].tolist()

        labels = False

        val_reader = Categorical(self.validation_data, self.q, weights=validation_weights, max_length=self.v_num,
                                 molecule=self.molecule, device=self.device, one_hot=True, labels=labels, additional_data=None)


        return torch.utils.data.DataLoader(
            val_reader,
            batch_size=self.batch_size,
            num_workers=self.worker_num,  # Set to 0 to view tensors while debugging
            pin_memory=self.pin_mem,
            shuffle=False
        )

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

            seq_read_counts = np.asarray([math.log(x + 1.0, math.e) for x in seq_read_counts])
            data = pd.DataFrame(data={'sequence': seqs, 'fasta_count': seq_read_counts})

            if type(self.weights) == str and "fasta" in self.weights:
                weights = np.asarray(seq_read_counts)
                data["seq_count"] = weights

            data_pds.append(data)

        all_data = pd.concat(data_pds)
        if type(self.weights) is np.ndarray:
            all_data["seq_count"] = self.weights

        assert len(all_data["sequence"][0]) == self.v_num  # make sure v_num is same as data_length

        labels = label_samples(all_data["fasta_count"], self.label_spacing, self.label_groups)

        all_data["label"] = labels

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

    def on_train_start(self):
        # Log which sequences belong to each dataset
        with open(self.logger.log_dir + "/dataset_indices.json", "w") as f:
            json.dump(self.dataset_indices, f)

    def training_step(self, batch, batch_idx):
        seq, x, y = batch
        pred = self(x.unsqueeze(1))
        train_loss = self.criterion(pred, y)

        # # Convert to labels
        # preds = torch.argmax(softmax, 1).clone().double() # convert to torch float 64
        #
        # predcpu = list(preds.detach().cpu().numpy())
        # ycpu = list(y.detach().cpu().numpy())
        # train_acc = sklearn.metrics.balanced_accuracy_score(ycpu, predcpu)

        # perform logging
        self.log("ptl/train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("ptl/train_accuracy", train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss


    def validation_step(self, batch, batch_idx):
        seq, x, y = batch
        preds = self(x.unsqueeze(1))
        val_loss = self.criterion(preds, y)

        # # Convert to labels
        # preds = torch.argmax(softmax, 1).clone().double()  # convert to torch float 64
        #
        # predcpu = list(preds.detach().cpu().numpy())
        # ycpu = list(y.detach().cpu().numpy())
        # val_acc = sklearn.metrics.balanced_accuracy_score(ycpu, predcpu)

        # perform logging
        self.log("ptl/val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True)
        # self.log("ptl/val_accuracy", val_acc, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": val_loss}
