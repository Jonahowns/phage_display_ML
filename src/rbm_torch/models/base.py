import time
import pandas as pd
import math
import json
import sys
import numpy as np
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
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


# class that takes care of generic methods for all of the models
class Base(LightningModule):
    def __init__(self, config, debug=False, precision="double"):
        super().__init__()

        # REMOVE THESE
        self.itlm_alpha = 0.0
        self.sample_stds = None

        # Pytorch Basic Options #
        ################################
        self.seed = config['seed']
        torch.manual_seed(self.seed)  # For reproducibility
        supported_precisions = {"double": torch.float64, "single": torch.float32}
        try:
            torch.set_default_dtype(supported_precisions[precision])
        except:
            print(f"Precision {precision} not supported.")
            sys.exit(-1)
        ################################

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


        # Sequence Weighting Weights
        # Not pretty but necessary to either provide the weights or to import from the fasta file
        # To import from the provided fasta file weights="fasta" in intialization of RBM
        weights = config['sequence_weights']
        self.weights = process_weights(weights)


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

        self.sampling_weights = config["sampling_weights"]
        # Labels for stratified sampling of datasets
        self.label_spacing = config["label_spacing"]
        self.label_groups = len(self.label_spacing) - 1
        assert len(self.label_spacing) - 1 == self.label_groups

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
                sys.exit()

            if q_data != self.q:
                print(
                    f"State Number mismatch! Expected q={self.q}, in dataset q={q_data}. All observed chars: {all_chars}")
                sys.exit(-1)

            # seq_read_counts = np.asarray([math.log(x + 1.0, math.e) for x in seq_read_counts])
            data = pd.DataFrame(data={'sequence': seqs, 'fasta_count': seq_read_counts})

            if type(self.weights) == str and "fasta" in self.weights:
                weights = np.asarray(seq_read_counts)
                data["seq_count"] = weights

            data_pds.append(data)

        all_data = pd.concat(data_pds)
        if type(self.weights) is np.ndarray:
            all_data["seq_count"] = self.weights
        if self.sample_stds is not None:
            all_data["sample_std"] = self.sample_stds

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
                not_test_inds, test_inds = next(GroupShuffleSplit(test_size=self.test_size, n_splits=1, random_state=self.seed).split(label_df, groups=label_df['sequence']))
                test_sets += label_df.index[test_inds].to_list()

                # Further split training set into train and test set
                train_inds, val_inds = next(GroupShuffleSplit(test_size=self.validation_size, n_splits=1, random_state=self.seed).split(label_df.iloc[not_test_inds], groups=label_df.iloc[not_test_inds]['sequence']))
                train_sets += label_df.iloc[not_test_inds].index[train_inds].to_list()
                val_sets += label_df.iloc[not_test_inds].index[val_inds].to_list()

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

    ## Sets Up Optimizer as well as Exponential Weight Decasy
    def configure_optimizers(self):
        optim = self.optimizer(self.parameters(), lr=self.lr, weight_decay=self.wd)
        # optim = self.optimizer(self.weight_param)
        # Exponential Weight Decay after set amount of epochs (set by decay_after)
        decay_gamma = (self.lrf / self.lr) ** (1 / (self.epochs * (1 - self.decay_after)))
        decay_milestone = math.floor(self.decay_after * self.epochs)
        my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=[decay_milestone], gamma=decay_gamma)
        optim_dict = {"lr_scheduler": my_lr_scheduler,
                      "optimizer": optim}
        return optim_dict

    ## Loads Training Data
    def train_dataloader(self, init_fields=True):
        # Get Correct Weights
        training_weights = None
        if "seq_count" in self.training_data.columns:
            training_weights = self.training_data["seq_count"].tolist()

        training_stds = None
        if "sample_std" in self.training_data.columns:
            training_stds = self.training_data["sample_std"].tolist()

        train_reader = Categorical(self.training_data, self.q, weights=training_weights, max_length=self.v_num,
                                   molecule=self.molecule, device=self.device, one_hot=True, labels=False, additional_data=training_stds)


        return torch.utils.data.DataLoader(
            train_reader,
            batch_size=self.batch_size,
            num_workers=self.worker_num,  # Set to 0 if debug = True
            pin_memory=self.pin_mem,
            shuffle=True
        )

    def val_dataloader(self):
        # Get Correct Validation weights
        validation_weights = None
        if "seq_count" in self.validation_data.columns:
            validation_weights = self.validation_data["seq_count"].tolist()

        validation_stds = None
        if "sample_std" in self.validation_data.columns:
            validation_stds = self.validation_data["sample_std"].tolist()

        val_reader = Categorical(self.validation_data, self.q, weights=validation_weights, max_length=self.v_num,
                                 molecule=self.molecule, device=self.device, one_hot=True, labels=None, additional_data=validation_stds)

        return torch.utils.data.DataLoader(
            val_reader,
            batch_size=self.batch_size,
            num_workers=self.worker_num,  # Set to 0 to view tensors while debugging
            pin_memory=self.pin_mem,
            shuffle=False
        )

    def validation_epoch_end(self, outputs):
        result_dict = {}
        for key, value in outputs[0].items():
            result_dict[key] = torch.stack([x[key] for x in outputs]).mean()

        self.logger.experiment.add_scalars("Val Scalars", result_dict, self.current_epoch)

    ## On Epoch End Collects Scalar Statistics and Distributions of Parameters for the Tensorboard Logger
    def training_epoch_end(self, outputs):
        result_dict = {}
        for key, value in outputs[0].items():
            if key == "loss":
                result_dict[key] = torch.stack([x[key].detach() for x in outputs]).mean()
            else:
                result_dict[key] = torch.stack([x[key] for x in outputs]).mean()

        self.logger.experiment.add_scalars("Train Scalars", result_dict, self.current_epoch)

        for name, p in self.named_parameters():
            self.logger.experiment.add_histogram(name, p.detach(), self.current_epoch)


# class FASTADataModule(LightningDataModule):
#     def __init__(self, fasta_file, molecule, v_num, q, label_spacing, label_groups, batch_size: int = 32, worker_num=1, val_size=0.15, test_size=0.1, seed=0):
#         super().__init__()
#         self.batch_size = batch_size
#         self.fasta_file = fasta_file
#         self.worker_num = worker_num
#         self.molecule = molecule
#
#
#         self.val_size = val_size
#         self.test_size = test_size
#         self.q = q
#         self.v_num = v_num
#         self.label_spacing = label_spacing
#         self.label_groups = label_groups
#         self.seed = seed
#
#
#     def setup(self, stage: str):
#         self.additional_data = False
#         if type(self.fasta_file) is str:
#             self.fasta_file = [self.fasta_file]
#
#         assert type(self.fasta_file) is list
#         data_pds = []
#
#         for file in self.fasta_file:
#             try:
#                 if self.worker_num == 0:
#                     threads = 1
#                 else:
#                     threads = self.worker_num
#                 seqs, seq_read_counts, all_chars, q_data = fasta_read(file, self.molecule, drop_duplicates=False, threads=threads)
#             except IOError:
#                 print(f"Provided Fasta File '{file}' Not Found")
#                 print(f"Current Directory '{os.getcwd()}'")
#                 sys.exit()
#
#             if q_data != self.q:
#                 print(
#                     f"State Number mismatch! Expected q={self.q}, in dataset q={q_data}. All observed chars: {all_chars}")
#                 sys.exit(-1)
#
#             # seq_read_counts = np.asarray([math.log(x + 1.0, math.e) for x in seq_read_counts])
#             data = pd.DataFrame(data={'sequence': seqs, 'fasta_count': seq_read_counts})
#
#             if type(self.weights) == str and "fasta" in self.weights:
#                 weights = np.asarray(seq_read_counts)
#                 data["seq_count"] = weights
#
#             data_pds.append(data)
#
#         all_data = pd.concat(data_pds)
#         if type(self.weights) is np.ndarray:
#             all_data["seq_count"] = self.weights
#         if self.sample_stds is not None:
#             all_data["sample_std"] = self.sample_stds
#
#         assert len(all_data["sequence"][0]) == self.v_num  # make sure v_num is same as data_length
#
#         # stratify_labels = None
#         # if self.stratify or self.pearson_xvar == "label" or self.sampling_strategy == "stratified":
#
#         # w8s = all_data.seq_count.to_numpy()
#         labels = self.label_samples(all_data["fasta_count"], self.label_spacing, self.label_groups)
#
#         all_data["label"] = labels
#         # stratify_labels = labels
#
#         # else:
#         #     all_data["label"] = 0.
#
#         train_sets, val_sets, test_sets = [], [], []
#         for i in range(self.label_groups):
#             label_df = all_data[all_data["label"] == i]
#             if self.test_size > 0.:
#                 # Split label df into train and test sets, taking into account duplicates
#                 train_inds, test_inds = next(GroupShuffleSplit(test_size=self.test_size, n_splits=1, random_state=self.seed).split(label_df, groups=label_df['sequence']))
#                 test_sets += label_df.index[test_inds].to_list()
#
#                 # Further split training set into train and test set
#                 train_inds, val_inds = next(GroupShuffleSplit(test_size=self.validation_size, n_splits=1, random_state=self.seed).split(label_df[train_inds], groups=label_df['sequence']))
#                 train_sets += label_df.index[train_inds].to_list()
#                 val_sets += label_df.index[val_inds].to_list()
#
#             else:
#                 # Split label df into train and validation sets, taking into account duplicates
#                 train_inds, val_inds = next(GroupShuffleSplit(test_size=self.validation_size, n_splits=1, random_state=self.seed).split(label_df, groups=label_df['sequence']))
#                 train_sets += label_df.index[train_inds].to_list()
#                 val_sets += label_df.index[val_inds].to_list()
#
#         self.training_data = all_data.iloc[train_sets]
#         self.validation_data = all_data.iloc[val_sets]
#
#         if self.sampling_weights is not None:
#             self.sampling_weights = self.sampling_weights[train_sets]
#
#         self.dataset_indices = {"train_indices": train_sets, "val_indices": val_sets}
#         if self.test_size > 0:
#             self.test_data = all_data.iloc[test_sets]
#             self.dataset_indices["test_indices"] = test_sets
#
#     def train_dataloader(self):
#         return DataLoader(self.mnist_train, batch_size=self.batch_size)
#
#     def val_dataloader(self):
#         return DataLoader(self.mnist_val, batch_size=self.batch_size)
#
#     def test_dataloader(self):
#         return DataLoader(self.mnist_test, batch_size=self.batch_size)
#
#     def teardown(self, stage: str):
#         # Used to clean-up when the run is finished
#         ...
#
#     def label_samples(self, w8s, label_spacing, label_groups):
#         if type(label_spacing) is list:
#             bin_edges = label_spacing
#         else:
#             if label_spacing == "log":
#                 bin_edges = np.geomspace(np.min(w8s), np.max(w8s), label_groups + 1)
#             elif label_spacing == "lin":
#                 bin_edges = np.linspace(np.min(w8s), np.max(w8s), label_groups + 1)
#             else:
#                 print(f"pearson label spacing option {label_spacing} not supported!")
#                 exit()
#         bin_edges = bin_edges[1:]
#
#         def assign_label(x):
#             bin_edge = bin_edges[0]
#             idx = 0
#             while x > bin_edge:
#                 idx += 1
#                 bin_edge = bin_edges[idx]
#
#             return idx
#
#         labels = list(map(assign_label, w8s))
#         return labels