import time
import pandas as pd
import math
import json
import sys
import numpy as np
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
# from pytorch_lightning.profiler import SimpleProfiler, PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split, GroupShuffleSplit

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW, Adagrad, Adadelta  # Supported Optimizers
from multiprocessing import cpu_count # Just to set the worker number
from torch.autograd import Variable

from rbm_torch.utils.utils import Categorical, fasta_read, label_samples, process_weights, configure_optimizer, StratifiedBatchSampler, WeightedSubsetRandomSampler
from torch.utils.data import WeightedRandomSampler


# class that takes care of generic methods for all models
class Base(LightningModule):
    def __init__(self, config, debug=False, precision="double"):
        super().__init__()

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

        matmul_precisions = {"single": "medium", "double": "high"}
        torch.set_float32_matmul_precision(matmul_precisions[precision])
        ################################

        self.batch_size = config['batch_size']  # Pretty self-explanatory
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
        # this attribute is set in load_run
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

        # Labels for stratified sampling of datasets
        try:
            self.label_groups = config["label_groups"]
        except KeyError:
            self.label_groups = 1


        if self.label_groups > 1:
            try:
                self.label_spacing = config["label_spacing"]
            except KeyError:
                print("Label Spacing Must be defined in run file!")
            assert len(self.label_spacing) - 1 == self.label_groups
        else:
            self.label_spacing = []

        try:
            self.group_fraction = config["group_fraction"]
            assert self.label_groups == len(self.group_fraction)
        except KeyError:
            self.group_fraction = [1 / self.label_groups for i in range(self.label_groups)]

        try:
            self.sample_multiplier = config["sample_multiplier"]
        except KeyError:
            self.sample_multiplier = 1.

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

        self.training_data_logs = []
        self.val_data_logs = []



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

        assert len(all_data["sequence"][0]) == self.v_num  # make sure v_num is same as data_length

        if self.label_spacing == []:
            self.label_spacing = [min(all_data["fasta_count"].tolist()) - 0.05, max(all_data["fasta_count"].tolist()) + 0.05]
        labels = label_samples(all_data["fasta_count"], self.label_spacing, self.label_groups)

        all_data["label"] = labels

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
            if self.sampling_weights == "fasta":
                self.sampling_weights = np.exp(all_data["fasta_count"].to_numpy())
            self.sampling_weights = self.sampling_weights[train_sets]

            self.sampling_weights = torch.tensor(self.sampling_weights)

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
        # Exponential Weight Decay after set amount of epochs (set by decay_after)
        decay_gamma = (self.lrf / self.lr) ** (1 / (self.epochs * (1 - self.decay_after)))
        decay_milestone = math.floor(self.decay_after * self.epochs)
        my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=[decay_milestone], gamma=decay_gamma)
        optim_dict = {"lr_scheduler": my_lr_scheduler,
                      "optimizer": optim}
        return optim_dict

    ## Loads Training Data
    def train_dataloader(self, init_fields=False):
        training_weights = None
        if "seq_count" in self.training_data.columns:
            training_weights = self.training_data["seq_count"].tolist()

        train_reader = Categorical(self.training_data, self.q, weights=training_weights, max_length=self.v_num,
                                   molecule=self.molecule, device=self.device, one_hot=True, labels=False)
        # Init Fields
        if init_fields:
            if hasattr(self, "fields"):
                with torch.no_grad():
                    initial_fields = train_reader.field_init()
                    self.fields += initial_fields
                    self.fields0 += initial_fields

        else:
            if hasattr(self, "fields"):
                with torch.no_grad():
                    initial_fields = torch.randn((self.v_num, self.q), device=self.device)*0.01
                    self.fields += initial_fields
                    self.fields0 += initial_fields

        # Sampling
        if self.sampling_strategy == "stratified":
            return torch.utils.data.DataLoader(
                train_reader,
                batch_sampler=StratifiedBatchSampler(self.training_data["label"].to_numpy(), batch_size=self.batch_size,
                                                     shuffle=True, seed=self.seed),
                num_workers=self.worker_num,  # Set to 0 if debug = True
                pin_memory=self.pin_mem
            )
        elif self.sampling_strategy == "weighted":
            return torch.utils.data.DataLoader(
                train_reader,
                sampler=WeightedRandomSampler(weights=self.sampling_weights, num_samples=self.batch_size * self.sample_multiplier, replacement=True),
                num_workers=self.worker_num,  # Set to 0 if debug = True
                batch_size=self.batch_size,
                pin_memory=self.pin_mem
            )
        elif self.sampling_strategy == "stratified_weighted":
            return torch.utils.data.DataLoader(
                train_reader,
                batch_sampler=WeightedSubsetRandomSampler(self.sampling_weights, self.training_data["label"].to_numpy(), self.group_fraction, self.batch_size, self.sample_multiplier),
                num_workers=self.worker_num,  # Set to 0 if debug = True
                pin_memory=self.pin_mem
            )
        else:
            self.sampling_strategy = "random"
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

        val_reader = Categorical(self.validation_data, self.q, weights=validation_weights, max_length=self.v_num,
                                 molecule=self.molecule, device=self.device, one_hot=True, labels=None, additional_data=None)

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

    def on_validation_epoch_end(self):
        result_dict = {}
        for key in self.val_data_logs[0].keys():
            result_dict[key] = torch.stack([x[key] for x in self.val_data_logs]).mean()

        self.logger.experiment.add_scalars("Val Scalars", result_dict, self.current_epoch)

        self.val_data_logs.clear()

    ## On Epoch End Collects Scalar Statistics and Distributions of Parameters for the Tensorboard Logger
    def on_train_epoch_end(self):
        result_dict = {}
        for key in self.training_data_logs[0].keys():
            if key == "loss":
                result_dict[key] = torch.stack([x[key].detach() for x in self.training_data_logs]).mean()
            else:
                try:
                    result_dict[key] = torch.stack([x[key] for x in self.training_data_logs]).mean()
                except RuntimeError:
                    print('sup bitch')

        self.logger.experiment.add_scalars("Train Scalars", result_dict, self.current_epoch)

        for name, p in self.named_parameters():
            self.logger.experiment.add_histogram(name, p.detach(), self.current_epoch)

        self.training_data_logs.clear()
            

# class that extends base class for any of our rbm/crbm models
class Base_drelu(Base):
    def __init__(self, config, debug=False, precision="double"):
        super().__init__(config, debug=debug, precision=precision)
        self.dr = 0.
        if "dr" in config.keys():
            self.dr = config["dr"]

        # Constants for faster math
        self.logsqrtpiover2 = torch.tensor(0.2257913526, device=self.device, requires_grad=False)
        self.pbis = torch.tensor(0.332672, device=self.device, requires_grad=False)
        self.a1 = torch.tensor(0.3480242, device=self.device, requires_grad=False)
        self.a2 = torch.tensor(- 0.0958798, device=self.device, requires_grad=False)
        self.a3 = torch.tensor(0.7478556, device=self.device, requires_grad=False)
        self.invsqrt2 = torch.tensor(0.7071067812, device=self.device, requires_grad=False)
        self.sqrt2 = torch.tensor(1.4142135624, device=self.device, requires_grad=False)

    # Return param as a numpy array
    def get_param(self, param_name):
        try:
            tensor = getattr(self, param_name).clone()
            return tensor.detach().numpy()
        except KeyError:
            print(f"Key {param_name} not found")
            sys.exit(1)

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
