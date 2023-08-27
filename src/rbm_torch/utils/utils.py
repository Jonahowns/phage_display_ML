"""
 Copyright 2018 - by Jerome Tubiana (jertubiana@gmail.com)
     All rights reserved

     Permission is granted for anyone to copy, use, or modify this
     software for any uncommercial purposes, provided this copyright
     notice is retained, and note is made of any changes that have
     been made. This software is distributed without any warranty,
     express or implied. In no event shall the author or contributors be
     liable for any damage arising out of the use of this software.

     The publication of research using this software, modified or not, must include
     appropriate citations to:

     Modifications-> Bug fixes in Sequence_logo_all
                  -> Sequence Logos use molecule keyword argument to produce the correct plot
                  -> Rewrote fasta reader and put here
                  -> Multi threaded fasta reader added
                  -> Updated sampling methods for CRBM and RBM

"""

import os
import matplotlib as mpl
import torch
import torch.nn.functional as F
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
import numpy as np
import matplotlib.pyplot as plt
from rbm_torch.utils.graph_utils import Sequence_logo, Sequence_logo_multiple, Sequence_logo_all
from matplotlib.ticker import FormatStrFormatter
import copy
from PIL import Image
import math
import time
from multiprocessing import Pool
import pandas as pd
import types
import json
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from torch.optim import SGD, AdamW, Adagrad, Adadelta, Adam  # Supported Optimizers
from torch.utils.data.sampler import Sampler


# Globals used for Converting Sequence Strings to Integers
aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
aalower = [x.lower() for x in aa]
aadictU = {aa[k]: k for k in range(len(aa))}
aadictL = {aalower[k]:k for k in range(len(aalower))}
aadict = {**aadictU, **aadictL}
aadict_inverse = {v: k for k, v in aadictU.items()}

dna = ['A', 'C', 'G', 'T', '-']
dnalower = [x.lower() for x in dna]
dnadictU = {dna[k]: k for k in range(len(dna))}
dnadictL = {dnalower[k]: k for k in range(len(dnalower))}
dnadict = {**dnadictU, **dnadictL}
dnadict_inverse = {v: k for k, v in dnadictU.items()}

rna = ['A', 'C', 'G', 'U', '-']
rnalower = [x.lower() for x in rna]
rnadictU = {rna[k]: k for k in range(len(rna))}
rnadictL = {rnalower[k]: k for k in range(len(rnalower))}
rnadict = {**rnadictU, **rnadictL}
rnadict_inverse = {v: k for k, v in rnadict.items()}

# Deal with wildcard values, all are equivalent
dnadict['*'] = dnadict['-']
dnadict['N'] = dnadict['-']
dnadict['n'] = dnadict['-']

rnadict['*'] = rnadict['-']
rnadict['N'] = rnadict['-']
rnadict['n'] = rnadict['-']

# Changing X to be the same value as a gap as it can mean any amino acid
aadict['X'] = aadict['-']
aadict['x'] = aadict['-']
aadict['*'] = aadict['-']

aadict['B'] = len(aa)
aadict['Z'] = len(aa)
aadict['b'] = len(aa)
aadict['z'] = -1
aadict['.'] = -1

letter_to_int_dicts = {"protein": aadict, "dna": dnadict, "rna": rnadict}
int_to_letter_dicts = {"protein": aadict_inverse, "dna": dnadict_inverse, "rna": rnadict_inverse}


import math
import torch
import time


optimizer_dict = {"SGD": SGD, "AdamW": AdamW, "Adadelta": Adadelta, "Adagrad": Adagrad, "Adam": Adam}


def load_run(runfile):
    if type(runfile) is str:
        try:
            with open(runfile, "r") as f:
                run_data = json.load(f)
        except IOError:
            print(f"Runfile {runfile} not found or empty! Please check!")
            exit(1)
    elif type(runfile) is dict:
        run_data = runfile
    else:
        print("Unsupported Format for run configuration. Must be filename of json config file or dictionary")
        exit(1)

    # Get info needed for all models
    # assert run_data["model_type"] in ["rbm", "crbm", "exp_rbm", "exp_crbm", "net_crbm", "pcrbm", "pool_crbm", "comp_crbm", "pool_class_crbm", "pool_regression_crbm", "variational_pool_crbm"]

    config = run_data["config"]

    data_dir = run_data["data_dir"]
    fasta_file = run_data["fasta_file"]

    # Deal with weights
    # Deal with weights
    weights = None
    # config["sampling_weights"] = None
    config["sample_stds"] = None
    if "fasta" in run_data["weights"]:
        weights = run_data["weights"]  # All weights are already in the processed fasta files
    elif run_data["weights"] is None or run_data["weights"] in ["None", "none", "equal"]:
        pass
    else:
        ## Assumes weight file to be in same directory as our data files.
        try:
            with open(run_data["data_dir"]+run_data["weights"]) as f:
                data = json.load(f)
            weights = np.asarray(data["weights"])

            # Deal with Sampling Weights
            try:
                if type(config["sampling_weights"]) is not str:
                    config["sampling_weights"] = np.asarray(config["sampling_weights"])

                sampling_weights = np.asarray(data["sampling_weights"])
                config["sampling_weights"] = sampling_weights
            except KeyError:
                config['sampling_weights'] = None

            # Deal with Sample Stds
            # try:
            #     sample_stds = np.asarray(data["sample_stds"])
            #     config["sample_stds"] = sample_stds
            # except KeyError:
            #     sample_stds = None

        except IOError:
            print(f"Could not load provided weight file {config['weights']}")
            exit(-1)

    config["sequence_weights"] = weights

    # Edit config for dataset specific hyperparameters
    if type(fasta_file) is not list:
        fasta_file = [fasta_file]

    config["fasta_file"] = [data_dir + f for f in fasta_file]

    seed = np.random.randint(0, 10000, 1)[0]
    config["seed"] = int(seed)
    if config["lr_final"] == "None":
        config["lr_final"] = None

    if "crbm" in run_data["model_type"]:
        # added since json files don't support tuples
        for key, val in config["convolution_topology"].items():
            for attribute in ["kernel", "dilation", "padding", "stride", "output_padding"]:
                val[f"{attribute}"] = (val[f"{attribute}x"], val[f"{attribute}y"])

    config["gpus"] = run_data["gpus"]
    return run_data, config

def process_weights(weights):
    if weights is None:
        return None
    elif type(weights) == str:
        if weights == "fasta":  # Assumes weights are in fasta file
            return "fasta"
        else:
            print(f"String Option {weights} not supported")
            exit(1)
    elif type(weights) == torch.tensor:
        return weights.numpy()
    elif type(weights) == np.ndarray:
        return weights
    else:
        print(f"Provided Weights of type {type(weights)} Not Supported, Must be None, a numpy array, torch tensor, or 'fasta'")
        exit(1)

def configure_optimizer(optimizer_str):
    try:
        return optimizer_dict[optimizer_str]
    except KeyError:
        print(f"Optimizer {optimizer_str} is not supported")
        exit(1)

# adapted from https://github.com/pytorch/pytorch/issues/7359
class WeightedSubsetRandomSampler:
    r"""Samples elements from a given list of indices with given probabilities (weights), with replacement.

    Arguments:
        weights (sequence)   : a 2ce of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
    """

    def __init__(self, weights, labels, group_fraction, batch_size, batches, per_sample_replacement=False):
        if not isinstance(batch_size, int):
            raise ValueError("num_samples should be a non-negative integer "
                             "value, but got num_samples={}".format(batch_size))
        if not isinstance(batches, int):
            raise ValueError("num_samples should be a non-negative integer "
                             "value, but got num_samples={}".format(batches))

        self.batch_size = batch_size
        self.n_batches = batches

        self.weights = torch.tensor(weights, dtype=torch.double)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.indices = torch.arange(0, self.weights.shape[0], 1)

        self.replacement = per_sample_replacement  # can't think of a good reason to have this on

        self.label_set = list(set(labels))
        self.sample_per_label = [math.floor(x*self.batch_size) for x in group_fraction]
        # self.sample_per_label = self.num_samples // len(self.label_set)

        self.label_weights, self.label_indices = [], []
        for i in self.label_set:
            lm = labels == i
            self.label_indices.append(self.indices[lm])
            self.label_weights.append(self.weights[lm])

    def __iter__(self):
        batch_samples = []
        for i in range(self.n_batches):
            samples = []
            for j in range(len(self.label_set)):
                samples.append(self.label_indices[j][torch.multinomial(self.label_weights[j], self.sample_per_label[j], self.replacement)])

            batch_samples.append(torch.cat(samples, dim=0))

        for bs in batch_samples:
            yield bs

    def __len__(self):
        return self.n_batches


# copied from https://discuss.pytorch.org/t/how-to-enable-the-dataloader-to-sample-from-each-class-with-equal-probability/911/6
class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True, seed=38):
        if torch.is_tensor(y):
            y = y.cpu().numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        self.n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=self.n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y), 1).numpy()
        self.y = y
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = self.seed  # should be governed by globally set seed

        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        # return len(self.y)
        return self.n_batches


def label_samples(w8s, label_spacing, label_groups):
    if type(label_spacing) is list:
        bin_edges = label_spacing
    else:
        if label_spacing == "log":
            bin_edges = np.geomspace(np.min(w8s), np.max(w8s), label_groups + 1)
        elif label_spacing == "lin":
            bin_edges = np.linspace(np.min(w8s), np.max(w8s), label_groups + 1)
        else:
            print(f"pearson label spacing option {label_spacing} not supported!")
            exit()
    bin_edges = bin_edges[1:]

    def assign_label(x):
        bin_edge = bin_edges[0]
        idx = 0
        while x > bin_edge:
            idx += 1
            bin_edge = bin_edges[idx]

        return idx

    labels = list(map(assign_label, w8s))
    return labels


class Categorical(Dataset):

    # Takes in pd dataframe with 2ces and weights of sequences (key: "sequences", weights: "sequence_count")
    # Also used to calculate the independent fields for parameter fields initialization
    def __init__(self, dataset, q, weights=None, max_length=20, molecule='protein', device='cpu', one_hot=False, labels=False, additional_data=None):

        # Drop Duplicates/ Reset Index from most likely shuffled sequences
        # self.dataset = dataset.reset_index(drop=True).drop_duplicates("sequence")
        self.dataset = dataset.reset_index(drop=True)

        # dictionaries mapping One letter code to integer for all macro molecule types
        try:
            self.base_to_id = letter_to_int_dicts[molecule]
        except:
            print(f"Molecule {molecule} not supported. Please use protein, dna, or rna")

        self.n_bases = q

        self.device = device # Makes sure everything is on correct device

        self.max_length = max_length # Number of Visible Nodes
        self.oh = one_hot
        # self.train_labels = self.dataset.binary.to_numpy()
        self.total = len(self.dataset.index)
        self.seq_data = self.dataset.sequence.to_numpy()

        self.additional_data = None
        if additional_data:
            self.additional_data = additional_data

        if self.oh:
            self.train_data = self.one_hot(self.categorical(self.seq_data))
        else:
            self.train_data = self.categorical(self.seq_data)

        if weights is not None:
            if type(weights) is list:
                self.train_weights = np.asarray(weights)
            elif type(weights) is np.ndarray:
                self.train_weights = weights
        else:
            # all equally weighted
            self.train_weights = np.asarray([1. for x in range(self.total)])

        self.labels = labels
        if self.labels:
            self.train_labels = self.dataset.label.to_numpy()

    def __getitem__(self, index):

        # self.count += 1
        # if (self.count % self.dataset.shape[0] == 0):
        #     self.on_epoch_end()

        seq = self.seq_data[index]  # str of sequence
        model_input = self.train_data[index]  # either vector of integers for categorical or one hot vector
        weight = self.train_weights[index]

        return_arr = [index, seq, model_input, weight]
        if self.labels:
            label = self.train_labels[index]
            return_arr.append(label)
        if self.additional_data is not None:
            data = self.additional_data[index]
            return_arr.append(data)

        return return_arr

    def categorical(self, seq_dataset):
        return torch.tensor(list(map(lambda x: [self.base_to_id[y] for y in x], seq_dataset)), dtype=torch.long)

    def one_hot(self, cat_dataset):
        return F.one_hot(cat_dataset, num_classes=self.n_bases)

    def log_scale(self, weights):
        return np.asarray([math.log(x + 0.0001) for x in weights])

    def field_init(self):
        out = torch.zeros((self.max_length, self.n_bases), device=self.device)
        position_index = torch.arange(0, self.max_length, 1, device=self.device)
        if self.oh:
            cat_tensor = self.train_data.argmax(-1)
        else:
            cat_tensor = self.train_data
        for b in range(self.total):
            # out[position_index, cat_tensor[b]] += self.train_weights[b]
            out[position_index, cat_tensor[b]] += 1
        out.div_(self.total)  # in place

        # invert softmax
        eps = 1e-6
        fields = torch.log((1 - eps) * out + eps / self.n_bases)
        fields -= fields.sum(1).unsqueeze(1) / self.n_bases
        return fields

    def __len__(self):
        return self.train_data.shape[0]

    # def distance(self, MSA):
    #     B = MSA.shape[0]
    #     N = MSA.shape[1]
    #     distance = np.zeros([B, B])
    #     for b in range(B):
    #         distance[b] = ((MSA[b] != MSA).mean(1))
    #         distance[b, b] = 2.
    #     return distance
    #
    # def count_neighbours(self, MSA, threshold=0.1):  # Compute reweighting
    #     # works but is quite slow, should probably move this eventually
    #     # msa_long = MSA.long()
    #     B = MSA.shape[0]
    #     neighs = np.zeros((B,), dtype=float)
    #     for b in range(B):
    #         if self.oh:
    #             # pairwise_dist = torch.logical_and(MSA[b].unsqueeze(0))
    #             pairwise_dists = (MSA[b].unsqueeze(0) * MSA).sum(-1).sum(-1) / self.max_length
    #             neighs[b] = (pairwise_dists > threshold).float().sum().item()
    #         else:
    #             pairwise_dist = (MSA[b].unsqueeze(0) - MSA)
    #             dists = (pairwise_dist == 0).float().sum(1) / self.max_length
    #             neighs[b] = (dists > threshold).float().sum()
    #
    #         if b % 10000 == 0:
    #             print("Progress:", round(b/B, 3) * 100)
    #
    #     # N = MSA.shape[1]
    #     # num_neighbours = np.zeros(B)
    #     #
    #     # for b in range(B):
    #     #     num_neighbours[b] = 1 + ((MSA[b] != MSA).float().mean(1) < threshold).sum()
    #     return neighs



    # def on_epoch_end(self):
    #     self.count = 0
    #     if self.shuffle:
    #         self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)





import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

## CRBM only functions
def suggest_conv_size(input_shape, padding_max=3, dilation_max=4, stride_max=5):
    v_num, q = input_shape

    print(f"Finding Whole Convolutions for Input with {v_num} inputs:")
    # kernel size
    for i in range(1, v_num+1):
        kernel = [i, q]
        # padding
        for p in range(padding_max+1):
            padding = [p, 0]
            # dilation
            for d in range(1, dilation_max+1):
                dilation = [d, 1]
                # stride
                for s in range(1, stride_max+1):
                    stride = [s, 1]
                    # Convolution Output size
                    convx_num = int(math.floor((v_num + padding[0] * 2 - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1))
                    convy_num = int(math.floor((q + padding[1] * 2 - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1))
                    # Reconstruction Size
                    recon_x = (convx_num - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel[0] - 1) + 1
                    recon_y = (convy_num - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel[1] - 1) + 1
                    if recon_x == v_num:
                        print(f"Whole Convolution Found: Kernel: {kernel[0]}, Stride: {stride[0]}, Dilation: {dilation[0]}, Padding: {padding[0]}")
    return


# used by crbm to initialize weight sizes
def pool1d_dim(input_shape, pool_topology):
    [batch_size, h_number, convolutions] = input_shape
    stride = pool_topology["stride"]
    padding = pool_topology["padding"]
    kernel = pool_topology["kernel"]
    # dilation = pool_topology["dilation"]
    dilation = 1

    pool_out_num = int(math.floor((convolutions + padding * 2 - dilation * (kernel - 1) - 1) / stride + 1))

    # Copied from https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    un_pool_out_size = (pool_out_num - 1) * stride - 2 * padding + 1 * (kernel - 1) + 1

    # Pad for the unsampled convolutions (depends on stride, and tensor size)
    output_padding = convolutions - un_pool_out_size

    if output_padding != 0:
        print("Cannot create full reconstruction, please choose different pool topology")

    pool_out_size = (batch_size, h_number, pool_out_num)
    reconstruction_size = (batch_size, h_number, un_pool_out_size)

    return {"pool_size": pool_out_size, "reconstruction_shape": reconstruction_size}

# used by crbm to initialize weight sizes
def conv2d_dim(input_shape, conv_topology):
    [batch_size, input_channels, v_num, q] = input_shape
    if type(v_num) is tuple and q == 1:
        v_num, q = v_num # 2d visible
    stride = conv_topology["stride"]
    padding = conv_topology["padding"]
    kernel = conv_topology["kernel"]
    dilation = conv_topology["dilation"]
    h_num = conv_topology["number"]

    # Copied From https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    convx_num = int(math.floor((v_num + padding[0]*2 - dilation[0] * (kernel[0]-1) - 1)/stride[0] + 1))
    convy_num = int(math.floor((q + padding[1] * 2 - dilation[1] * (kernel[1]-1) - 1)/stride[1] + 1))  # most configurations will set this to 1

    # Copied from https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    recon_x = (convx_num - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel[0] - 1) + 1
    recon_y = (convy_num - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel[1] - 1) + 1

    # Pad for the unsampled visible units (depends on stride, and tensor size)
    output_padding = (v_num - recon_x, q - recon_y)

    # Size of Convolution Filters
    weight_size = (h_num, input_channels, kernel[0], kernel[1])

    # Size of Hidden unit Inputs h_uk
    conv_output_size = (batch_size, h_num, convx_num, convy_num)

    return {"weight_shape": weight_size, "conv_shape": conv_output_size, "output_padding": output_padding}

def conv1d_dim(input_shape, conv_topology):
    [batch_size, input_channels, v_num] = input_shape

    stride = conv_topology["stride"]
    padding = conv_topology["padding"]
    kernel = conv_topology["kernel"]
    dilation = conv_topology["dilation"]
    h_num = conv_topology["number"]

    # Copied From https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    convx_num = int(math.floor((v_num + padding*2 - dilation * (kernel-1) - 1)/stride + 1))

    # Copied from https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    recon_x = (convx_num - 1) * stride - 2 * padding + dilation * (kernel - 1) + 1

    # Pad for the unsampled visible units (depends on stride, and tensor size)
    output_padding = v_num - recon_x

    # Size of Convolution Filters
    weight_size = (h_num, input_channels, kernel)

    # Size of Hidden unit Inputs h_uk
    conv_output_size = (batch_size, h_num, convx_num)

    return {"weight_shape": weight_size, "conv_shape": conv_output_size, "output_padding": output_padding}


######### Data Reading Methods #########

# returns list of strings containing sequences
# optionally returns the affinities in the file found
# ex. with 5 as affinity
# >seq1-5
# ACGPTTACDKLLE

# Fasta File Reader
def fasta_read(fastafile, molecule, threads=1, drop_duplicates=False):
    """
    Parameters
    ----------
    fastafile: str,
        fasta file name and path
    molecule: str,
        type of data can be {"dna", "rna", or "protein"}
    threads: int, optional,
        number of cpu processes to use to read file
    drop_duplicates: bool,

    """
    o = open(fastafile)
    all_content = o.readlines()
    o.close()

    line_num = math.floor(len(all_content)/threads)
    # Which lines of file each process should read
    initial_bounds = [line_num*(i+1) for i in range(threads)]
    # initial_bounds = initial_bounds[:-1]
    initial_bounds.insert(0, 0)
    new_bounds = []
    for bound in initial_bounds[:-1]:
        idx = bound
        while not all_content[idx].startswith(">"):
            idx += 1
        new_bounds.append(idx)
    new_bounds.append(len(all_content))

    split_content = (all_content[new_bounds[xid]:new_bounds[xid+1]] for xid, x in enumerate(new_bounds[:-1]))

    p = Pool(threads)

    start = time.time()
    results = p.map(process_lines, split_content)
    end = time.time()

    print("Process Time", end-start)
    all_seqs, all_counts, all_chars = [], [], []
    for i in range(threads):
        all_seqs += results[i][0]
        all_counts += results[i][1]
        for char in results[i][2]:
            if char not in all_chars:
                all_chars.append(char)

    # Sometimes multiple characters mean the same thing, this code checks for that and adjusts q accordingly
    valuedict = {"protein": aadict, "dna": dnadict, "rna":rnadict}
    assert molecule in valuedict.keys()
    char_values = [valuedict[molecule][x] for x in all_chars]
    unique_char_values = list(set(char_values))

    q = len(unique_char_values)

    if drop_duplicates:
        if not all_counts:   # check if counts were found from fasta file
            all_counts = [1 for x in range(len(all_seqs))]
        assert len(all_seqs) == len(all_counts)
        df = pd.DataFrame({"sequence": all_seqs, "copy_num":all_counts})
        ndf = df.drop_duplicates(subset="sequence", keep="first")
        all_seqs = ndf.sequence.tolist()
        all_counts = ndf.copy_num.tolist()

    return all_seqs, all_counts, all_chars, q


# Worker for fasta_read
def process_lines(assigned_lines):
    titles, seqs, all_chars = [], [], []

    hdr_indices = []
    for lid, line in enumerate(assigned_lines):
        if line.startswith('>'):
            hdr_indices.append(lid)

    for hid, hdr in enumerate(hdr_indices):

        index = assigned_lines[hdr].find("-")  # first dash
        if index > -1:
            try:
                titles.append(float(assigned_lines[hdr][index+1:].rstrip()))
                # titles.append(float(assigned_lines[hdr].rstrip().split('-')[1]))
            except IndexError:
                pass

        if hid == len(hdr_indices) - 1:
            seq = "".join([line.rstrip() for line in assigned_lines[hdr + 1:]])
        else:
            seq = "".join([line.rstrip() for line in assigned_lines[hdr + 1: hdr_indices[hid+1]]])

        seqs.append(seq.upper())

    for seq in seqs:
        letters = set(list(seq))
        for l in letters:
            if l not in all_chars:
                all_chars.append(l)

    return seqs, titles, all_chars


def fasta_read_serial(fastafile, seq_read_counts=False, drop_duplicates=False, char_set=False, yield_q=False):
    o = open(fastafile)
    titles = []
    seqs = []
    all_chars = []
    for line in o:
        if line.startswith('>'):
            if seq_read_counts:
                titles.append(float(line.rstrip().split('-')[1]))
        else:
            seq = line.rstrip()
            letters = set(list(seq))
            for l in letters:
                if l not in all_chars:
                    all_chars.append(l)
            seqs.append(seq)
    o.close()
    if drop_duplicates:
        all_seqs = pd.DataFrame(seqs).drop_duplicates()
        seqs = all_seqs.values.tolist()
        seqs = [j for i in seqs for j in i]

    if seq_read_counts:
        return seqs, titles
    else:
        return seqs



######### Data Generation Methods #########
def extract_cluster_crbm_pool(model, hidden_indices):
    tmp_model = copy.deepcopy(model)
    if "relu" in tmp_model._get_name():
        param_keys = ["gamma", "theta", "W", "0gamma", "0theta"]
    else:
        param_keys = ["gamma+", "gamma-", "theta+", "theta-", "W",  "0gamma+", "0theta+", "0gamma-", "0theta-"]

    assert len(hidden_indices) == len(model.hidden_convolution_keys)
    for kid, key in enumerate(tmp_model.hidden_convolution_keys):
        for pkey in param_keys:
            setattr(tmp_model, f"{key}_{pkey}", torch.nn.Parameter(getattr(tmp_model, f"{key}_{pkey}")[hidden_indices[kid]], requires_grad=False))
        tmp_model.max_inds[kid] = (tmp_model.max_inds[kid])[hidden_indices[kid]]

    # edit keys
    keys_to_del = []
    for kid, key in enumerate(tmp_model.hidden_convolution_keys):
        new_hidden_number = len(hidden_indices[kid])
        if new_hidden_number == 0:
            keys_to_del.append(key)
        else:
            tmp_model.convolution_topology[key]["number"] = new_hidden_number
            wdims = list(tmp_model.convolution_topology[key]["weight_dims"])
            wdims[0] = new_hidden_number
            tmp_model.convolution_topology[key]["weight_dims"] = tuple(wdims)
            cdims = list(tmp_model.convolution_topology[key]["convolution_dims"])
            cdims[1] = new_hidden_number
            tmp_model.convolution_topology[key]["convolution_dims"] = tuple(cdims)

    for key in keys_to_del:
        tmp_model.convolution_topology.pop(key)
        tmp_model.hidden_convolution_keys.pop(key)

    return tmp_model

def gen_data_biased_ih(model, ih_means, ih_stds, samples=500):
    if type(ih_means) is not list:
        ih_means = [ih_means]
        ih_stds = [ih_stds]

    v_out = []
    for i in range(len(ih_means)):
        v_out.append(ih_means[i][None, :] + torch.randn((samples, 1)) * ih_stds[i][None, :])

    hs = model.sample_from_inputs_h(v_out)
    return model.sample_from_inputs_v(model.compute_output_h(hs))


def gen_data_biased_h(model, h_means, h_stds, samples=500):
    if type(h_means) is not list:
        h_means = [h_means]
        h_stds = [h_stds]

    sampled_h = []
    for i in range(len(h_means)):
        sampled_h.append(h_means[i][None, :] + torch.randn((samples, 1)) * h_stds[i][None, :])

    return model.sample_from_inputs_v(model.compute_output_h(sampled_h))




def gen_data_lowT(model, beta=1, which = 'marginal' ,Nchains=10, Lchains=100, Nthermalize=0, Nstep=1, N_PT=1, reshape=True, update_betas=False, config_init=[]):
    tmp_model = copy.deepcopy(model)
    name = tmp_model._get_name()
    if "CRBM" in name:
        setattr(tmp_model, "fields", torch.nn.Parameter(getattr(tmp_model, "fields") * beta, requires_grad=False))
        if "class" in name:
            setattr(tmp_model, "y_bias", torch.nn.Parameter(getattr(tmp_model, "y_bias") * beta, requires_grad=False))

        if which == 'joint':
            if "relu" in "name":
                param_keys = ["gamma", "theta", "W"]
            else:
                param_keys = ["gamma+", "gamma-", "theta+", "theta-", "W"]
            if "class" in name:
                param_keys.append("M")
            for key in tmp_model.hidden_convolution_keys:
                for pkey in param_keys:
                    setattr(tmp_model, f"{key}_{pkey}", torch.nn.Parameter(getattr(tmp_model, f"{key}_{pkey}") * beta, requires_grad=False))
        elif which == "marginal":
            if "relu" in "name":
                param_keys = ["gamma+", "gamma-", "theta+", "theta-", "W", "0gamma+", "0gamma-", "0theta+", "0theta-"]
            else:
                param_keys = ["gamma", "0gamma", "theta", "0theta", "W"]
            if "class" in name:
                param_keys.append("M")
            new_convolution_keys = copy.deepcopy(tmp_model.hidden_convolution_keys)

            # Setup Steps for editing the hidden layer topology of our model
            setattr(tmp_model, "convolution_topology", copy.deepcopy(model.convolution_topology))
            tmp_model_conv_topology = getattr(tmp_model, "convolution_topology")  # Get and edit tmp_model_conv_topology

            if "pool" in name:
                tmp_model.pools = tmp_model.pools * beta
                tmp_model.unpools = tmp_model.unpools * beta
            else:
                # Also need to fix up parameter hidden_layer_W
                tmp_model.register_parameter("hidden_layer_W", torch.nn.Parameter(getattr(tmp_model, "hidden_layer_W").repeat(beta), requires_grad=False))

            # Add keys for new layers, add entries to convolution_topology for new layers, and add parameters for new layers
            for key in tmp_model.hidden_convolution_keys:
                for b in range(beta - 1):
                    new_key = f"{key}_{b}"
                    new_convolution_keys.append(new_key)
                    tmp_model_conv_topology[f"{new_key}"] = copy.deepcopy(tmp_model_conv_topology[f"{key}"])

                    for pkey in param_keys:
                        new_param_key = f"{new_key}_{pkey}"
                        # setattr(tmp_model, new_param_key, torch.nn.Parameter(getattr(tmp_model, f"{key}_{pkey}"), requires_grad=False))
                        tmp_model.register_parameter(new_param_key, torch.nn.Parameter(getattr(tmp_model, f"{key}_{pkey}"), requires_grad=False))

            tmp_model.hidden_convolution_keys = new_convolution_keys
    elif "RBM" in name:
        with torch.no_grad():
            if which == 'joint':
                tmp_model.params["fields"] *= beta
                tmp_model.params["W_raw"] *= beta
                tmp_model.params["gamma+"] *= beta
                tmp_model.params["gamma-"] *= beta
                tmp_model.params["theta+"] *= beta
                tmp_model.params["theta-"] *= beta
            elif which == 'marginal':
                if type(beta) == int:
                    tmp_model.params["fields"] *= beta
                    tmp_model.params["W_raw"] = torch.nn.Parameter(torch.repeat_interleave(model.params["W_raw"], beta, dim=0), requires_grad=False)
                    tmp_model.params["gamma+"] = torch.nn.Parameter(torch.repeat_interleave(model.params["gamma+"], beta, dim=0), requires_grad=False)
                    tmp_model.params["gamma-"] = torch.nn.Parameter(torch.repeat_interleave(model.params["gamma-"], beta, dim=0), requires_grad=False)
                    tmp_model.params["theta+"] = torch.nn.Parameter(torch.repeat_interleave(model.params["theta+"], beta, dim=0), requires_grad=False)
                    tmp_model.params["theta-"] = torch.nn.Parameter(torch.repeat_interleave(model.params["theta-"], beta, dim=0), requires_grad=False)
                    tmp_model.params["0gamma+"] = torch.nn.Parameter(torch.repeat_interleave(model.params["0gamma+"], beta, dim=0), requires_grad=False)
                    tmp_model.params["0gamma-"] = torch.nn.Parameter(torch.repeat_interleave(model.params["0gamma-"], beta, dim=0), requires_grad=False)
                    tmp_model.params["0theta+"] = torch.nn.Parameter(torch.repeat_interleave(model.params["0theta+"], beta, dim=0), requires_grad=False)
                    tmp_model.params["0theta-"] = torch.nn.Parameter(torch.repeat_interleave(model.params["0theta-"], beta, dim=0), requires_grad=False)
            tmp_model.prep_W()

    return tmp_model.gen_data(Nchains=Nchains,Lchains=Lchains,Nthermalize=Nthermalize,Nstep=Nstep,N_PT=N_PT,reshape=reshape,update_betas=update_betas,config_init = config_init)

def gen_data_lowT_cluster(model, cluster_indx, beta=1, which = 'marginal' ,Nchains=10, Lchains=100, Nthermalize=0, Nstep=1, N_PT=1, reshape=True, update_betas=False, config_init=[]):
    tmp_model = copy.deepcopy(model)
    name = tmp_model._get_name()
    if "CRBM" in name:
        setattr(tmp_model, "fields", torch.nn.Parameter(getattr(tmp_model, "fields") * beta, requires_grad=False))
        if "class" in name:
            setattr(tmp_model, "y_bias", torch.nn.Parameter(getattr(tmp_model, "y_bias") * beta, requires_grad=False))

        if which == 'joint':
            param_keys = ["gamma+", "gamma-", "theta+", "theta-", "W"]
            if "class" in name:
                param_keys.append("M")
            for key in tmp_model.hidden_convolution_keys[cluster_indx]:
                for pkey in param_keys:
                    setattr(tmp_model, f"{key}_{pkey}", torch.nn.Parameter(getattr(tmp_model, f"{key}_{pkey}") * beta, requires_grad=False))
        elif which == "marginal":
            param_keys = ["gamma+", "gamma-", "theta+", "theta-", "W", "0gamma+", "0gamma-", "0theta+", "0theta-"]
            if "class" in name:
                param_keys.append("M")
            new_convolution_keys = copy.deepcopy(tmp_model.hidden_convolution_keys[cluster_indx])

            # Setup Steps for editing the hidden layer topology of our model
            setattr(tmp_model, "convolution_topology", copy.deepcopy(model.convolution_topology))
            tmp_model_conv_topology = getattr(tmp_model, "convolution_topology")  # Get and edit tmp_model_conv_topology

            if "pool" in name or "pcrbm" in name:
                tmp_model.pools = tmp_model.pools[cluster_indx] * beta
                tmp_model.unpools = tmp_model.unpools[cluster_indx] * beta
            else:
                # Also need to fix up parameter hidden_layer_W
                tmp_model.register_parameter("hidden_layer_W", torch.nn.Parameter(getattr(tmp_model, "hidden_layer_W").repeat(beta), requires_grad=False))

            # Add keys for new layers, add entries to convolution_topology for new layers, and add parameters for new layers
            for key in tmp_model.hidden_convolution_keys[cluster_indx]:
                for b in range(beta - 1):
                    new_key = f"{key}_{b}"
                    new_convolution_keys.append(new_key)
                    tmp_model_conv_topology[cluster_indx][f"{new_key}"] = copy.deepcopy(tmp_model_conv_topology[f"{key}"])

                    for pkey in param_keys:
                        new_param_key = f"{new_key}_{pkey}"
                        # setattr(tmp_model, new_param_key, torch.nn.Parameter(getattr(tmp_model, f"{key}_{pkey}"), requires_grad=False))
                        tmp_model.register_parameter(new_param_key, torch.nn.Parameter(getattr(tmp_model, f"{key}_{pkey}"), requires_grad=False))

            tmp_model.hidden_convolution_keys[cluster_indx] = new_convolution_keys


    return tmp_model.gen_data_cluster(cluster_indx, Nchains=Nchains,Lchains=Lchains,Nthermalize=Nthermalize,Nstep=Nstep,N_PT=N_PT,reshape=reshape,update_betas=update_betas,config_init = config_init)


def gen_data_zeroT(model, which = 'marginal' ,Nchains=10,Lchains=100,Nthermalize=0,Nstep=1,N_PT=1,reshape=True,update_betas=False,config_init=[]):
    tmp_model = copy.deepcopy(model)
    if "class" in tmp_model._get_name():
        print("Zero Temp Generation of Data not available for classification CRBM")
    with torch.no_grad():
        if which == 'joint':
            tmp_model.markov_step = types.MethodType(markov_step_zeroT_joint, tmp_model)
        elif which == 'marginal':
            tmp_model.markov_step = types.MethodType(markov_step_zeroT_marginal, tmp_model)
        return tmp_model.gen_data(Nchains=Nchains,Lchains=Lchains,Nthermalize=Nthermalize,Nstep=Nstep,N_PT=N_PT,reshape=reshape,update_betas=update_betas,config_init = config_init)

def markov_step_zeroT_joint(self, v, beta=1):
    I = self.compute_output_v(v)
    h = self.transform_h(I)
    I = self.compute_output_h(h)
    nv = self.transform_v(I)
    return nv, h

def markov_step_zeroT_marginal(self, v,beta=1):
    I = self.compute_output_v(v)
    h = self.mean_h(I)
    I = self.compute_output_h(h)
    nv = self.transform_v(I)
    return nv, h


##### Other model utility functions

# Get Model from checkpoint File with specified version and directory
# def get_checkpoint(version, dir=""):
#     checkpoint_dir = dir + "/version_" + str(version) + "/checkpoints/"
#
#     for file in os.listdir(checkpoint_dir):
#         if file.endswith(".ckpt"):
#             checkpoint_file = os.path.join(checkpoint_dir, file)
#     return checkpoint_file

def get_beta_and_W(model, hidden_key=None, include_gaps=False, separate_signs=False):
    name = model._get_name()
    if "CRBM" in name or "crbm" in name:
        if hidden_key is None:
            print("Must specify hidden key in get_beta_and_W for crbm")
            exit(-1)
        else:
            W = model.get_param(hidden_key + "_W").squeeze(1)
            if separate_signs:
                Wpos = np.maximum(W, 0)
                Wneg = np.minimum(W, 0)
                if include_gaps:
                    return np.sqrt((Wpos ** 2).sum(-1).sum(-1)), Wpos, np.sqrt((Wneg ** 2).sum(-1).sum(-1)), Wneg
                else:
                    return np.sqrt((Wpos[:, :, :-1] ** 2).sum(-1).sum(-1)), Wpos, np.sqrt((Wneg[:, :, :-1] ** 2).sum(-1).sum(-1)), Wneg
            else:
                if include_gaps:
                    return np.sqrt((W ** 2).sum(-1).sum(-1)), W
                else:
                    return np.sqrt((W[:, :, :-1] ** 2).sum(-1).sum(-1)), W
    elif "RBM" in name:
        W = model.get_param("W")
        if include_gaps:
            return np.sqrt((W ** 2).sum(-1).sum(-1)), W
        else:
            return np.sqrt((W[:, :, :-1] ** 2).sum(-1).sum(-1)), W

def all_weights(model, name=None, rows=5, order_weights=True):
    model_name = model._get_name()
    if name is None:
        name = model._get_name()

    if "CRBM" in model_name or "crbm" in model_name:
        if "cluster" in model_name:
            for cluster_indx in range(model.clusters):
                for key in model.hidden_convolution_keys[cluster_indx]:
                    wdim = model.convolution_topology[cluster_indx][key]["weight_dims"]
                    kernelx = wdim[2]
                    if kernelx <= 10:
                        ncols = 2
                    else:
                        ncols = 1
                    conv_weights(model, key, f"{name}_{key}" + key, rows, ncols, 7, 5, order_weights=order_weights)
        else:
            for key in model.hidden_convolution_keys:
                wdim = model.convolution_topology[key]["weight_dims"]
                kernelx = wdim[2]
                if kernelx <= 10:
                    ncols = 2
                else:
                    ncols = 1
                conv_weights(model, key, name + "_" + key, rows, ncols, 7, 5, order_weights=order_weights)
    elif "RBM" in model_name:
        beta, W = get_beta_and_W(model)
        if order_weights:
            order = np.argsort(beta)[::-1]
        else:
            order = np.arange(0, beta.shape[0], 1)
        wdim = W.shape[1]
        if wdim <= 20:
            ncols = 2
        else:
            ncols = 1
        fig = Sequence_logo_all(W[order], data_type="weights", name=name + '.pdf', nrows=rows, ncols=ncols, figsize=(7, 5), ticks_every=10, ticks_labels_size=10, title_size=12, dpi=200, molecule=model.molecule)

    plt.close() # close all open figures


def conv_weights(crbm, hidden_key, name, rows, columns, h, w, order_weights=True):
    beta, W = get_beta_and_W(crbm, hidden_key)
    if order_weights:
        order = np.argsort(beta)[::-1]
    else:
        order = np.arange(0, beta.shape[0], 1)
    fig = Sequence_logo_all(W[order], data_type="weights", name=name + '.pdf', nrows=rows, ncols=columns, figsize=(h,w) ,ticks_every=5,ticks_labels_size=10,title_size=12, dpi=200, molecule=crbm.molecule)
