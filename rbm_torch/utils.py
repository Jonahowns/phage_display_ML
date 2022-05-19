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
from matplotlib.ticker import FormatStrFormatter
import copy
from PIL import Image
import math
import time
from multiprocessing import Pool
import pandas as pd
import types
from torch.utils.data import Dataset

# Globals used for Converting Sequence Strings to Integers
aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
aalower = [x.lower() for x in aa]
aadictU = {aa[k]: k for k in range(len(aa))}
aadictL = {aalower[k]:k for k in range(len(aalower))}
aadict = {**aadictU, **aadictL}

dna = ['A', 'C', 'G', 'T', '-']
dnalower = [x.lower() for x in dna]
dnadictU = {dna[k]: k for k in range(len(dna))}
dnadictL = {dnalower[k]: k for k in range(len(dnalower))}
dnadict = {**dnadictU, **dnadictL}

rna = ['A', 'C', 'G', 'U', '-']
rnalower = [x.lower() for x in rna]
rnadictU = {rna[k]: k for k in range(len(rna))}
rnadictL = {rnalower[k]: k for k in range(len(rnalower))}
rnadict = {**rnadictU, **rnadictL}

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


class Categorical(Dataset):

    # Takes in pd dataframe with sequences and weights of sequences (key: "sequences", weights: "sequence_count")
    # Also used to calculate the independent fields for parameter fields initialization
    def __init__(self, dataset, q, weights=None, max_length=20, shuffle=True, base_to_id='protein', device='cpu', one_hot=False, neighbor_threshold=None):

        # Drop Duplicates/ Reset Index from most likely shuffled sequences
        # self.dataset = dataset.reset_index(drop=True).drop_duplicates("sequence")
        self.dataset = dataset.reset_index(drop=True)

        self.shuffle = shuffle
        self.on_epoch_end()

        # dictionaries mapping One letter code to integer for all macro molecule types
        if base_to_id == 'protein':
            self.base_to_id = aadict
        elif base_to_id == 'dna':
            self.base_to_id = dnadict
        elif base_to_id == 'rna':
            self.base_to_id = rnadict
        self.n_bases = q

        self.device = device # Makes sure everything is on correct device

        self.max_length = max_length # Number of Visible Nodes
        self.oh = one_hot
        # self.train_labels = self.dataset.binary.to_numpy()
        self.total = len(self.dataset.index)
        self.seq_data = self.dataset.sequence.to_numpy()
        self.train_data = self.categorical(self.seq_data)
        if self.oh:
            self.train_oh = F.one_hot(self.train_data, q)

        if neighbor_threshold is not None:
            neighs = self.count_neighbours(self.train_data, threshold=neighbor_threshold)
            self.train_weights = 1./neighs
        elif weights is not None:
            if len(self.train_data) != len(weights):
                print("Provided Weights are not the correct length")
                exit(1)
            self.train_weights = np.asarray(weights)
            self.train_weights /= self.train_weights.sum()
        else:
            # all equally weighted
            self.train_weights = 1./np.asarray([1. for x in range(self.total)])

    def __getitem__(self, index):

        self.count += 1
        if (self.count % self.dataset.shape[0] == 0):
            self.on_epoch_end()

        seq = self.seq_data[index]
        cat_seq = self.train_data[index]
        weight = self.train_weights[index]

        if self.oh:
            one_hot = self.train_oh[index]
            return seq, cat_seq, one_hot, weight
        else:
            return seq, cat_seq, weight

    def categorical(self, seq_dataset):
        return torch.tensor(list(map(lambda x: [self.base_to_id[y] for y in x], seq_dataset)), dtype=torch.long)

    def one_hot(self, cat_dataset):
        one_hot_vector = F.one_hot(cat_dataset, num_classes=self.n_bases)
        return one_hot_vector

    # verified to work exactly as done in tubiana's implementation
    def field_init(self):
        out = torch.zeros((self.max_length, self.n_bases), device=self.device)
        position_index = torch.arange(0, self.max_length, 1, device=self.device)
        for b in range(self.total):
            out[position_index, self.train_data[b]] += self.train_weights[b]
        out.div_(self.total)  # in place

        # invert softmax
        eps = 1e-6
        fields = torch.log((1 - eps) * out + eps / self.n_bases)
        fields -= fields.sum(1).unsqueeze(1) / self.n_bases
        return fields

    def distance(MSA):
        B = MSA.shape[0]
        N = MSA.shape[1]
        distance = np.zeros([B, B])
        for b in range(B):
            distance[b] = ((MSA[b] != MSA).mean(1))
            distance[b, b] = 2.
        return distance

    def count_neighbours(MSA, threshold=0.1):  # Compute reweighting
        B = MSA.shape[0]
        N = MSA.shape[1]
        num_neighbours = np.zeros(B)
        for b in range(B):
            num_neighbours[b] = ((MSA[b] != MSA).mean(1) < threshold).sum()
        return num_neighbours

    def __len__(self):
        return self.train_data.shape[0]

    def on_epoch_end(self):
        self.count = 0
        if self.shuffle:
            self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)



######### Data Reading Methods #########

# returns list of strings containing sequences
# optionally returns the affinities in the file found
# ex. with 5 as affinity
# >seq1-5
# ACGPTTACDKLLE
# Fasta File Reader
def fasta_read(fastafile, molecule, threads=1, drop_duplicates=False):
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


def process_lines(assigned_lines):
    titles, seqs, all_chars = [], [], []

    hdr_indices = []
    for lid, line in enumerate(assigned_lines):
        if line.startswith('>'):
            hdr_indices.append(lid)

    for hid, hdr in enumerate(hdr_indices):
        try:
            titles.append(float(assigned_lines[hdr].rstrip().split('-')[1]))
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

def gen_data_lowT(model, beta=1, which = 'marginal' ,Nchains=10, Lchains=100, Nthermalize=0, Nstep=1, N_PT=1, reshape=True, update_betas=False, config_init=[]):
    tmp_model = copy.deepcopy(model)
    if tmp_model._get_name() == "RBM":
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

    elif tmp_model._get_name() == "CRBM":
        setattr(tmp_model, "fields", torch.nn.Parameter(getattr(tmp_model, "fields") * beta, requires_grad=False))

        if which == 'joint':
            param_keys = ["gamma+", "gamma-", "theta+", "theta-", "W"]
            for key in tmp_model.hidden_convolution_keys:
                for pkey in param_keys:
                    setattr(tmp_model, f"{key}_{pkey}", torch.nn.Parameter(getattr(tmp_model, f"{key}_{pkey}")*beta, requires_grad=False))
        elif which == "marginal":
            param_keys = ["gamma+", "gamma-", "theta+", "theta-", "W", "0gamma+", "0gamma-", "0theta+", "0theta-"]
            for key in tmp_model.hidden_convolution_keys:
                for pkey in param_keys:
                    setattr(tmp_model, f"{key}_{pkey}", torch.nn.Parameter(torch.repeat_interleave(getattr(tmp_model, f"{key}_{pkey}"), beta, dim=0), requires_grad=False))

    return tmp_model.gen_data(Nchains=Nchains,Lchains=Lchains,Nthermalize=Nthermalize,Nstep=Nstep,N_PT=N_PT,reshape=reshape,update_betas=update_betas,config_init = config_init)

def gen_data_zeroT(RBM, which = 'marginal' ,Nchains=10,Lchains=100,Nthermalize=0,Nstep=1,N_PT=1,reshape=True,update_betas=False,config_init=[]):
    tmp_RBM = copy.deepcopy(RBM)
    with torch.no_grad():
        if which == 'joint':
            tmp_RBM.markov_step = types.MethodType(markov_step_zeroT_joint, tmp_RBM)
        elif which == 'marginal':
            tmp_RBM.markov_step = types.MethodType(markov_step_zeroT_marginal, tmp_RBM)
        return tmp_RBM.gen_data(Nchains=Nchains,Lchains=Lchains,Nthermalize=Nthermalize,Nstep=Nstep,N_PT=N_PT,reshape=reshape,update_betas=update_betas,config_init = config_init)

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


## Implementation inspired from https://stackoverflow.com/questions/42615527/sequence-logos-in-matplotlib-aligning-xticks
## Color choice inspired from: http://weblogo.threeplusone.com/manual.html

def clean_ax(ax):
    ax.axis("off")

def get_ax(ax, i, nrows, ncols):
    if (ncols > 1) & (nrows > 1):
        col = int(i % ncols)
        row = int(i / ncols)
        ax_ = ax[row, col]
    elif (ncols > 1) & (nrows == 1):
        ax_ = ax[i]
    elif (ncols == 1) & (nrows > 1):
        ax_ = ax[i]
    else:
        ax_ = ax
    return ax_


def select_sites(W, window=5, theta_important=0.25):
    n_sites = W.shape[0]
    norm = np.abs(W).sum(-1)
    important = np.nonzero(norm / norm.max() > theta_important)[0]
    selected = []
    for imp in important:
        selected += range(max(0, imp - window), min(imp + window + 1, n_sites))
    selected = np.unique(selected)
    return selected


def ticksAt(selected, ticks_every=10):
    n_selected = len(selected)
    all_ticks = []
    all_ticks_labels = []
    previous_select = selected[0]
    k = 0
    for select in selected:
        if (select - previous_select) > 1:
            k += 1
        if (select % ticks_every == 0) | ((select - previous_select) > 1):
            if not k in all_ticks:
                all_ticks.append(k + 1)
                all_ticks_labels.append(select + 1)
        previous_select = copy.copy(select)
        k += 1
    return np.array(all_ticks), np.array(all_ticks_labels)


def breaksAt(x, maxi_size, ax):
    ax.plot([x, x], [-maxi_size, maxi_size], linewidth=5, c='black', linestyle='--')


def letterAt(letter, x, y, yscale=1, ax=None, type='protein'):
    if type == 'protein':
        text = LETTERS[letter]
    if type == 'dna':
        text = LETTERSdna[aa_to_dna[letter]]
    if type == 'rna':
        text = LETTERSrna[aa_to_rna[letter]]
    t = mpl.transforms.Affine2D().scale(1 * globscale, yscale * globscale) + \
        mpl.transforms.Affine2D().translate(x, y) + ax.transData
    p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter], transform=t)
    if ax != None:
        ax.add_artist(p)
    return p


def aa_color(letter):
    if letter in ['C']:
        return 'green'
    elif letter in ['F', 'W', 'Y']:
        return [199 / 256., 182 / 256., 0., 1.]  # 'gold'
    elif letter in ['Q', 'N', 'S', 'T']:
        return 'purple'
    elif letter in ['V', 'L', 'I', 'M']:
        return 'black'
    elif letter in ['K', 'R', 'H']:
        return 'blue'
    elif letter in ['D', 'E']:
        return 'red'
    elif letter in ['A', 'P', 'G']:
        return 'grey'
    elif letter in ['$\\boxminus$']:
        return 'black'
    else:
        return 'black'


def build_scores(matrix, epsilon=1e-4):
    n_sites = matrix.shape[0]
    n_colors = matrix.shape[1]
    all_scores = []
    for site in range(n_sites):
        conservation = np.log2(21) + (np.log2(matrix[site] + epsilon) * matrix[site]).sum()
        liste = []
        order_colors = np.argsort(matrix[site])
        for c in order_colors:
            liste.append((list_aa[c], matrix[site, c] * conservation))
        all_scores.append(liste)
    return all_scores


def build_scores2(matrix):
    n_sites = matrix.shape[0]
    n_colors = matrix.shape[1]
    epsilon = 1e-4
    all_scores = []
    for site in range(n_sites):
        liste = []
        c_pos = np.nonzero(matrix[site] >= 0)[0]
        c_neg = np.nonzero(matrix[site] < 0)[0]

        order_colors_pos = c_pos[np.argsort(matrix[site][c_pos])]
        order_colors_neg = c_neg[np.argsort(-matrix[site][c_neg])]
        for c in order_colors_pos:
            liste.append((list_aa[c], matrix[site, c], '+'))
        for c in order_colors_neg:
            liste.append((list_aa[c], -matrix[site, c], '-'))
        all_scores.append(liste)
    return all_scores


def build_scores_break(matrix, selected, epsilon=1e-4):
    has_breaks = (selected[1:] - selected[:-1]) > 1
    has_breaks = np.concatenate((np.zeros(1), has_breaks), axis=0)
    n_sites = len(selected)
    n_colors = matrix.shape[1]

    epsilon = 1e-4
    all_scores = []
    maxi_size = 0
    for site, has_break in zip(selected, has_breaks):
        if has_break:
            all_scores.append([('BREAK', 'BREAK', 'BREAK')])
        #             all_scores.append([('BREAK','BREAK','BREAK')] )
        conservation = np.log2(21) + (np.log2(matrix[site] + epsilon) * matrix[site]).sum()
        liste = []
        order_colors = np.argsort(matrix[site])
        for c in order_colors:
            liste.append((list_aa[c], matrix[site, c] * conservation))
        maxi_size = max(maxi_size, conservation)
        all_scores.append(liste)
    return all_scores, maxi_size


def build_scores2_break(matrix, selected):
    has_breaks = (selected[1:] - selected[:-1]) > 1
    has_breaks = np.concatenate((np.zeros(1), has_breaks), axis=0)
    n_sites = len(selected)
    n_colors = matrix.shape[1]

    epsilon = 1e-4
    all_scores = []
    for site, has_break in zip(selected, has_breaks):
        if has_break:
            all_scores.append([('BREAK', 'BREAK', 'BREAK')])
        liste = []
        c_pos = np.nonzero(matrix[site] >= 0)[0]
        c_neg = np.nonzero(matrix[site] < 0)[0]

        order_colors_pos = c_pos[np.argsort(matrix[site][c_pos])]
        order_colors_neg = c_neg[np.argsort(-matrix[site][c_neg])]
        for c in order_colors_pos:
            liste.append((list_aa[c], matrix[site, c], '+'))
        for c in order_colors_neg:
            liste.append((list_aa[c], -matrix[site, c], '-'))
        all_scores.append(liste)
    maxi_size = np.abs(matrix).sum(-1).max()
    return all_scores, maxi_size

# Needed to generate Sequence Logos

fp = FontProperties(family="Arial", weight="bold")
globscale = 1.35
aa_to_dna = {'A': 'A', 'C': 'C', 'D': 'G', 'E': 'T', 'F': '$\\boxminus$'}
aa_to_rna = {'A': 'A', 'C': 'C', 'D': 'G', 'E': 'U', 'F': '$\\boxminus$'}

dna = ['A', 'C', 'G', 'T', '$\\boxminus$']
rna = ['A', 'C', 'G', 'U', '$\\boxminus$']
list_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '$\\boxminus$']

LETTERS = dict([(letter, TextPath((-0.30, 0), letter, size=1, prop=fp)) for letter in list_aa])
LETTERSdna = dict([(letter, TextPath((-0.30, 0), letter, size=1, prop=fp)) for letter in dna])
LETTERSrna = dict([(letter, TextPath((-0.30, 0), letter, size=1, prop=fp)) for letter in rna])
COLOR_SCHEME = dict([(letter, aa_color(letter)) for letter in list_aa])


def Sequence_logo(matrix, ax=None, data_type=None, figsize=None, ylabel=None, title=None, epsilon=1e-4, show=True, ticks_every=1, ticks_labels_size=14, title_size=20, molecule='protein'):
    if data_type is None:
        if matrix.min() >= 0:
            data_type = 'mean'
        else:
            data_type = 'weights'

    if data_type == 'mean':
        all_scores = build_scores(matrix, epsilon=epsilon)
    elif data_type == 'weights':
        all_scores = build_scores2(matrix)
    else:
        print('data type not understood')
        return -1

    if ax is not None:
        show = False
        return_fig = False
    else:
        if figsize is None:
            figsize = (max(int(0.3 * matrix.shape[0]), 2), 3)
        fig, ax = plt.subplots(figsize=figsize)
        return_fig = True

    x = 1
    maxi = 0
    mini = 0
    for scores in all_scores:
        if data_type == 'mean':
            y = 0
            for base, score in scores:
                if score > 0.01:
                    letterAt(base, x, y, score, ax, type=molecule)
                y += score
            x += 1
            maxi = max(maxi, y)


        elif data_type == 'weights':
            y_pos = 0
            y_neg = 0
            for base, score, sign in scores:
                if sign == '+':
                    letterAt(base, x, y_pos, score, ax, molecule)
                    y_pos += score
                else:
                    y_neg += score
                    letterAt(base, x, -y_neg, score, ax, molecule)
            x += 1
            maxi = max(y_pos, maxi)
            mini = min(-y_neg, mini)

    if data_type == 'weights':
        maxi = max(maxi, abs(mini))
        mini = -maxi

    if ticks_every > 1:
        xticks = range(1, x)
        xtickslabels = ['%s' % k if k % ticks_every == 0 else '' for k in xticks]
        ax.set_xticks(xticks, xtickslabels)
    else:
        ax.set_xticks(range(1, x))
    ax.set_xlim((0, x))
    ax.set_ylim((mini, maxi))
    if ylabel is None:
        if data_type == 'mean':
            ylabel = 'Conservation (bits)'
        elif data_type == 'weights':
            ylabel = 'Weights'
    ax.set_ylabel(ylabel, fontsize=title_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', which='major', labelsize=ticks_labels_size)
    ax.tick_params(axis='both', which='minor', labelsize=ticks_labels_size)
    if title is not None:
        ax.set_title(title, fontsize=title_size)
    if return_fig:
        plt.tight_layout()
        if show:
            plt.show()
        return fig


def Sequence_logo_breaks(matrix, data_type=None, selected=None, window=5, theta_important=0.25, figsize=None, nrows=1, ylabel=None, title=None, epsilon=1e-4, show=True, ticks_every=5, ticks_labels_size=14, title_size=20, molecule='protein'):
    if data_type is None:
        if matrix.min() >= 0:
            data_type = 'mean'
        else:
            data_type = 'weights'

    if selected is None:
        if data_type == 'mean':
            'NO SELECTION SUPPORTED FOR MEAN VECTOR'
            return
        else:
            selected = select_sites(matrix, window=window, theta_important=theta_important)
    else:
        selected = np.array(selected)
    print('Number of sites selected: %s' % len(selected))

    xticks, xticks_labels = ticksAt(selected, ticks_every=ticks_every)

    if data_type == 'mean':
        all_scores, maxi_size = build_scores_break(matrix, selected, epsilon=epsilon)
    elif data_type == 'weights':
        all_scores, maxi_size = build_scores2_break(matrix, selected)
    else:
        print('data type not understood')
        return -1

    nbreaks = ((selected[1:] - selected[:-1]) > 1).sum()
    width = (len(selected) + nbreaks) / nrows

    if figsize is None:
        figsize = (max(int(0.3 * width), 2), 3 * nrows)

    fig, ax = plt.subplots(figsize=figsize, nrows=nrows)
    if nrows > 1:
        for row in range(nrows):
            ax[row].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    x = 1
    maxi = 0
    mini = 0

    if nrows > 1:
        row = 0
        ax_ = ax[row]
        xmins = np.zeros(nrows)
        xmaxs = np.ones(nrows) * (len(selected) + nbreaks + 1)
    else:
        ax_ = ax

    for scores in all_scores:
        if data_type == 'mean':
            y = 0
            if scores[0][0] == 'BREAK':
                breaksAt(x, maxi_size, ax_)
                if nrows > 1:
                    if x > (1 + row) * width:
                        xmaxs[row] = copy.copy(x) + 1
                        xmins[row + 1] = copy.copy(x)
                        row += 1
                        ax_ = ax[row]
            else:
                for base, score in scores:
                    if score > 0.01:
                        letterAt(base, x, y, score, ax_, type=molecule)
                    y += score
                x += 1
                maxi = max(maxi, y)


        elif data_type == 'weights':
            y_pos = 0
            y_neg = 0
            if scores[0][0] == 'BREAK':
                breaksAt(x, maxi_size, ax_)
                if nrows > 1:
                    if x > (1 + row) * width:
                        xmaxs[row] = copy.copy(x) + 1
                        xmins[row + 1] = copy.copy(x)
                        row += 1
                        ax_ = ax[row]

            else:
                for base, score, sign in scores:
                    if sign == '+':
                        letterAt(base, x, y_pos, score, ax_, type=molecule)
                        y_pos += score
                    else:
                        y_neg += score
                        letterAt(base, x, -y_neg, score, ax_, type=molecule)
            x += 1
            maxi = max(y_pos, maxi)
            mini = min(-y_neg, mini)

    if data_type == 'weights':
        maxi = max(maxi, abs(mini))
        mini = -maxi

    if nrows > 1:
        for row in range(nrows):
            ax[row].set_xlim((xmins[row], xmaxs[row]))
            ax[row].set_ylim((mini, maxi))
            subset = (xticks > xmins[row]) & (xticks < xmaxs[row])
            ax[row].set_xticks(xticks[subset])
            ax[row].set_xticklabels(xticks_labels[subset])
    else:
        plt.xticks(xticks, xticks_labels)
        plt.xlim((0, x))
        plt.ylim((mini, maxi))

    if ylabel is None:
        if data_type == 'mean':
            ylabel = 'Conservation (bits)'
        elif data_type == 'weights':
            ylabel = 'Weights'
    if nrows > 1:
        ax[0].set_ylabel(ylabel, fontsize=title_size)
        for row in range(1, nrows):
            ax[row].set_ylabel('. . .', fontsize=title_size)
    else:
        ax.set_ylabel(ylabel, fontsize=title_size)

    if nrows > 1:
        for k in range(nrows):
            ax[k].spines['right'].set_visible(False)
            ax[k].spines['top'].set_visible(False)
            ax[k].yaxis.set_ticks_position('left')
            ax[k].xaxis.set_ticks_position('bottom')
            ax[k].tick_params(axis='both', which='major', labelsize=ticks_labels_size)
            ax[k].tick_params(axis='both', which='minor', labelsize=ticks_labels_size)

    else:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.tick_params(axis='both', which='major', labelsize=ticks_labels_size)
        ax.tick_params(axis='both', which='minor', labelsize=ticks_labels_size)

    if title is not None:
        if nrows > 1:
            ax[0].set_title(title, fontsize=title_size)
        else:
            ax.set_title(title, fontsize=title_size)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, selected


def Sequence_logo_multiple(matrix, data_type=None, figsize=None, ylabel=None, title=None, epsilon=1e-4, ncols=1, show=True, count_from=0, ticks_every=1, ticks_labels_size=14, title_size=20, molecule='protein'):
    if data_type is None:
        if matrix.min() >= 0:
            data_type = 'mean'
        else:
            data_type = 'weights'

    N_plots = matrix.shape[0]
    nrows = int(np.ceil(N_plots / float(ncols)))

    if figsize is None:
        figsize = (max(int(0.3 * matrix.shape[1]), 2), 3)

    figsize = (figsize[0] * ncols, figsize[1] * nrows)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    if ylabel is None:
        if data_type == 'mean':
            ylabel = 'Conservation (bits)'
        elif data_type == 'weights':
            ylabel = 'Weights'
    if type(ylabel) == str:
        ylabels = [ylabel + ' #%s' % i for i in range(1 + count_from, N_plots + count_from + 1)]
    else:
        ylabels = ylabel

    if title is None:
        title = ''
    if type(title) == str:
        titles = [title for _ in range(N_plots)]
    else:
        titles = title

    for i in range(N_plots):
        ax_ = get_ax(ax, i, nrows, ncols)

        Sequence_logo(matrix[i], ax=ax_, data_type=data_type, ylabel=ylabels[i], title=titles[i],
                      epsilon=epsilon, show=False, ticks_every=ticks_every, ticks_labels_size=ticks_labels_size, title_size=title_size, molecule=molecule)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def Sequence_logo_all(matrix, name='all_Sequence_logo.pdf', nrows=5, ncols=2, data_type=None, figsize=None, ylabel=None, title=None, epsilon=1e-4, ticks_every=5, ticks_labels_size=14, title_size=20, dpi=100, molecule='protein'):
    if data_type is None:
        if matrix.min() >= 0:
            data_type = 'mean'
        else:
            data_type = 'weights'
    n_plots = matrix.shape[0]
    plots_per_page = nrows * ncols
    n_pages = int(np.ceil(n_plots / float(plots_per_page)))
    rng = np.random.randn(1)[0]  # avoid file conflicts in case of multiple threads.
    mini_name = name[:-4]
    images = []
    for i in range(n_pages):
        if type(ylabel) == list:
            ylabel_ = ylabel[i * plots_per_page:min(plots_per_page * (i + 1), n_plots)]
        else:
            ylabel_ = ylabel
        if type(title) == list:
            title_ = title[i * plots_per_page:min(plots_per_page * (i + 1), n_plots)]
        else:
            title_ = title
        fig = Sequence_logo_multiple(matrix[plots_per_page * i:min(plots_per_page * (i + 1), n_plots)], data_type=data_type, figsize=figsize, ylabel=ylabel_, title=title_, epsilon=epsilon, ncols=ncols, show=False, count_from=plots_per_page * i, ticks_every=ticks_every,
                                     ticks_labels_size=ticks_labels_size, title_size=title_size, molecule=molecule)
        file = f"tmp_{rng}_#{i}.jpg"
        fig.savefig(mini_name + file, dpi=dpi)
        fig.clear()
        images.append(Image.open(mini_name + file))
        command = 'rm ' + mini_name + file
        os.system(command)

    images[0].save(name, "PDF", resolution=100.0, save_all=True, append_images=images[1:])
    return 'done'
