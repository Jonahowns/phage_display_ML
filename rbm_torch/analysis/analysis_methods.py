import sys
sys.path.append("../")
from rbm import fasta_read
import rbm_utils

import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess as sp
import numpy as np
import torch


int_to_letter_dicts = {"protein": rbm_utils.aadict, "dna": rbm_utils.dnadict, "rna": rbm_utils.rnadict}

# Methods for generating plots etc.
def assign(x):
    if x < 2:
        return "Low"
    elif x < 10:
        return "Mid"
    else:
        return "High"


# Helper Functions for loading data and loading RBMs not in our current directory
def fetch_data(fasta_names, dir="", counts=False):
    for xid, x in enumerate(fasta_names):
        seqs, counts = fasta_read(dir + "/" + x + ".fasta", drop_duplicates=True, seq_read_counts=True)
        round_label = [x for i in range(len(seqs))]
        assignment = [assign(i) for i in counts]
        if xid == 0:
            data_df = pd.DataFrame({"sequence": seqs, "copynum": counts, "round": round_label, "assignment": assignment})
        else:
            data_df = pd.concat([data_df, pd.DataFrame({"sequence": seqs, "copynum": counts, "round": round_label, "assignment": assignment})])

    return data_df


def get_checkpoint_path(round, version=None, rbmdir=""):
    ndir = rbmdir + round + "/"
    if version:
        version_dir = ndir + f"version_{version}/"
    else:   # Get Most recent i.e. highest version number
        v_dirs = glob(ndir + "/*/", recursive=True)
        versions = [int(x[:-1].rsplit("_")[-1]) for x in v_dirs]  # extracted version numbers

        maxv = max(versions)  # get highest version number
        indexofinterest = versions.index(maxv)  # Get index of the highest version
        version_dir = v_dirs[indexofinterest]  # Access directory path of the highest version

    y = glob(version_dir + "checkpoints/*.ckpt", recursive=False)[0]
    return y, version_dir


# Returns dictionary of arrays of likelihoods
def generate_likelihoods(rounds, RBM, all_data):
    likelihoods = {}
    for x in rounds:
        seqs, likeli = RBM.predict(all_data[all_data["round"] == x])
        likelihoods[x] = likeli
    return likelihoods


# Plot Likelihoods as kde curves with each round in a new row
def plot_likelihoods(likeli, title, xaxislabel, order, labels, colors, clip=None):
    plot_num = len(likeli.keys())
    fig, axs = plt.subplots(plot_num, 1, sharex=True, sharey=False)
    for xid, x in enumerate(order):
        if clip is not None:
            y = sns.kdeplot(likeli[x], shade=False, alpha=0.5, color=colors[xid], ax=axs[xid], label=labels[xid], clip=clip)
        else:
            y = sns.kdeplot(likeli[x], shade=False, alpha=0.5, color=colors[xid], ax=axs[xid], label=labels[xid])
        if xid == len(order) - 1:
            y.set(xlabel=xaxislabel)
        axs[xid].legend()
    fig.suptitle(title)
    plt.show()


## Distribution of Counts inside each experimental dataset
def count_dist(data_w_counts, title):
    sns.histplot(data_w_counts, x="round", hue="assignment", multiple="stack", palette="rocket")
    plt.suptitle(title)
    plt.show()


# Likelihoods must be performed on all the same sequences for both rounds
def compare_likelihood_correlation(likeli1, likeli2, title, rounds):
    fig, axs = plt.subplots(1, 1)
    axs.scatter(likeli1, likeli2, cmap="plasma", alpha=0.75, marker=".")
    axs.set(xlabel=f"Log-Likelihood {rounds[1]}", ylabel=f"Log-Likelihood {rounds[0]}")
    fig.suptitle(title)
    plt.show()


# Return dataframe of specific round with entries with likelihood < lmax and > lmin
def data_subset(data_df, likelihood_dict, target, lmin, lmax):
    tdf = data_df[data_df["round"] == target]
    all_seqs = tdf.sequence.tolist()
    all_counts = tdf.copynum.tolist()
    likelihood = likelihood_dict[target]
    seqs, counts, ls = zip(*[(all_seqs[xid], all_counts[xid], x) for xid, x in enumerate(likelihood) if lmin < x < lmax])
    final = pd.DataFrame({"sequence": seqs, "copynum": counts, "likelihood": ls})
    return final


def seq_logo(dataframe, output_file, weight=False, outdir=""):
    out = outdir + output_file
    df = dataframe[["sequence", "copynum"]]
    df.to_csv('tmp.csv', sep='\t', index=False, header=False)
    if weight:
        sp.check_call(f"/home/jonah/kpLogo/bin/kpLogo tmp.csv -simple -o {out} -alphabet ACDEFGHIKLMNPQRSTVWY- -fontsize 20 -seq 1 -weight 2", shell=True)
    else:
        sp.check_call(f"/home/jonah/kpLogo/bin/kpLogo tmp.csv -simple -o {out} -alphabet ACDEFGHIKLMNPQRSTVWY- -fontsize 20 -seq 1", shell=True)
    sp.check_call("rm tmp.csv", shell=True)
    return out


def view_weights(rbm, type="max", selected=None, molecule="protein", title=None):
    beta, W = rbm.get_beta_and_W(rbm)
    order = np.argsort(beta)[::-1]
    W = W[order]
    assert type in ["max", "select"]
    assert molecule in ["protein", "dna", "rna"]
    if type == "max":
        assert isinstance(selected, int)
        selected_weights = W[:selected]
    elif type == "select":
        selected_weights = W[:len(selected)] # make array of correct size
        assert isinstance(selected, list)
        for id, i in enumerate(selected):
            selected_weights[id] = W[i]  # Overwrite with weights we are interested in

    # Assume we want weights
    fig = rbm_utils.Sequence_logo_multiple(selected_weights, data_type="weights", title=title, ncols=1, molecule=molecule)

def dataframe_to_input(dataframe, base_to_id, v_num):
    seqs = dataframe["sequence"].tolist()
    cat_ten = torch.zeros((len(seqs), v_num), dtype=torch.long)
    for iid, seq in enumerate(seqs):
        for n, base in enumerate(seq):
            cat_ten[iid, n] = base_to_id[base]
    return cat_ten

def cgf_with_weights_plot(rbm, dataframe, hidden_unit_numbers):
    # Convert Sequences to Integer Format and Compute Hidden Unit Input
    v_num = rbm.v_num
    base_to_id = int_to_letter_dicts[rbm.molecule]
    data_tensor = dataframe_to_input(dataframe, base_to_id, v_num)
    input_hiddens = rbm.compute_output_v(data_tensor)

    # Get Beta and sort hidden Units by Frobenius Norms
    beta, W = rbm.get_beta_and_W(rbm)
    order = np.argsort(beta)[::-1]

    gs_kw = dict(width_ratios=[3, 1], height_ratios=[1 for x in hidden_unit_numbers])
    grid_names = [[f"weight{i}", f"cgf{i}"] for i in range(len(hidden_unit_numbers))]
    fig, axd = plt.subplot_mosaic(grid_names, gridspec_kw=gs_kw, figsize=(10, 5*len(hidden_unit_numbers)), constrained_layout=True)
    for hid, hu_num in enumerate(hidden_unit_numbers):
        ix = order[hu_num]
        lims = [(np.sum(np.min(w, axis=1)), np.sum(np.max(w, axis=1))) for w in W]
        npoints = 1000
        x = lims[ix]
        fullrange = np.arange(x[0], x[1], (x[1] - x[0] + 1 / npoints) / npoints)
        # fullranges = np.array([np.arange(x[0], x[1], (x[1]-x[0]+1/npoints)/npoints) for x in lims], dtype=object)
        pre_cgf = rbm.cgf_from_inputs(fullrange.transpose())









