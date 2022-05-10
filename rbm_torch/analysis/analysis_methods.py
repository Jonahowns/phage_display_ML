import sys
sys.path.append("../")
from rbm import fasta_read, RBM, get_beta_and_W
import rbm_utils
from crbm import CRBM

import math
import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import json
import subprocess as sp
import numpy as np
import torch
import matplotlib.image as mpimg
import nbformat as nbf
from notebook_generation_methods import generate_notebook

# Clusters are 1 indexed


int_to_letter_dicts = {"protein": rbm_utils.aadict, "dna": rbm_utils.dnadict, "rna": rbm_utils.rnadict}

# Colors Used for Likelihood Plots, can always add / change order
supported_colors = ["b", "r", "g", "y", "m", "c", "bl", "k", "c", "DarkKhaki", "DarkOrchid"]

# Helper Functions for loading data and loading RBMs not in our current directory
# assignment function assigns label based off the count (ex. returns "low" for count < 10 )
def fetch_data(fasta_names, dir="", counts=False, assignment_function=None, threads=1, molecule="protein"):
    for xid, x in enumerate(fasta_names):
        seqs, counts, all_chars, q_data = fasta_read(dir + "/" + x + ".fasta", molecule, drop_duplicates=True, threads=threads)
        round_label = [x for i in range(len(seqs))]
        if assignment_function is not None:
            assignment = [assignment_function(i) for i in counts]
        else:
            assignment = ["N/A" for i in counts]
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
def generate_likelihoods(rounds, RBM, all_data, identifier, key="round", dir="./generated/"):
    likelihoods = {}
    sequences = {}
    for x in rounds:
        seqs, likeli = RBM.predict(all_data[all_data[key] == x])
        likelihoods[x] = likeli
        sequences[x] = seqs
    data = {'likelihoods': likelihoods, "sequences": sequences}
    out = open(dir+identifier+".json", "w")
    json.dump(data, out)
    out.close()


def get_likelihoods(likelihoodfile):
    with open(likelihoodfile) as f:
        data = json.load(f)
    return data


# Plot Likelihoods as kde curves with each round in a new row
def plot_likelihoods(likeli,  order, labels, title=None, xaxislabel="log-likelihood", xlim=None, cdf=False):
    colors = supported_colors
    plot_num = len(order)
    fig, axs = plt.subplots(plot_num, 1, sharex=True, sharey=False)
    for xid, x in enumerate(order):
        if xlim is not None:
            axs[xid].set_xlim(*xlim)
        y = sns.kdeplot(likeli[x], shade=False, alpha=0.5, color=colors[xid], ax=axs[xid], label=labels[xid], cumulative=cdf)
        if xid == len(order) - 1:
            y.set(xlabel=xaxislabel)
        axs[xid].legend()
    if title:
        fig.suptitle(title)
    else:
        fig.suptitle("Log-Likelihood Gaussian KDE Curve of Likelihoods by Dataset")
    plt.show()


# Distribution of Counts inside each experimental dataset
def count_dist(data_w_counts, title):
    fig, axs = plt.subplots(2, 1)
    sns.histplot(data_w_counts, ax=axs[0], x="round", hue="assignment", multiple="stack", palette="rocket", stat="percent")
    sns.histplot(data_w_counts, ax=axs[1], x="round", hue="assignment", multiple="stack", palette="rocket", stat="count")
    plt.suptitle(title)
    plt.show()


# Likelihoods must be performed on all the same sequences for both rounds
def compare_likelihood_correlation(likeli1, likeli2, title, rounds):
    fig, axs = plt.subplots(1, 1)
    axs.scatter(likeli1, likeli2, cmap="plasma", alpha=0.75, marker=".")
    # Fit dashed black line
    coef = np.polyfit(likeli1, likeli2, 1)
    poly1d_fn = np.poly1d(coef)
    # poly1d_fn is now a function which takes in x and returns an estimate for y
    axs.plot(likeli1, poly1d_fn(likeli1), '--k') #'--k'=black dashed line, 'yo' = yellow circle marker
    axs.set(xlabel=f"Log-Likelihood {rounds[0]}", ylabel=f"Log-Likelihood {rounds[1]}")
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


def dataframe_to_input(dataframe, base_to_id, v_num, weights=False):
    seqs = dataframe["sequence"].tolist()
    cat_ten = torch.zeros((len(seqs), v_num), dtype=torch.long)
    for iid, seq in enumerate(seqs):
        for n, base in enumerate(seq):
            cat_ten[iid, n] = base_to_id[base]
    if weights:
        weights = dataframe["copynum"].tolist()
        return cat_ten, weights
    else:
        return cat_ten


def cgf_with_weights_plot(rbm, dataframe, hidden_unit_numbers):
    # Convert Sequences to Integer Format and Compute Hidden Unit Input
    v_num = rbm.v_num
    h_num = rbm.h_num
    base_to_id = int_to_letter_dicts[rbm.molecule]
    data_tensor, weights = dataframe_to_input(dataframe, base_to_id, v_num, weights=True)
    rbm.prep_W()
    input_hiddens = rbm.compute_output_v(data_tensor).detach().numpy()

    # Get Beta and sort hidden Units by Frobenius Norms
    beta, W = get_beta_and_W(rbm)
    order = np.argsort(beta)[::-1]

    gs_kw = dict(width_ratios=[3, 1], height_ratios=[1 for x in hidden_unit_numbers])
    grid_names = [[f"weight{i}", f"cgf{i}"] for i in range(len(hidden_unit_numbers))]
    fig, axd = plt.subplot_mosaic(grid_names, gridspec_kw=gs_kw, figsize=(10, 5*len(hidden_unit_numbers)), constrained_layout=True)

    npoints = 1000  # Number of points for graphing CGF curve
    lims = [(np.sum(np.min(w, axis=1)), np.sum(np.max(w, axis=1))) for w in W]  # Get limits for each hidden unit
    fullranges = torch.zeros((npoints, h_num))
    for i in range(h_num):
        x = lims[i]
        fullranges[:, i] = torch.tensor(np.arange(x[0], x[1], (x[1] - x[0] + 1 / npoints) / npoints).transpose())

    pre_cgf = rbm.cgf_from_inputs_h(fullranges)
    # fullrange = torch.tensor(np.arange(x[0], x[1], (x[1] - x[0] + 1 / npoints) / npoints).transpose())
    # fullranges = np.array([np.arange(x[0], x[1], (x[1]-x[0]+1/npoints)/npoints) for x in lims], dtype=object)
    for hid, hu_num in enumerate(hidden_unit_numbers):
        ix = order[hu_num]  # get weight index
        # Make Sequence Logo
        rbm_utils.Sequence_logo(W[ix], ax=axd[f"weight{hid}"], data_type="weights", ylabel=f"Weight #{hu_num}", ticks_every=5, ticks_labels_size=14, title_size=20, molecule='protein')

        t_x = np.asarray(fullranges[:, ix])
        t_y = np.asarray(pre_cgf[:, ix])
        deltay = np.min(t_y)
        counts, bins = np.histogram(input_hiddens[:, ix], bins=30, weights=weights)
        factor = np.max(t_y) / np.max(counts)
        # WEIGHTS SHOULD HAVE SAME SIZE AS BINS
        axd[f"cgf{hid}"].hist(bins[:-1], bins, color='grey', label='All sequences', weights=counts*factor,
                   histtype='step', lw=3, fill=True, alpha=0.7, edgecolor='black', linewidth=1)
        axd[f"cgf{hid}"].plot(t_x, t_y - deltay, lw=3, color='C1')
        axd[f"cgf{hid}"].set_ylabel('CGF', fontsize=18)
        axd[f"cgf{hid}"].tick_params(axis='both', direction='in', length=6, width=2, colors='k')
        axd[f"cgf{hid}"].tick_params(axis='both', labelsize=16)
        axd[f"cgf{hid}"].yaxis.tick_right()
        axd[f"cgf{hid}"].yaxis.set_label_position("right")
    plt.show()


def plot_input_mean(RBM, I, hidden_unit_numbers, I_range=None, weights=None, xlabels=None, figsize=(3, 3)):
    if type(hidden_unit_numbers) in [int]:
        hidden_unit_numbers = [hidden_unit_numbers]

    # Get order of highest normed weights
    beta, W = get_beta_and_W(RBM)
    order = np.argsort(beta)[::-1]

    # Change hidden unit numbers to correspond to the max weights inputs
    hidden_unit_numbers = [order[x] for x in hidden_unit_numbers]

    nfeatures = len(hidden_unit_numbers)
    nrows = int(np.ceil(nfeatures / float(2)))

    if I_range is None:
        I_min = I.min()
        I_max = I.max()
        I_range = (I_max - I_min) * torch.arange(0, 1 + 0.01, 0.01) + I_min

    mean = RBM.mean_h(torch.repeat_interleave(I_range.unsqueeze(1), RBM.h_num, dim=1))

    gs_kw = dict(width_ratios=[1, 1], height_ratios=[1 for x in range(nrows)])
    grid_names = [[f"{i}_l", f"{i}_r"] for i in range(nrows)]
    fig, axd = plt.subplot_mosaic(grid_names, gridspec_kw=gs_kw, figsize=(2 * figsize[0], nrows * figsize[1]),
                                  constrained_layout=True)

    mean = mean.detach().numpy()
    I_range = I_range.detach().numpy()
    I = I.detach().numpy()

    if xlabels is None:
        xlabels = [r'Input $I_{%s}$' % (i) for i in range(nfeatures)]

    row_dict = {0: "_l", 1: "_r"}
    for i in range(nrows):  # row number
        for j in range(2):  # Column Number
            if not i * 2 + j < len(hidden_unit_numbers):
                ax = axd[f"{i}{row_dict[j]}"]
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax = axd[f"{i}{row_dict[j]}"]
                ax2 = ax.twinx()
                ax2.hist(I[:, hidden_unit_numbers[i * 2 + j]], density=True, weights=weights, bins=100)
                ax.plot(I_range, mean[:, hidden_unit_numbers[i * 2 + j]], c='black', linewidth=2)
                xmin = np.min(I[:, hidden_unit_numbers[i * 2 + j]])
                xmax = I[:, hidden_unit_numbers[i * 2 + j]].max()
                ymin = mean[:, hidden_unit_numbers[i * 2 + j]].min()
                ymax = mean[:, hidden_unit_numbers[i * 2 + j]].max()

                ax.set_xlim([xmin, xmax])
                step = int((xmax - xmin) / 4.0) + 1
                xticks = np.arange(int(xmin), int(xmax) + 1, step)
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticks, fontsize=12)
                ax.set_ylim([ymin, ymax])
                step = int((ymax - ymin) / 4.0) + 1
                yticks = np.arange(int(ymin), int(ymax) + 1, step)
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticks, fontsize=12)
                ax2.set_yticks([])
                for tl in ax.get_yticklabels():
                    tl.set_fontsize(14)
                ax.set_zorder(ax2.get_zorder() + 1)
                ax.patch.set_visible(False)
                ax.set_xlabel(xlabels[i * 2 + j], fontsize=14)

    plt.show()


def shaded_kde_curve(ax, path, xmin, xmax, color):
    vertices = [ (x, y) for x, y in path.vertices if xmin < x < xmax]
    vertices.insert(0, (xmin, 0.))
    vertices.append((xmax, 0.))
    xfill, yfill = zip(*vertices)
    ax.fill(xfill, yfill, color)


def color_subplot(ax, color):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)


def cluster_inputs(I, hidden_unit_2d_combo, padding=0.05, size_to=None, hue_to=None, size_label=None, hue_label=None):

    # mean = RBM.mean_h(torch.repeat_interleave(I_range.unsqueeze(1), RBM.h_num, dim=1))

    [hidden1, hidden2] = hidden_unit_2d_combo

    # mean = mean.detach().numpy()
    # I_range = I_range.detach().numpy()
    I = I.detach().numpy()

    xlab = r'Input $I_{%s}$'%hidden1
    ylab = r'Input $I_{%s}$'%hidden2

    Ih1 = I[:, hidden1]
    Ih2 = I[:, hidden2]

    sns.color_palette("mako", as_cmap=True)
    marker_size = 10
    if size_to is not None and hue_to is None:
        dataset = pd.DataFrame({"input1": Ih1, "input2": Ih2, size_label: size_to})
        g = sns.scatterplot(data=dataset, x="input1", y="input2", size=size_label, alpha=0.7, s=marker_size, palette="mako")
    elif size_to is None and hue_to is not None:
        dataset = pd.DataFrame({"input1": Ih1, "input2": Ih2, hue_label: hue_to})
        g = sns.scatterplot(data=dataset, x="input1", y="input2", hue=hue_label, alpha=0.7, s=marker_size, palette="mako")
    elif size_to is not None and hue_to is not None:
        dataset = pd.DataFrame({"input1": Ih1, "input2": Ih2, size_label: size_to, hue_label: hue_to})
        g = sns.scatterplot(data=dataset, x="input1", y="input2", size=size_label, hue=hue_label, alpha=0.7, s=marker_size, palette="mako")
    else:
        dataset = pd.DataFrame({"input1": Ih1, "input2": Ih2})
        g = sns.scatterplot(data=dataset, x="input1", y="input2", alpha=0.7, s=marker_size, palette="mako")

    # sns.histplot(data_w_counts, ax=axs[1], x="round", hue="assignment", multiple="stack", palette="rocket", stat="count")
    g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    g.set_xlim([Ih1.min()-padding, Ih1.max()+padding])
    g.set_ylim([Ih2.min()-padding, Ih2.max()+padding])
    g.set_xlabel(xlab)
    g.set_ylabel(ylab)

    # How to make Legend if needed
    # leg = g.axes.flat[0].get_legend()
    # new_title = 'Inputs'
    # leg.set_title(new_title)
    # new_labels = ['label 1', 'label 2']
    # for t, l in zip(leg.texts, new_labels):
    #     t.set_text(l)

    plt.show()


def seqlogo_subplot(ax, path, type="info"):
    img = mpimg.imread(f"{path}.{type}.png")
    ax.imshow(img, interpolation="nearest")
    ax.axis("off")
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticklabels([])


# bounds listed in ascending orderex. [[-140, -100], [-90, -72]]
def multi_peak_seq_log_fig(data, likelihoods, round, bounds, weight=False, title=None, xlim=None):
    data_subsets, seqlogo_paths = [], []
    for iid,  i in enumerate(bounds):
        sub = data_subset(data, likelihoods, round, i[0], i[1])
        seqlogo_paths.append(seq_logo(sub, f"peak{iid+1}_likeli_{round}", weight=weight, outdir="./generated/"))
        data_subsets.append(sub)

    peak_num = len(bounds)
    gs_kw = dict(width_ratios=[1 for x in range(peak_num)], height_ratios=[1, 1])
    fig, axd = plt.subplot_mosaic([['top' for x in range(peak_num)], [f'lower{x}' for x in range(peak_num)]],
                                  gridspec_kw=gs_kw, figsize=(15, 5), constrained_layout=False)

    axd["top"].set_xlim(*xlim)
    sns.kdeplot(likelihoods[round], ax=axd["top"])

    path = axd["top"].get_children()[0].get_path()

    for iid, i in enumerate(bounds):
         shaded_kde_curve(axd["top"], path, i[0], i[1], supported_colors[iid])

    axd["top"].set_xlabel("log-likelihood")
    for iid, i in enumerate(bounds):
        color_subplot(axd[f"lower{iid}"], supported_colors[iid])
        seqlogo_subplot(axd[f"lower{iid}"], seqlogo_paths[iid])

    if title is None:
        fig.suptitle(title)
    else:
        fig.suptitle(f"Composition of peaks across RBM Likelihood of Round{round}")

    plt.show()



if __name__ == '__main__':
    mdir = "/mnt/D1/globus/pig_trained_rbms/"
    rounds = ["b3", "n1", "np1", "np2", "np3"]
    c1_rounds = [x + "_c1" for x in rounds]
    c2_rounds = [x + "_c2" for x in rounds]

    data_c2 = fetch_data(c2_rounds, dir="../../pig_tissue", counts=True)
    b3_data = data_c2[data_c2["round"] == "b3_c2"]
    # b3_input, b3_weight_list = dataframe_to_input(b3_data, int_to_letter_dicts["protein"], 45, weights=True)
    checkp, v_dir = get_checkpoint_path("b3_c2", rbmdir=mdir)
    b3_rbm = RBM.load_from_checkpoint(checkp)
    cgf_with_weights_plot(b3_rbm, b3_data, [0, 1, 2, 5, 8, 9, 10, 12, 14, 16])
    print("hello")









