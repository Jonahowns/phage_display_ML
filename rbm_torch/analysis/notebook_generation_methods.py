from rbm_torch.analysis.global_info import get_global_info

# datatype_str = "pig_ge2"
# mfolder = '/mnt/D1/phage_display_analysis/' # Folder on My local computer where I transfer all the RBMS




imports = """\
import sys
sys.path.append("../")

from rbm import fasta_read, get_beta_and_W, all_weights, RBM
import analysis_methods as am

from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import subprocess as sp
import matplotlib.image as mpimg
"""

# Data details and imports must be defined
pig_default_notebook = [
"""\
# generate weights in each rbmdirectory
for rid, rbm in enumerate(rbm_names):
    checkp, version_dir = am.get_checkpoint_path(rbm, rbmdir=local_rbm_dir)
    tmp = RBM.load_from_checkpoint(checkp)
    all_weights(tmp, version_dir +rbm+"_weights", 5, 1, 6, 2, "protein")
""",
"""\
# Stores all data in a dictionary ("data")
all_data = am.fetch_data(rounds, dir=data_dir, counts=True, molecule=molecule)
""",
"""\
paths_u, paths_w = [], []
for r in rounds:
    paths_u.append(am.seq_logo(all_data[all_data["round"] == r], f"{r}_seqlogo", weight=False, outdir="./generated/"))
    paths_w.append(am.seq_logo(all_data[all_data["round"] == r], f"{r}_w_seqlogo", weight=True, outdir="./generated/"))
""",
"""\
# Seq Logo showing Frequency of Each Amino Acid at each position
fig, axs = plt.subplots(5, 2)
fig.set_size_inches(15, 12)
for rid, r in enumerate(rounds):
    img1 = mpimg.imread(f"{paths_u[rid]}.freq.png")
    img2 = mpimg.imread(f"{paths_w[rid]}.freq.png")
    axs[rid][0].imshow(img1)
    axs[rid][1].imshow(img2)
    axs[rid][0].axis("off")
    axs[rid][1].axis("off")
    axs[rid][0].set_title(f"{r} Frequency Logo")
    axs[rid][1].set_title(f"{r} Weighted Frequency Logo")
plt.show()
""",
"""\
# Seq Logo showing Information of Each Amino Acid at each position
fig, axs = plt.subplots(5, 2)
fig.set_size_inches(15, 12)
for rid, r in enumerate(rounds):
    img1 = mpimg.imread(f"{paths_u[rid]}.info.png")
    img2 = mpimg.imread(f"{paths_u[rid]}.info.png")
    axs[rid][0].imshow(img1)
    axs[rid][1].imshow(img2)
    axs[rid][0].axis("off")
    axs[rid][1].axis("off")
    axs[rid][0].set_title(f"{r} Frequency Logo")
    axs[rid][1].set_title(f"{r} Weighted Frequency Logo")
plt.show()
""",
f"""\
# calculate likelihoods from last round rbm only
checkp, v_dir = am.get_checkpoint_path(rbm_names[-1], rbmdir=local_rbm_dir)
last_round_rbm = RBM.load_from_checkpoint(checkp)

# this takes awhile, might be something I optimize further in the future
am.generate_likelihoods(rounds, last_round_rbm, all_data, str(rbm_names[-1]) + "_all_likelihoods")
""",
"""\
last_round_likelihoods = am.get_likelihoods("./generated/" + str(rbm_names[-1]) + "_all_likelihoods.json")
""",
"""
# Plot Likelihoods of Each batch of Data
last_round_title = f"All data Log-Likelihood From {rbm_names[-1].upper()} RBM Cluster {cluster}"

am.plot_likelihoods(last_round_likelihoods["likelihoods"], rounds, rounds, title=last_round_title, xlim=(-250, -60), cdf=False)
""",
"""\
# calculate likelihoods from first round rbm only
checkp, v_dir = am.get_checkpoint_path(rbm_names[0], rbmdir=local_rbm_dir)
first_round_rbm = RBM.load_from_checkpoint(checkp)

# this takes awhile, might be something I optimize further in the future
am.generate_likelihoods(rounds, first_round_rbm, all_data, str(rbm_names[0]) + "_all_likelihoods")
""",
"""\
first_round_likelihoods = am.get_likelihoods("./generated/" + str(rbm_names[0]) + "_all_likelihoods.json")
""",
"""
# Plot Likelihoods of Each batch of Data
first_round_title = f"All data Log-Likelihood From {rbm_names[0].upper()} RBM Cluster {cluster}"

am.plot_likelihoods(first_round_likelihoods["likelihoods"], rounds, rounds, title=first_round_title, xlim=(-250, -60), cdf=False)
""",
"""\
lr_label = rounds[-1].upper()
fr_label = rounds[0].upper()
am.compare_likelihood_correlation(last_round_likelihoods["likelihoods"][rounds[0]], first_round_likelihoods["likelihoods"][rounds[0]], f"{lr_label} vs {fr_label} RBMs on {fr_label} dataset", [lr_label, fr_label])
""",
"""\
lr_label = rounds[-1].upper()
fr_label = rounds[0].upper()
am.compare_likelihood_correlation(last_round_likelihoods["likelihoods"][rounds[-1]], first_round_likelihoods["likelihoods"][rounds[-1]], f"{lr_label} vs {fr_label} RBMs on {fr_label} dataset", [lr_label, fr_label])
"""
]

# Assign notebooks (which should all be defined above) to the value used by in generate
notebooks = {"default_pig": pig_default_notebook,
             "default_cov": pig_default_notebook}

# returns a list of strings that are all converted into code cells
def generate_notebook(datatype_str, cluster=None, weights=False, notebook="default_pig"):
    # Get the global info, contains paths, which rounds in dataset, and other goodies
    info = get_global_info(datatype_str, cluster=cluster, weights=weights)

    data_details = []
    for key in info.keys():
        if type(info[key]) == str:
            data_details.append(f"{key} = '{info[key]}'")
        else:
            data_details.append(f"{key} = {info[key]}")

    full_notebook = [imports, "\n".join(data_details), *notebooks[notebook]]

    return full_notebook




if __name__ == '__main__':
    clust = 1









