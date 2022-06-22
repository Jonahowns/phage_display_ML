import sys
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import pickle
import math

from rbm_torch.analysis.global_info import supported_datatypes
from rbm_torch.utils import fasta_read
from sklearn.preprocessing import MinMaxScaler


def load_neighbor_file(neigh_file):
    with open(neigh_file, "rb") as o:
        data = pickle.load(o)
    return data

def summary_np(nparray):
    return f"{nparray.max()}, {nparray.min()}, {nparray.mean()}, {np.median(nparray)}"

def log_scale(listofnumbers, base=1):
    return np.asarray([math.log(x + base) for x in listofnumbers])

def quick_hist(x, outfile, yscale="log"):
    fig, axs = plt.subplots(1, 1)
    axs.hist(x, bins=100)
    axs.set_yscale(yscale)
    plt.savefig(outfile + ".png")
    plt.close()

def scale_values_np(vals, min=0.05, max=0.95):
    nscaler = MinMaxScaler(feature_range=(min, max))
    return nscaler.fit_transform(vals.reshape(-1, 1))


# Intended to be run on an already processed fasta file with no duplicates, uniform length, and affinites denoted in the fasta file
def scale_weights(fasta_file_in, fasta_out_dir, neighbor_pickle_file, molecule="protein", threads=12, precision=5, scale_log=True, copynum_coeff=1.0, neighbor_coeff=1.0, normalize_threshold="median"):
    seqs, affs, all_chars, q = fasta_read(fasta_file_in, molecule, threads=threads, drop_duplicates=False)

    fasta_file_name = os.path.basename(fasta_file_in)
    fasta_name = fasta_file_name.split(".")[0]
    fasta_in_dir = os.path.dirname(fasta_file_in)

    full_out_dir = fasta_out_dir + f"{fasta_name}_scaled/"

    if not os.path.isdir(full_out_dir):
        os.mkdir(full_out_dir)

    log = open(full_out_dir + f"{fasta_name}_log.txt", "w")

    print("Copy Number Before", summary_np(np.asarray(affs)), file=log)
    quick_hist(affs, full_out_dir+f"affs_before_{fasta_name}")

    try:
        neighs = load_neighbor_file(neighbor_pickle_file)
    except IOError:
        print(f"Neighbor File {neighbor_pickle_file} not found")
        exit(-1)

    print("Neighbor Number Before:", summary_np(np.asarray(neighs)), file=log)
    quick_hist(neighs, full_out_dir+f"neighs_before_{fasta_name}")

    if scale_log:
        affs = log_scale(affs, base=1)

        print("Copy Number Log Scaled", summary_np(affs), file=log)
        quick_hist(affs.tolist(), full_out_dir+f"affs_log_{fasta_name}")

        neighs = log_scale(neighs, base=1)

        print("Neighbor Number Log Scaled", summary_np(neighs), file=log)
        quick_hist(neighs.tolist(), full_out_dir+ f"neighs_log_{fasta_name}")
    else:
        affs = np.asarray(affs)
        neighs = np.asarray(neighs)

    neighs_scaled = scale_values_np(neighs)

    affs_scaled = scale_values_np(affs)

    print("Neighbor Number Scaled", summary_np(neighs_scaled), file=log)
    quick_hist(neighs_scaled.squeeze(1).tolist(), full_out_dir+f"neighs_scaled_{fasta_name}")

    print("Copy Number Scaled:", summary_np(affs_scaled), file=log)
    quick_hist(affs_scaled.squeeze(1).tolist(), full_out_dir+f"affs_scaled_{fasta_name}")

    new_weights = copynum_coeff*affs_scaled + (1 - neighs_scaled) * neighbor_coeff

    if normalize_threshold is not None:
        if normalize_threshold == "median":
            threshold = np.median(new_weights)
        elif type(normalize_threshold) is float:
            threshold = normalize_threshold
        else:
            print(f"Normalize threshold Value {normalize_threshold} not recongized.")
            exit(1)

        # eunuchize these bad sequences
        cajones = new_weights <= threshold
        new_weights[cajones] /= cajones.sum()

        print(f"Normalized Very bad sequences (most likely, check the histograms) below threshold {threshold}", file=log)

    print("New Weights:", summary_np(new_weights), file=log)
    quick_hist(new_weights.squeeze(1).tolist(), full_out_dir+f"new_weights_{fasta_name}")

    print("Writing File to: ", fasta_out_dir+fasta_file_name, file=log)
    write_fasta(seqs, [round(x, precision) for x in new_weights.squeeze(1).tolist()], fasta_out_dir+fasta_file_name)
    log.close()

# Goal Remove huge impact of large number of nonspecific/non-binding sequences with copy number of 1.
# This is especially needed for early rounds of selection
def standardize_affinities(affs, out_plot=None, negative_weights=False):
    # First let's calculate how many sequences of each affinity there are
    aff_num = Counter(affs)
    uniq_aff = list(set(affs))
    aff_totals = [aff_num[x] for x in uniq_aff]

    # With our affinity totals we can now "standardize" the sequence impact by making the sum of weights of each affinity equal to 1
    standardization_dict = {x: 1./aff_totals[xid]*math.log(x + 0.001) for xid, x in enumerate(uniq_aff)}  # format: old_aff is key and new one is value
    stand_affs = list(map(standardization_dict.get, affs))  # Replace each value in affs with the dictionary replacement defined in standardization dict

    if negative_weights:
        minimum_aff = min(stand_affs) # find minimum
        stand_affs = [x if x > minimum_aff else -x for x in stand_affs]  # set these as negative weights

    if out_plot is not None:
        fig, axs = plt.subplots(1, 1)
        axs.hist(stand_affs, bins=100)
        axs.set_yscale('log')
        axs.set_xlabel("Weight Value")
        axs.set_ylabel("Number of Sequences")
        plt.savefig(out_plot+".png", dpi=400)

    return stand_affs

## Fasta File Methods
def write_fasta(seqs, affs, out):
    o = open(out, 'w')
    for xid, x in enumerate(seqs):
        print('>seq' + str(xid) + '-' + str(affs[xid]), file=o)
        print(x, file=o)
    o.close()

def fasta_read_basic(fastafile):
    o = open(fastafile)
    seqs = []
    for line in o:
        if line.startswith('>'):
            continue
        else:
            seqs.append(line.rstrip().upper())
    o.close()

    all_chars = []
    for seq in seqs:
        letters = set(list(seq))
        for l in letters:
            if l not in all_chars:
                all_chars.append(l)

    return seqs, all_chars

def gunter_read(gunterfile):
    o = open(gunterfile)
    seqs, copy_num = [], []
    for line in o:
        c_num, seq = line.split()
        seqs.append(seq.upper())
        copy_num.append(float(c_num))
    o.close()

    all_chars = []
    for seq in seqs:
        letters = set(list(seq))
        for l in letters:
            if l not in all_chars:
                all_chars.append(l)

    return seqs, copy_num, all_chars

def csv_read(csv_file, sequence_label="sequence", copy_num_label="copy_num"):
    df = pd.read_csv(csv_file)
    seqs = df[sequence_label].tolist()
    copy_num = df[copy_num_label].tolist()

    all_chars = []
    for seq in seqs:
        letters = set(list(seq))
        for l in letters:
            if l not in all_chars:
                all_chars.append(l)

    return seqs, copy_num, all_chars

def data_prop(seqs, round, outfile=sys.stdout, calculate_copy_number=True):
    if outfile != sys.stdout:
        outfile = open(outfile, 'w+')


    useqs = list(set(seqs))

    if calculate_copy_number:
        cpy_num = Counter(seqs)
        copy_number = [cpy_num[x] for x in useqs]
    print(f'Removed {len(seqs)-len(useqs)} Repeat Sequences', file=outfile)
    ltotal = []
    for s in useqs:
        l = len(s)
        ltotal.append(l)
    roundlist = [round for x in useqs]
    if calculate_copy_number:
        df = pd.DataFrame({"sequence": useqs, "length": ltotal, "round": roundlist, "copy_num": copy_number})
    else:
        df = pd.DataFrame({"sequence": useqs, "length": ltotal, "round": roundlist})
    lp = set(ltotal)
    lps = sorted(lp)
    counts = []
    for x in lps:
        c = 0
        for aas in useqs:
            if len(aas) == x:
                c+=1
        counts.append(c)
        print('Length:', x, 'Number of Sequences', c, file=outfile)
    # if violin_out is not None:
    #     seaborn.violinplot(x=edf.Length)
    #     # fig, ax = plt.subplots(1, 1)
    #     # ax.violinplot([x for x in ltotal if x < 60], points=200, vert=False, widths=1.1,
    #     #              showmeans=True, showextrema=True, showmedians=True,
    #     #              quantiles=[0.05, 0.1, 0.8, 0.9], bw_method=0.5)
    #     # ax.set_xlabel("Sequence Length")
    #     plt.savefig(violin_out+".png", dpi=300)
    return df

def prep_data(seqs, lmin=0, lmax=10, cpy_num=None):
    fseqs = []
    affs = []
    for sid, s in enumerate(seqs):
        if lmin <= len(s) <= lmax:
            fseqs.append(s)
            if cpy_num is not None:
                affs.append(cpy_num[sid])
        else:
            continue
    if cpy_num is not None:
        return fseqs, affs
    else:
        return fseqs

# Adds gaps to sequences at specified index (position_indx) to match maxlen
def gap_adder(seqs, maxlen, position_indx=-1):
    nseqs = []
    for seq in seqs:
        if position_indx == -1:
            nseqs.append(seq.replace("*", "-") + "".join(["-" for i in range(maxlen - len(seq))]))
        else:
            tseq = seq.replace("*", "-")
            dashes = "".join(["-" for i in range(maxlen - len(seq))])
            tseq.insert(dashes, position_indx)
            nseqs.append(tseq)
    return nseqs

# Clusters are defined by the length of the sequences
# Extractor takes all data and writes fasta file for sequences of the specified lengths
def extractor(seqs, cnum, lenindices, outdir, cpy_num, uniform_length=True, position_indx=[-1]):
    for i in range(cnum):
        ctmp, caffs = prep_data(seqs, lenindices[i][0], lenindices[i][1], cpy_num=cpy_num)
        if uniform_length:
            ctmp = gap_adder(ctmp, lenindices[i][1], position_indx=position_indx[i])
        write_fasta(ctmp, caffs, outdir + '.fasta')


def process_raw_fasta_files(*files, in_dir=None, out_dir=None, violin_out=None, input_format="fasta"):
    dfs = []
    all_chars = []  # Find what characters are in dataset
    for file in files:
        rnd = os.path.basename(file).split(".")[0]  # Round is the name of the fasta file
        if in_dir is not None:
            file = in_dir + file
        if input_format == "fasta":
            seqs, rnd_chars = fasta_read_basic(file)
            all_chars += rnd_chars
            df = data_prop(seqs, rnd, outfile=out_dir + f"{rnd}_len_report.txt", calculate_copy_number=True)
            dfs.append(df)
        elif input_format == "gunter":
            seqs, copy_num, rnd_chars = gunter_read(file)
            all_chars += rnd_chars
            df = data_prop(seqs, rnd, outfile=out_dir + f"{rnd}_len_report.txt", calculate_copy_number=False)
            df["copy_num"] = copy_num
            dfs.append(df)
        elif input_format == "caris":
            seqs, copy_num, rnd_chars = csv_read(file, sequence_label="sequence", copy_num_label="reads")
            all_chars += rnd_chars
            df = data_prop(seqs, rnd, outfile=out_dir + f"{rnd}_len_report.txt", calculate_copy_number=False)
            df["copy_num"] = copy_num
            dfs.append(df)
        else:
            print(f"Input file format {input_format} is not supported.")
            print(exit(-1))
    all_chars = list(set(all_chars))
    master_df = pd.concat(dfs)
    if violin_out is not None:
        sns.violinplot(data=master_df, x="round", y="length")
        if out_dir is not None:
            violin_out = out_dir + violin_out
        plt.savefig(violin_out+".png", dpi=300)
    print("Observed Characters:", all_chars)
    return master_df

# Prepares data and puts it into a fasta file
# Datatype must be defined in global_info.py to work properly
# target_dir is where the files should be saved to
# master_df is what we get from process_raw_fasta_files
# Character Conversion will replace characters of strings. Must be dict. ex. {"T": "U"} will replace all Ts with Us
# Remove chars deletes sequences with the provided chars. Must be a list
def prepare_data_files(datatype_str, master_df, target_dir, character_conversion=None, remove_chars=None):
    try:
        dt = supported_datatypes[datatype_str]
    except KeyError:
        print(f"No datatype of specifier {datatype_str} found. Please add to global_info.py")
        exit(-1)

    # Add
    if dt["process"] is not None:
        target_dir = target_dir + dt["process"] + f"_{dt['clusters']}_clusters/"
    # Make directory for files if not already specified
    if not os.path.isdir(target_dir):
        os.mkdir(f"./{target_dir}")

    rounds = list(set(master_df["round"].tolist()))
    for round in rounds:
        round_data = master_df[master_df["round"] == round]
        r_seqs = round_data.sequence.tolist()
        r_copynum = round_data.copy_num.tolist()
        if remove_chars is not None:
            for char in remove_chars:
                og_len = len(r_seqs)
                rs, rc = zip(*[(seq, copy_num) for seq, copy_num in zip(r_seqs, r_copynum) if seq.find(char) == -1])
                r_seqs, r_copynum = list(rs), list(rc)
                new_len = len(r_seqs)
                print(f"Removed {og_len-new_len} sequences with character {char}")

        if character_conversion is not None:
            for key, value in character_conversion.items():
                r_seqs = [x.replace(key, value) for x in r_seqs]
        extractor(r_seqs, dt['clusters'], dt["cluster_indices"], target_dir+round, r_copynum, uniform_length=True, position_indx=dt["gap_position_indices"])
