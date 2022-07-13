import sys
from collections import Counter
from copy import copy

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import json
import pickle
import math

from rbm_torch.global_info import supported_datatypes
from rbm_torch.utils.utils import fasta_read

from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool
import time
from itertools import repeat, chain


def make_weight_file(filebasename, weights, extension, dir="./"):
    """ Creates Weight File for

    Parameters
    ----------
    filebasename: str,
        output file name (no extension)
    weights: list of floats
        weight of each sequence in a separate fasta file, must be in same order as fasta file
    extension: str
        additional str specifier added to model names to indicate the model was trained with this
        corresponding weight file
    dir: str, optional, default="./"
        directory that weight file is saved to

    """
    with open(f'{dir}{filebasename}.json', 'w') as f:
        json.dump({"weights": weights, "extension": extension}, f)

def load_neighbor_file(neigh_file):
    """ Load pickled neighbor number file (generated by submit_neighbor_job.py)

    Parameters
    ----------
    neigh_file: str,
        full name of neighbor file

    Returns
    -------
    data: list
        list of neighbors from neighbor file
    """
    with open(neigh_file, "rb") as o:
        data = pickle.load(o)
    return data

def summary_np(nparray):
    """ Quick summary of 1D numpy array values

    Parameters
    ----------
    nparray: numpy array,
        1D numpy array of floats

    Returns
    -------
    summary: str,
        string with comma separated max of array, min or array, mean of array, and median of array
    """
    return f"{nparray.max()}, {nparray.min()}, {nparray.mean()}, {np.median(nparray)}"

def log_scale(listofnumbers, base=1):
    """ Return log(x+base) for each number x in provided list

    Parameters
    ----------
    listofnumbers: list,
        list of float values
    base: float, optional, default=1
        value added to each log operation to avoid 0 values ex. log(1) = 0

    Returns
    -------
    data: np array
        log of value+base for value in parameter listofnumbers
    """
    return np.asarray([math.log(x + base) for x in listofnumbers])

def quick_hist(x, outfile=None, yscale="log", bins=100):
    """ Make histogram of values, and save to file

    Parameters
    ----------
    x: list,
        values to make histogram with
    outfile: str,
        filename histogram is saved too, do not include file extension
    yscale: str, optional, default="log"
        set scale of yscale (matplotlib subplot scale options supported)
    bins: int, optional, default=100
        number of bins in histogram

    """
    fig, axs = plt.subplots(1, 1)
    axs.hist(x, bins=bins)
    axs.set_yscale(yscale)
    if outfile is not None:
        plt.savefig(outfile + ".png")
    else:
        plt.show()
    plt.close()

def scale_values_np(vals, min=0.05, max=0.95):
    """ Scale provided 1D np array values to b/t min and max

    Parameters
    ----------
    vals: np array
        1D np array of floats
    min: float, optional, default=0.05
        lowest value of vals will be scaled to this value
    max: flaot, optional, default=0.95
        largest value of vals will be scaled to this vales

    Returns
    -------
    data: np array
        np array of scaled values
    """
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
def standardize_affinities(affs, out_plots=None, scale="log", dividers=[5, 10, 25], target_scaling=[5, 10, 100], divider_type="percentile", negate_index=None, splitter=None):
    """ Generates new affinites as: new aff = 1/(#of_sequences_at_aff)*math.log(aff+0.001)

    Parameters
    ----------
    affs: list,
        list of affinities/copy numbers to "standardize"
    out_plot: str, optional, default=None
        file where plot of new affinites should be saved, don't include file extension
    dividers: list, optional, default=[5, 10, 25]


    Returns
    -------
    standardized_affinities: list
        new affinities after being "standardized"

    Notes
    -----
    Goal of this function is to remove huge impact of large number of nonspecific/non-binding sequences with copy number of 1.
    """
    # First let's calculate how many sequences of each affinity there are
    # aff_num = Counter(affs)
    # aff_totals = [aff_num[x] for x in uniq_aff]

    uniq_aff = list(set(affs))
    np_uniq_aff = np.asarray(uniq_aff)

    if divider_type == "percentile":
        percentiles = copy(dividers)
        # boundaries = [np.percentile(np_uniq_aff, p) for p in dividers]
    elif divider_type == "copynum":
        percentiles = [np.mean(np_uniq_aff <= q)*100 for q in dividers]
    boundaries = [np.percentile(np_uniq_aff, p) for p in percentiles]  # convert copynum values to quantiles
        # boundaries = [np.mean(np_uniq_aff <= q)*1000 for q in dividers]  # convert copynum values to quantiles

    boundaries.insert(0, 0)
    boundaries.append(max(uniq_aff)+1)

    totals = []
    for j in range(len(dividers)+1):
        totals.append(len([i for i in affs if boundaries[j] < i <= boundaries[j+1]]))
        
        
    if out_plots is not None:
        fig, axs = plt.subplots(1, 1)
        axs.hist(affs, bins=100)
        axs.set_yscale('log')
        axs.set_xlabel("Weight Value")
        axs.set_ylabel("Number of Sequences")
        for i in boundaries[1:-1]:
            plt.axvline(i, c="r")
        plt.savefig(out_plots+"_pre_transformation.png", dpi=400)
        

    def assign_denominator(x):
        for j in range(len(dividers)+1):
            if boundaries[j] < x <= boundaries[j+1]:
                return totals[j]

    # With our affinity totals we can now "standardize" the sequence impact by making the sum of weights of each affinity equal to 1
    if scale == "linear":
        lin_affs = scale_values_np(np.asarray(uniq_aff), min=1e-2, max=1.)
        la = lin_affs.squeeze(1).tolist()
        standardization_dict = {x: 1. / assign_denominator(x) * la[xid] for xid, x in enumerate(uniq_aff)}
    elif scale == "log":
        standardization_dict = {x: 1. / assign_denominator(x) * math.log(x + 0.001) for xid, x in enumerate(uniq_aff)}  # format: old_aff is key and new one is value
    else:
        print(f"scale option {scale} not supported")
        exit(1)

    stand_affs = list(map(standardization_dict.get, affs))  # Replace each value in affs with the dictionary replacement defined in standardization dict

    # avoid going too low
    single_precision_eps = np.finfo("float32").eps
    if min(stand_affs) < single_precision_eps:
        fix_min = single_precision_eps / min(stand_affs)
        stand_affs = [x * fix_min for x in stand_affs]
        standardization_dict = {k: v*fix_min for k, v in standardization_dict.items()}

    # Calculate Rough Sums, so we have a general idea of the total weight of each section
    uniq_stand_affs = list(set(stand_affs))

    new_boundaries = [np.percentile(np.asarray(uniq_stand_affs), p) for p in percentiles]
    new_boundaries.append(max(uniq_stand_affs)+1)

    percentile_sums = []
    for nib, nb in enumerate(new_boundaries):
        if nib != 0:
            percentile_sums.append(sum([x for x in stand_affs if new_boundaries[nib-1] < x <= nb]))
        else:
            percentile_sums.append(sum([x for x in stand_affs if x <= nb]))


    multi_factors = []
    for i in range(1, len(dividers)+1):
        sum_ratio = percentile_sums[i] / percentile_sums[i-1]
        multi_factors.append(target_scaling[i-1]/sum_ratio)
        percentile_sums[i] *= multi_factors[-1]

    def assign_factor(x):
        for j in range(1, len(new_boundaries)):
            if new_boundaries[j-1] < x <= new_boundaries[j]:
                return np.prod(multi_factors[j-1])
        return 1.
    
    stand_affs = [x * assign_factor(x) for x in stand_affs]
    standardization_dict = {k : v * assign_factor(v) for k, v in standardization_dict.items()}
    new_boundaries = [x * np.prod(multi_factors[:xid]) for xid, x in enumerate(new_boundaries[:-1])]
    new_boundaries.append(max(stand_affs)+1)
    new_boundaries.insert(0, 0.)

    new_percentile_sums = []
    for nib in range(len(new_boundaries)-1):
        new_percentile_sums.append(sum([x for x in stand_affs if new_boundaries[nib] < x <= new_boundaries[nib+1]]))

    if negate_index is not None:
        stand_affs = [x if x > new_boundaries[negate_index + 1] else -x for x in stand_affs]

    if splitter is not None:
        stand_affs = [x - splitter[0] if x < 0 else x + splitter[1] for x in stand_affs]

    if out_plots is not None:
        with open(out_plots+"_affinity_mapping.txt", "w") as o:
            print(f"Standardization run with options: scale={scale}, dividers={dividers}, divider_type={divider_type}, target_scaling={target_scaling}", file=o)
            print("Sums of all weights within a given range", file=o)
            for i in range(len(new_percentile_sums)):
                if i == 0:
                    print(f"range {0} to {new_boundaries[i+1]}: sum = {new_percentile_sums[i]}", file=o)
                else:
                    print(f"range {new_boundaries[i]} to {new_boundaries[i+1]}: sum = {new_percentile_sums[i]}", file=o)
            print("Affinity original value to new value mapping", file=o)
            for key in sorted(standardization_dict):
                print(f"{key} : {standardization_dict[key]}", file=o)

    if out_plots is not None:
        fig, axs = plt.subplots(1, 1)
        axs.hist(stand_affs, bins=100)
        axs.set_yscale('log')
        axs.set_xlabel("Weight Value")
        axs.set_ylabel("Number of Sequences")
        for i in new_boundaries[1:-1]:
            plt.axvline(i, c="r")
        plt.savefig(out_plots+"_post_transformation.png", dpi=400)

    return stand_affs


def negate_affinites(affs, threshold, out_plot = None, negative_factor = 1.0):
    new_affs = [x if x > threshold else -x*negative_factor for x in affs]

    if out_plot is not None:
        fig, axs = plt.subplots(1, 1)
        axs.hist(new_affs, bins=100)
        axs.set_yscale('log')
        axs.set_xlabel("Weight Value")
        axs.set_ylabel("Number of Sequences")
        plt.savefig(out_plot+".png", dpi=400)

    return new_affs

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
    if cnum > 1:
        for i in range(cnum):
            ctmp, caffs = prep_data(seqs, lenindices[i][0], lenindices[i][1], cpy_num=cpy_num)
            if uniform_length:
                ctmp = gap_adder(ctmp, lenindices[i][1], position_indx=position_indx[i])
            write_fasta(ctmp, caffs, outdir + f'_c{i+1}.fasta')
    else:
        ctmp, caffs = prep_data(seqs, lenindices[0][0], lenindices[0][1], cpy_num=cpy_num)
        if uniform_length:
            ctmp = gap_adder(ctmp, lenindices[0][1], position_indx=position_indx[0])
        write_fasta(ctmp, caffs, outdir + '.fasta')


# Should refactor to be process_raw_sequence_files
def process_raw_fasta_files(*files, in_dir="./", out_dir="./", violin_out=None, input_format="fasta"):
    """ Read in Sequence Files and extract relevant information to a pandas dataframe

    Parameters
    ----------
    *files: str
        filenames of all sequence files (include extension)
    in_dir: str, optional, default="./"
        directory where fasta files are stored (relative to where you call this function from)
    out_dir: str, optional, default="./"
        directory where full length reports are saved (relative to where you call this function from)
    violin_out: str, optional, default=None,
        file to save violin plot of data lengths to (do not include file extension)
    input_format: str, optional, default="fasta"
        format of the provided sequence files, supported options {"fasta", "gunter", "caris"}

    Returns
    -------
    dataframe: pandas.DataFrame
        contains all sequence, copy_number, length, and round (file names) information from input files
    """
    dfs = []
    all_chars = []  # Find what characters are in dataset
    for file in files:
        rnd = os.path.basename(file).split(".")[0]  # Round is the name of the fasta file
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
        violin_out = out_dir + violin_out
        plt.savefig(violin_out+".png", dpi=300)
    print("Observed Characters:", all_chars)
    return master_df

def copynum_topology(dataframe, rounds):
    # get all seqs
    all_uniq_seqs = list(set(dataframe["sequence"].tolist()))
    # make a copy number array to store copy number of each round
    copynums = np.empty((len(all_uniq_seqs), len(rounds)))
    copynums[:] = np.nan

    # replace round identifiers with their index in the rounds
    # rdict = {r: rounds.index(r) for r in rounds}
    # dataframe.replace({"round": rdict}, inplace=True)

    progress_bar_divisor = len(all_uniq_seqs) // 50

    for sid, seq in enumerate(all_uniq_seqs):
        matching = dataframe[dataframe["sequence"] == seq]
        for index, row in matching.iterrows():
            copynums[sid][rounds.index(row["round"])] = row["copy_num"]
        if sid % progress_bar_divisor == 0:
            print(f"{sid / len(all_uniq_seqs) * 100 }% complete")

    copynum_dict = {r: copynums[:, rid] for rid, r in enumerate(rounds)}
    ndf = pd.DataFrame({"sequence": all_uniq_seqs, **copynum_dict})
    # ndf["mean"] = ndf.apply(lambda row : np.nanmean(np.asarray([row[x] for x in rounds])), axis=1)
    return ndf


def copynum_topology_faster(dataframe, rounds, threads_per_task=1):
    dfs = []
    for r in rounds:
        dfs.append(dataframe[dataframe["round"] == r])
        
    merged_df_pairs = []
    for i in range(0, len(rounds)):
        for j in range(1, len(rounds)):
            if i >= j:
                continue
            merged = pd.merge(dfs[i], dfs[j], how='inner', left_on='sequence', right_on='sequence')
            merged_df_pairs.append(merged)

    # Get sequences that appear in more than 1 dataset
    all_seqs_lists = [x["sequence"].tolist() for x in merged_df_pairs]
    seqs_to_query = list(set([j for i in all_seqs_lists for j in i]))

    threads = len(rounds)

    if threads_per_task > 1:
        queries_per_task = math.ceil(len(seqs_to_query)/threads_per_task)
        split_query = [seqs_to_query[i*queries_per_task:(i+1)*queries_per_task] for i in range(threads_per_task)]
        all_seq_queries = split_query * len(rounds)
        ndfs = []
        for i in range(threads):
            for j in range(threads_per_task):
                ndfs.append(dfs[i])

        p = Pool(threads * threads_per_task)
        start = time.time()
        results = p.starmap(query_seq_in_dataframe, zip(all_seq_queries, ndfs))
        end = time.time()

        print("Process Time", end - start)

        copynums = np.empty((len(seqs_to_query), len(rounds)))
        for i in range(threads):
            single_df_results = results[i*threads_per_task:(i+1)*threads_per_task]
            copynums[:, i] = list(chain(*single_df_results))

    else:
        p = Pool(threads)
        start = time.time()
        results = p.starmap(query_seq_in_dataframe, zip(repeat(seqs_to_query), dfs))
        end = time.time()

        print("Process Time", end - start)

        copynums = np.empty((len(seqs_to_query), len(rounds)))
        for i in range(threads):
            copynums[:, i] = results[i]

    copynum_dict = {r: copynums[:, rid] for rid, r in enumerate(rounds)}
    ndf = pd.DataFrame({"sequence": seqs_to_query, **copynum_dict})
    return ndf

def query_seq_in_dataframe(seqs_to_query, dataframe):
    query_results = []
    dataframe_seqs = list(dataframe["sequence"].values)
    for s in seqs_to_query:
        if s in dataframe_seqs:
            row = dataframe.iloc[dataframe_seqs.index(s)]
            query_results.append(row["copy_num"])
        else:
            query_results.append(np.nan)
    return np.asarray(query_results)

def dataframe_to_fasta(df, out, count_key="copy_num"):
    write_fasta(df["sequence"].tolist(), df[count_key].tolist(), out)



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
    # if dt["id"] is not None:
    #     target_dir = target_dir + dt["process"] + f"_{dt['clusters']}_clusters/"
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
