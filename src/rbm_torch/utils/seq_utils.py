import pandas as pd
from sklearn.metrics import pairwise_distances
from rbm_torch.utils import utils
import torch
import numpy as np

def cat_to_seq(categorical_tensor, molecule="protein"):
    base_to_id = utils.int_to_letter_dicts[molecule]
    seqs = []
    for i in range(categorical_tensor.shape[0]):
        seq = ""
        for j in range(categorical_tensor.shape[1]):
            seq += base_to_id[categorical_tensor[i][j]]
        seqs.append(seq)
    return seqs


def seq_to_cat(seqs, molecule="protein"):
    """takes seqs as list of strings and returns a categorical vector"""
    base_to_id = utils.letter_to_int_dicts[molecule]
    return torch.tensor(list(map(lambda x: [base_to_id[y] for y in x], seqs)), dtype=torch.long)


def cat_to_one_hot(cat_seqs, q):
    """ takes a categorical vector and returns its one hot encoded representation"""
    one_hot = np.zeros((cat_seqs.shape[0], cat_seqs.shape[1]*q))
    for i in range(cat_seqs.shape[0]):
        for j in range(cat_seqs.shape[1]):
            one_hot[i, j*q:(j+1)*q][cat_seqs[i, j]] = 1
    return one_hot


def find_nearest(sequence, dataframe, hamming_threshold=0, molecule="protein"):
    """Find sequences in dataframe within number of mutations of a given sequence"""

    # categorical vector for query sequence
    cat_query = seq_to_cat([sequence], molecule=molecule)

    seq_len = len(sequence)

    # categorical vector for database sequences
    database_seqs = dataframe["sequence"].tolist()

    database_cat = seq_to_cat(database_seqs, molecule=molecule)

    dist_matrix = pairwise_distances(cat_query, database_cat, metric="hamming") * seq_len

    seqs_of_interest = dist_matrix[0] <= hamming_threshold

    return dataframe[seqs_of_interest]


def prune_similar_sequences(dataframe, hamming_threshold=0, molecule="protein"):
    """generate subset of sequences that are at least x mutations away from one another,
    first occurrence is kept so make sure to sort dataframe prior"""
    dataframe.reset_index(drop=True, inplace=True)
    seqs = dataframe["sequence"].tolist()
    index = dataframe.index.tolist()

    cat = seq_to_cat(seqs, molecule=molecule)
    X = cat.numpy().astype(np.int8)

    seq_len = len(seqs[0])
    selected_seqs, selected_indices, selected_cat = [], [], []
    total_seqs = len(seqs)
    for i in range(total_seqs):  # len(m1_seqs)
        if i == 0:
            selected_seqs.append(seqs[i])
            selected_indices.append(index[i])
            selected_cat.append(X[i])
        else:
            # number of mutations this sequence is from all sequences in the selected subset
            dist_matrix = pairwise_distances([X[i]], selected_cat, metric="hamming") * seq_len
            if min(dist_matrix[0]) > hamming_threshold:
                selected_seqs.append(seqs[i])
                selected_indices.append(index[i])
                selected_cat.append(X[i])

    print(f"Kept {len(selected_seqs)} of {total_seqs}")

    dataframe = dataframe.iloc[selected_indices, :]
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe
