from sklearn.metrics import pairwise_distances_chunked
import sys
sys.path.append("../rbm_torch/")
import utils
import numpy as np
import pickle
import argparse


parser = argparse.ArgumentParser(description="CRBM Training on Phage Display Dataset")
parser.add_argument('data_dir', type=str, help="Location of Data File")
parser.add_argument('dataset_file', type=str, help="Data File name")
parser.add_argument('molecule', type=str, help="What kind of data? protein, dna or rna")
parser.add_argument('threads', type=int, help="Number of cores available")
parser.add_argument('threshold1', type=float, help="Hamming Distance/ Total length must be less than this number to be considered a neighbor")
parser.add_argument('threshold2', type=float, help="Hamming Distance/ Total length must be less than this number to be considered a neighbor")
args = parser.parse_args()

dataset_dir = args.data_dir
ffile = args.dataset_file
molecule = args.molecule
threads = args.threads

seqs, affs, all_chars, q = utils.fasta_read(dataset_dir + ffile, molecule, threads=threads, drop_duplicates=False)

r6_cat = utils.seq_to_cat(seqs, molecule=molecule)

X = r6_cat.numpy().astype(np.int8)

# X should be a categorical vector of shape (seqs, bases)
def calc_neighbors(X, threshold_1=0.15, threshold_2=0.25):
    def reduce_func(D_chunk, start):
        # print(D_chunk)
        neigh1 = np.asarray(D_chunk <= threshold_1).sum(1)
        neigh2 = np.asarray(D_chunk <= threshold_2).sum(1)
        return list(neigh1), list(neigh2)

    gen = pairwise_distances_chunked(X, X, reduce_func=reduce_func, metric="hamming")

    neigh1, neigh2 = [], []
    for n1, n2 in gen:
        neigh1 += n1
        neigh2 += n2
        print(round(len(neigh1) / X.shape[0] * 100, 3), "% done")

    return neigh1, neigh2


t1 = args.threshold1
t2 = args.threshold2
n1, n2 = calc_neighbors(X, threshold_1=t1, threshold_2=t2)
o = open(f"./{dataset_dir}/{ffile}_{int(t1 * 100)}_pairwise_distances.pkl", "w+")
pickle.dump(n1, o)
o.close()

o = open(f"./{dataset_dir}/{ffile}_{int(t2 * 100)}_pairwise_distances.pkl", "w+")
pickle.dump(n2, o)
o.close()
