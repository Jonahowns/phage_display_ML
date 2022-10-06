import numpy as np
import pandas as pd

# Let's classify each of the experimental sequences in a cluster
exp_seqs = {
    "SP5_sup": 'ACCATGGTAGGTATTGCTTGGTAGGGATAGTGGGCTTGGT',
    "SP5_pap": '-ACCATGGTAGGTATTGCTTGGTAGGGATAGTGGGCTTTG',
    "SP6": "CCCATGGTAGGTATTGCTTGGTAGGGATAGTGGGCTTGGT",
    "SP7_sup": "CGGAGGGTAGGTAGTGCTTGGTAGGGAAACTCCGCCGGGT",
    "SP7_pap": "-AGGAGGGTAGGTAGTGCTTGGTAGGGAAACTCCGCCGAT",
    "SP6C": 'CCCATGGTAGGTATTGCTTGGTAGCGATAGTGGGCTTGGT',
    "SP634":  "CCCATGGTAGGTATTGCTTGGTAGGGATAGTGGG------",
    "SP634G": "CCCATGGTAGGTATTGGTTGGTAGGGATAGTGGG------",
    "SP634C": "CCCATGGTAGGTATTGCTTGGTAGCGATAGTGGG------",
    "SP634A": "CCCATGGTAGGTATTGCATGGTAGGGATAGTGGG------",
    "SP630":  "---CATGGTAGGTATTGCATGGTAGGGATAGTG-------",
    "SP619":  "------------TATTGCATGGTAGGGATAG---------",
    "RBD1":   "AGGAGGGTAGGTAGTGCTTGGTAGGGAAACTCCGCCGATT",
    "RBD2":   "GTTAGGTTCTGGATTAGGTTAGGGTTGTGTTGTTGTTAGG",
    "RBD3":   "TACAGTTGGTTGTAGGTTTTTGTTAGGTTAGTTTAGGGTT",
    "RBD4":   "TGGGTGTTTTGGTTGTAGGGTTTAGGTTTAGGGTACTCTT",
}

# nowhere in the paper are all these values compared in one experiment
# Rather, the values are extrapolated based off the values sp5, sp6, sp7 values in Figure 1h
fluorescence = {
     "SP5_sup": 1500,
    "SP5_pap": 1500,
    "SP6": 2500,
    "SP7_sup": 2750,
    "SP7_pap": 2750,
    "SP6C": 45,
    "SP634":  2250,
    "SP634G": 10,
    "SP634C": 10,
    "SP634A": 50,
    "SP630":  500,
    "SP619":  10,
    "RBD1":   178.6,
    "RBD2":   196.42,
    "RBD3":   214.42,
    "RBD4":   392.82,
}


def fill_gaps(dict, set_nuc="-"):
    keys = dict.keys()
    values = dict.values()
    new_exp_seqs = {}
    for i in keys:
        seq = dict[i]
        while "-" in seq:
            if set_nuc == "-":
                new_char = np.random.choice(["A", "C", "G", "T"])
            else:
                new_char = set_nuc
            ind = seq.find("-")
            seq = seq[:ind] + new_char + seq[ind+1:]
        new_exp_seqs[i] = seq
    return new_exp_seqs


def generate_test_set(set_nuc="-"):
    test_set = fill_gaps(exp_seqs, set_nuc=set_nuc)
    return pd.DataFrame({"name": test_set.keys(), "sequence": test_set.values(), "fluorescence": fluorescence.values()})
