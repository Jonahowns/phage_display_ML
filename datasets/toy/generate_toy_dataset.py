from rbm_torch.utils.data_prep import write_fasta
import numpy as np

dataset_size = 2000
dataset_clusters = [("ACGTTTAA", 0.25), ("CCGACGGT", 0.5), ("GGGAAGGG", 0.25) ]

min_size, max_size = 16, 20
seq_lens = [np.random.randint(min_size, max_size+1, int(dataset_size*fraction)) for motif, fraction in dataset_clusters]
motif_positions = [np.random.randint(0, min_size-len(motif), int(dataset_size*fraction)) for motif, fraction in dataset_clusters]

seqs = []
for did, dc in enumerate(dataset_clusters):
    motif, fraction = dc
    sls = seq_lens[did]
    mps = motif_positions[did]
    for i in range(int(dataset_size*fraction)):
        # random sequence
        seq = "".join([np.random.choice(["A", "C", "G", "T"]) for _ in range(sls[i])])
        # put motif in position
        seq = seq[:mps[i]] + motif + seq[mps[i]+len(motif):]
        # seq[mps[i]: mps[i] + len(motif)] = motif
        # adjust len of seq
        if len(seq) != max_size:
            seq += "".join(["-" for _ in range(max_size - len(seq))])
        seqs.append(seq)

affs = list(np.arange(0, dataset_size, 1))

write_fasta(seqs, affs, "toy.fasta")
