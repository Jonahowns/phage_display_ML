from nupack import *
import pandas as pd

config.parallelism = True
config.cache = 8.0 # GB


from rbm_torch.utils.utils import fasta_read

import math
from copy import copy


# def write_ss_file(file, seqs, structures):
#     o = open(file, "w+")
#     for iid, i in enumerate(seqs):
#         print(i, file=o)
#         print(structures[iid], file=o)
#     o.close()


# sodium range [0.05,1.1]
# magnesium range [0.0, 0.2]
def nupack_predict_ss(file, format="csv", molecule="dna", ensemble="stacking", celsius=25, sodium=0.157, magnesium=0.03):
    assert molecule in ["dna", "rna"]
    if format == "fasta":
        all_seqs, all_counts, all_chars, q = fasta_read(file, molecule, threads=6, drop_duplicates=True)
        no_wildcards = [seq for seq in all_seqs if "N" not in seq and "-" not in seq]  # Must remove all wildcard sequences or nupack won't work
        df = pd.DataFrame({"sequence": no_wildcards})

    elif format == "csv":
        in_df = pd.read_csv(file)
        all_seqs = in_df["sequence"].tolist()
        no_wildcards = [True if "N" not in seq and "-" not in seq else False for seq in all_seqs]  # Must remove all wildcard sequences or nupack won't work
        df = copy(in_df[no_wildcards])
        no_wildcards = df["sequence"].tolist()

    strands = [Strand(seq, name=str(i)) for i, seq in enumerate(no_wildcards)]
    model = Model(material=molecule.upper(), celsius=celsius, sodium=sodium, magnesium=magnesium)
    batch_size = 1
    total_batches = math.ceil(len(strands) / batch_size)
    secondary_structures, energies = [], []
    for i in range(len(strands)):
        if i % 2 == 0:
            print(f"Progress {i/(total_batches-1) * 100}")
        mfe_res = mfe(strands=strands[i], model=model)  # Calculate mfe for all provided sequences
        secondary_structures.append(mfe_res[0].structure)
        energies.append(mfe_res[0].energy)

    df["ss"] = secondary_structures
    df["mfe"] = energies
    outfilename = file.split("/")[-1].split('.')[0] + "_sspred.csv"
    df.to_csv(outfilename)





if __name__ == "__main__":
    # 25 C and 37 C, no impact on binders SP5, SP5. SP7
    # nupack_predict_ss("../../../../datasets/cov/r12.fasta", molecule="dna", celsius=25, sodium=0.157, magnesium=0.03)

    nupack_predict_ss("../../../../datasets/cov/analysis/gen_l2_seqs_v47_weightedsubsetsampling.csv", format="csv", molecule="dna", celsius=37, sodium=0.137, magnesium=0.003)
