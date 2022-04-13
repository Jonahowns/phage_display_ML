import nupack as nu
import sys
sys.path.append("../../")
from rbm_utils import fasta_read
import math

# sodium range [0.05,1.1]
# magnesium range [0.0, 0.2]
def nupack_predict_ss(fasta_file, molecule="dna", celsius=23, sodium=0.5, magnesium=0.01):
    assert molecule in ["dna", "rna"]
    all_seqs, all_counts, all_chars, q = fasta_read(fasta_file, molecule, threads=6, drop_duplicates=True)
    no_wildcards = [seq for seq in all_seqs if "N" not in seq and "-" not in seq]  # Must remove all wildcard sequences or nupack won't work

    model = nu.Model(material=molecule.upper(), celsius=celsius, sodium=sodium, magnesium=magnesium)
    batch_size = 250
    total_batches = math.ceil(len(no_wildcards) / batch_size)
    secondary_structures = []
    for i in range(total_batches):
        if i == total_batches - 1:
            mfe = nu.mfe(strands=no_wildcards[i * batch_size:], model=model)  # Calculate mfe for all provided sequences
        else:
            mfe = nu.mfe(strands=no_wildcards[i * batch_size:(i + 1)*batch_size], model=model)  # Calculate mfe for all provided sequences
        secondary_structures += [str(x.structure) for x in mfe]
    return no_wildcards, secondary_structures

def write_ss_file(file, seqs, structures):
    o = open(file, "w+")
    for iid, i in enumerate(seqs):
        print(i, file=o)
        print(structures[iid], file=o)
    o.close()

if __name__ == "__main__":
    # 25 C and 37 C, no impact on binders SP5, SP5. SP7
    seqs, structs = nupack_predict_ss("../../../cov/r12.fasta", molecule="dna", celsius=25, sodium=0.157, magnesium=0.03)
    write_ss_file("r12_ss.txt", seqs, structs)
