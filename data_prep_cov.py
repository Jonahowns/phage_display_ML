import sys
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from rbm_torch.analysis.global_info import supported_datatypes

# from rbm_torch.rbm_utils import aadict, dnadict, rnadict

## Fasta File Methods
def write_fasta(seqs, affs, out):
    o = open(out, 'w')
    for xid, x in enumerate(seqs):
        print('>seq' + str(xid) + '-' + str(affs[xid]), file=o)
        print(x, file=o)
    o.close()

def fasta_read(fastafile):
    o = open(fastafile)
    seqs = []
    for line in o:
        if line.startswith('>'):
            continue
        else:
            seqs.append(line.rstrip())
    o.close()
    return seqs

def data_prop(seqs, round, outfile=sys.stdout):
    if outfile != sys.stdout:
        outfile = open(outfile, 'w+')

    cpy_num = Counter(seqs)
    useqs = list(set(seqs))
    copy_number = [cpy_num[x] for x in useqs]
    print(f'Removed {len(seqs)-len(useqs)} Repeat Sequences', file=outfile)
    ltotal = []
    for s in useqs:
        l = len(s)
        ltotal.append(l)
    roundlist = [round for x in useqs]
    df = pd.DataFrame({"sequence": useqs, "length": ltotal, "round": roundlist, "copynum": copy_number})
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
    return useqs, copy_number, df


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
def extractor(seqs, cnum, lenindices, outdir, cpy_num, uniform_length=True, position_indx=-1):
    for i in range(cnum):
        ctmp, caffs = prep_data(seqs, lenindices[i][0], lenindices[i][1], cpy_num=cpy_num)
        if uniform_length:
            ctmp = gap_adder(ctmp, lenindices[i][1], position_indx=position_indx)
        write_fasta(ctmp, caffs, outdir + '.fasta')



def process_raw_fasta_files(*files, in_dir=None, out_dir=None, violin_out=None):
    dfs = []
    for fasta_file in files:
        rnd = os.path.basename(fasta_file)  # Round is the name of the fasta file
        if in_dir is not None:
            fasta_file = in_dir + fasta_file
        seqs = fasta_read(fasta_file)
        useqs, cpy_num, df = data_prop(seqs, rnd, outfile=f"{rnd}_len_report.txt")
        dfs.append(df)
    master_df = pd.concat(dfs)
    if violin_out is not None:
        sns.violinplot(data=master_df, x="round", y="length")
        if out_dir is not None:
            violin_out = out_dir + violin_out
        plt.savefig(violin_out+".png", dpi=300)
    return master_df

# Prepares data and puts it into a fasta file
# Datatype must be defined in global_info.py to work properly
# target_dir is where the files should be saved to
# master_df is what we get from process_raw_fasta_files
def prepare_data_files(datatype, master_df, target_dir):
    try:
        dt = supported_datatypes[datatype]
    except KeyError:
        print(f"No datatype of specifier {datatype} found. Please add to global_info.py")
        exit(-1)

    # Add
    if datatype["process"] is not None:
        target_dir = target_dir + dt["process"] + f"_{dt['clusters']}_clusters/"
    # Make directory for files if not already specified
    if not os.path.isdir(target_dir):
        os.mkdir(f"./{target_dir}")

    rounds = list(set(master_df["round"].tolist()))
    for round in rounds:
        round_data = master_df[master_df["round"] == round]
        r_seqs = round_data.sequence.tolist()
        r_cpynum = round_data.cpynum.tolist()
        extractor(r_seqs, dt['clusters'], dt["cluster_indices"], target_dir, r_cpynum, uniform_length=True)












# focus = 'pig'
# model = "crbm"
# mfolder = '/mnt/D1/phage_display_analysis/'
# ge2_datatype = {"id": "ge2", "process": "gaps_end", "clusters": 2, "gap_position_indices": [-1, -1], "cluster_indices": [[12, 22], [35, 45]]}
# gm2_datatype = {"id": "gm2", "process": "gaps_middle", "clusters": 2, "gap_position_indices": [2, 16], "cluster_indices": [[12, 22], [35, 45]]}  # based solely off looking at the sequence logos
# ge4_datatype = {"id": "ge4", "process": "gaps_end", "clusters": 4, "gap_position_indices": [-1, -1, -1, -1], "cluster_indices": [[12, 16], [17, 22], [35, 39], [40, 45]]}
# gm4_datatype = {"id": "gm4", "process": "gaps_middle", "clusters": 4, "gap_position_indices": [2, 2, 16, 16], "cluster_indices": [[12, 16], [17, 22], [35, 39], [40, 45]]}
#
# datatype = ge4_datatype  # Change this to change the which dataset is generating files


# focus = 'cov'
# mfolder = '/mnt/D1/sars-cov-2-data/processed/'
# model = "crbm"
# # Fix this up (correct dirs)
# if focus == 'cov':
#     # subdir = 'cov/fasta files/'
#     odir = './cov/'  # out directory
#     # rounds = [f"HanS_R{i}.txt" for i in range(1, 13)] # Unprocessed Files
#     rounds = [f"r{i}" for i in range(1, 13)]  # Processed Files
#
#     datatype_dir = datatype["process"] + f"_{datatype['clusters']}_clusters"
#     if not os.path.isdir(datatype_dir):
#         os.mkdir(f"./{datatype_dir}")
#         os.mkdir(f"./{datatype_dir}/trained_{model}s/")
#
