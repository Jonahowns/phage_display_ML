import numpy as np
import statistics as stats
import sys
from collections import Counter
import subprocess as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import seaborn as sns

aa = ['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
aad = {'-': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12,
       'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}

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
    df = pd.DataFrame({"Sequence": useqs, "Length": ltotal, "Round": roundlist, "Copy Number": copy_number})
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

def prep_data(seqs, lmin=0, lmax=10, cpy_num=0):
    fseqs = []
    affs = []
    for sid, s in enumerate(seqs):
        if lmin <= len(s) <= lmax:
            fseqs.append(s)
            if cpy_num != 0:
                affs.append(cpy_num[sid])
        else:
            continue
    if cpy_num != 0:
        return fseqs, affs
    else:
        return fseqs

def gap_adder(seqs, maxlen):
    nseqs = []
    for seq in seqs:
        nseqs.append(seq.replace("*", "-") + "".join(["-" for i in range(maxlen - len(seq))]))
    return nseqs

def extractor(seqs, cnum, lenindices, outdir, cpy_num):
    for i in range(cnum):
        print(i, lenindices[i][0], lenindices[i][1])
        ctmp, caffs = prep_data(seqs, lenindices[i][0], lenindices[i][1], cpy_num=cpy_num)
        # c_adj = gap_adder(ctmp, lenindices[i][1])
        write_fasta(ctmp, caffs, outdir + '.fasta')


focus = 'cov'
mfolder = '/mnt/D1/sars-cov-2-data/processed/'

# Fix this up (correct dirs)
if focus == 'cov':
    # subdir = 'cov/fasta files/'
    odir = './cov/'  # out directory
    # rounds = [f"HanS_R{i}.txt" for i in range(1, 13)] # Unprocessed Files
    rounds = [f"r{i}" for i in range(1, 13)]  # Processed Files



def initial_report(i):
    seqs = fasta_read(mfolder+rounds[i-1])
    seqs, cpy_num, df = data_prop(seqs, rounds[i-1], outfile=odir+rounds[i-1]+'seq_len_report.txt')
    return df

def extract_data(i, cnum, c_indices):
    seqs = fasta_read(mfolder+rounds[i-1])
    seqs, cpy_num, df = data_prop(seqs, f"r{i}",outfile=odir + rounds[i-1] + 'seq_len_report.txt')
    extractor(seqs, cnum, c_indices, odir + f"r{i}", cpy_num)

# Processing the Files
# for j in range(1, len(rounds)+1):
    # df = initial_report(j)
#     dfs.append(df)
#     extract_data(j, 1, [[40, 40]])




### Nice Violin Plot of Data Lengths
# dfs = []

#
# ultdf = pd.concat(dfs)
#
# seaborn.violinplot(x=ultdf.Round, y=ultdf.Length)
# # fig, ax = plt.subplots(1, 1)
# # ax.violinplot([x for x in ltotal if x < 60], points=200, vert=False, widths=1.1,
# #              showmeans=True, showextrema=True, showmedians=True,
# #              quantiles=[0.05, 0.1, 0.8, 0.9], bw_method=0.5)
# # ax.set_xlabel("Sequence Length")
# plt.savefig("./pig_tissue/data_length_vis.png", dpi=300)



#### Prepare Submission Scripts

if focus == "cov":
    # path is from ProteinMotifRBM/ to /pig_tissue/trained_rbms/
    dest_path = "../cov/trained_rbms/"
    src_path = "../cov/"

    all_data_files = [x + '.fasta' for x in rounds]

    all_rbm_names = rounds

    script_names = ["cov"+str(i) for i in range(len(all_rbm_names))]

    paths_to_data = [src_path + x for x in all_data_files]



def write_submission_scripts(rbmnames, script_names, paths_to_data, destination, hiddenunits, focus, epochs, weights=False):
    # NAME DATA_PATH DESTINATION HIDDEN
    for i in range(len(rbmnames)):
        o = open('./rbm_torch/rbm_train_htc.sh', 'r')
        filedata = o.read()
        o.close()

        if "c"+str(1) in rbmnames[i]: # cluster 1 has 22 visible units
            vis = 22
        elif "c"+str(2) in rbmnames[i]:# cluster 2 has 22 visible units
            vis = 45
        elif "r" in rbmnames[i]:
            vis = 40 # Cov data is 40 Nucleotides

        
        # Replace the Strings we want
        filedata = filedata.replace("NAME", rbmnames[i]+script_names[i])
        filedata = filedata.replace("FOCUS", focus)
        filedata = filedata.replace("MOLECULE", "protein")
        filedata = filedata.replace("DATA_PATH", paths_to_data[i])
        filedata = filedata.replace("DESTINATION", destination)
        filedata = filedata.replace("HIDDEN", str(hiddenunits))
        filedata = filedata.replace("VISIBLE", str(vis))
        filedata = filedata.replace("PARTITION", "htcgpu")
        filedata = filedata.replace("QUEUE", "normal")
        filedata = filedata.replace("GPU_NUM", str(1))
        filedata = filedata.replace("EPOCHS", str(epochs))
        filedata = filedata.replace("WEIGHTS", str(weights))

        with open("./rbm_torch/agave_submit/" + script_names[i], 'w+') as file:
            file.write(filedata)

    if weights:
        focus += "_w"
    with open("./rbm_torch/agave_submit/submit" + focus + ".sh", 'w+') as file:
        file.write("#!/bin/bash\n")
        for i in range(len(script_names)):
            file.write("sbatch " + script_names[i] + "\n")


write_submission_scripts(all_rbm_names, script_names, paths_to_data, dest_path, 20, focus, 200, weights=False)
#
w_script_names = [x+"_w" for x in script_names]
#
write_submission_scripts(all_rbm_names, w_script_names, paths_to_data, dest_path, 20, focus, 200, weights=True)
