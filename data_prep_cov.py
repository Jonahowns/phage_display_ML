import numpy as np
import statistics as stats
import sys
from collections import Counter
import subprocess as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import seaborn as sns
import os

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

def extractor(seqs, cnum, lenindices, outdir, cpy_num, uniform_length=True, position_indx=-1):
    for i in range(cnum):
        print(i, lenindices[i][0], lenindices[i][1])
        ctmp, caffs = prep_data(seqs, lenindices[i][0], lenindices[i][1], cpy_num=cpy_num)
        if uniform_length:
            ctmp = gap_adder(ctmp, lenindices[i][1], position_indx=position_indx)
        write_fasta(ctmp, caffs, outdir + '.fasta')


focus = 'cov'
mfolder = '/mnt/D1/sars-cov-2-data/processed/'
model = "crbm"
# Fix this up (correct dirs)
if focus == 'cov':
    # subdir = 'cov/fasta files/'
    odir = './cov/'  # out directory
    # rounds = [f"HanS_R{i}.txt" for i in range(1, 13)] # Unprocessed Files
    rounds = [f"r{i}" for i in range(1, 13)]  # Processed Files
    ge_datatype = {"process": "gap_end", "clusters": 2, "gap_position_indices": [-1, -1], "cluster_indices": [[12, 22], [35, 45]]}
    gm_datatype = {"process": "gap_middle", "clusters": 2, "gap_position_indices": [2, 16], "cluster_indices": [[12, 22], [35, 45]]}  # based solely off looking at the sequence logos

    datatype = gm_datatype  # Change this to change the which dataset is generating files
    datatype_dir = datatype["process"] + f"_{datatype['clusters']}_clusters"
    if not os.path.isdir(datatype_dir):
        os.mkdir(f"./{datatype_dir}")
        os.mkdir(f"./{datatype_dir}/trained_{model}s/")



def initial_report(i):
    seqs = fasta_read(mfolder+rounds[i-1])
    seqs, cpy_num, df = data_prop(seqs, rounds[i-1], outfile=odir+rounds[i-1]+'seq_len_report.txt')
    return df


def extract_data(i, c_indices, datatypedict, uniform_length=True):
    seqs = fasta_read(mfolder+rounds[i-1])
    c_indices = datatypedict["cluster_indices"]
    cnum = datatypedict["clusters"]
    seqs, cpy_num, df = data_prop(seqs, f"r{i}", outfile=odir + rounds[i-1] + 'seq_len_report.txt')
    extractor(seqs, cnum, c_indices, odir + f"r{i}", cpy_num, uniform_length=uniform_length, position_indx=datatypedict["gap_position_indices"])

# Processing the Files
# for j in range(1, len(rounds)+1):
#     df = initial_report(j)
#     # dfs.append(df)
#     for i in range(datatype["clusters"]):
#         extract_data(j, datatype, uniform_length=True)




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

# For switching b/t datasets that were processed differently



#### Prepare Submission Scripts
#
if focus == "cov":
    # data type variable
    datatype_dir = datatype["process"]+f"_{datatype['clusters']}_clusters"
    # path is from ProteinMotifRBM/ to /pig_tissue/trained_rbms/
    dest_path = f"../cov/{datatype_dir}/trained_{model}s/"
    src_path = f"../cov/"

    all_data_files = [x + '.fasta' for x in rounds]

    all_model_names = rounds

    script_names = ["cov"+str(i+1) for i in range(len(all_model_names))]

    paths_to_data = [src_path + x for x in all_data_files]


# Processing the Files
# for j in range(1, len(rounds)+1):
    # df = initial_report(j)
#     dfs.append(df)
#     extract_data(j, 1, [[40, 40]])

def write_submission_scripts(modelnames, script_names, paths_to_data, destination, hiddenunits, focus, epochs, weights=False, gaps=True):
    # NAME DATA_PATH DESTINATION HIDDEN
    for i in range(len(modelnames)):
        o = open(f'rbm_torch/submission_templates/{model}_train.sh', 'r')
        filedata = o.read()
        o.close()

        if "c"+str(1) in modelnames[i]: # cluster 1 has 22 visible units
            vis = 22
        elif "c"+str(2) in modelnames[i]:# cluster 2 has 22 visible units
            vis = 45
        elif "r" in modelnames[i]:
            vis = 40 # Cov data is 40 Nucleotides

        
        # Replace the Strings we want
        filedata = filedata.replace("NAME", modelnames[i]+script_names[i])
        filedata = filedata.replace("FOCUS", focus)
        filedata = filedata.replace("DATA_PATH", paths_to_data[i])
        filedata = filedata.replace("PARTITION", "sulcgpu2")
        filedata = filedata.replace("QUEUE", "sulcgpu1")
        filedata = filedata.replace("GPU_NUM", str(1))
        filedata = filedata.replace("EPOCHS", str(epochs))
        filedata = filedata.replace("WEIGHTS", str(weights))

        with open(f"./rbm_torch/agave_submit_{model}/" + script_names[i], 'w+') as file:
            file.write(filedata)

    if weights:
        focus += "_w"
    with open(f"./rbm_torch/agave_submit_{model}/submit" + focus + ".sh", 'w+') as file:
        file.write("#!/bin/bash\n")
        for i in range(len(script_names)):
            file.write("sbatch " + script_names[i] + "\n")


write_submission_scripts(all_model_names, script_names, paths_to_data, dest_path, 20, focus, 200, weights=False, gaps=False)
#
w_script_names = [x+"_w" for x in script_names]
#
write_submission_scripts(all_model_names, w_script_names, paths_to_data, dest_path, 20, focus, 200, weights=True, gaps=False)
