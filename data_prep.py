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

# YYC WGQ
cdr3_start_mut = ['YYC', 'Y*C', '*YC', 'YY*']
cdr3_end_mut = ["WGQ", '*GQ', "W*Q", "WG*"]

cdr3_mut_possiblilites = []
for i in range(len(cdr3_start_mut)):
    for j in range(len(cdr3_end_mut)):
        cdr3_mut_possiblilites += [[cdr3_start_mut[i], cdr3_end_mut[j]]]


def error_check(seq, error_possibilities):
    found = False
    for start, end in error_possibilities:
        sloc = seq.find(start)
        eloc = seq.find(end)
        if 0 <= sloc < eloc:   # start must before end
            extracted = seq[sloc+len(start):eloc]
            print(extracted)
            found = True
            break

    if found:
        return extracted
    else:
        return -1


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
    edf = df[df["Length"] < 60]
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
    return useqs, copy_number, edf


def corrector(seqs, mutations, lmin=0, lmax=10):
    fseqs = prep_data(seqs, lmin=lmin, lmax=lmax)
    adjseqs = []
    for f in fseqs:
        nseq = error_check(f, mutations)
        if nseq != -1:
            adjseqs.append(nseq)
    print(f'Corrector Fixed {len(adjseqs)} sequences of {len(fseqs)}')
    return adjseqs


def prep_data(seqs, lmin=0, lmax=10, cpy_num=0):
    fseqs = []
    for sid, s in enumerate(seqs):
        if lmin <= len(s) <= lmax:
            fseqs.append(s)
        else:
            continue
    if cpy_num != 0:
        affs = [cpy_num[xid] for xid, x in enumerate(fseqs)]
        return fseqs, affs
    else:
        return fseqs

# def ensemble_checker(seqfile, *seqs):
#     eseqs = fasta_read(seqfile)
#     results = []
#     for seq in seqs:
#         seqoi = list(seq)
#         highest_sim_score = 0.
#         msseq = seq
#         for eseq in eseqs:
#             es = list(eseq)
#             seq_similarity = 0.
#             for i in range(len(es)):
#                 if seqoi[i] == es[i]:
#                     seq_similarity += 1.
#             seq_similarity /= len(seq)
#             if seq_similarity > highest_sim_score:
#                 highest_sim_score = seq_similarity
#                 msseq = eseq
#         results.append((seq, highest_sim_score, msseq))
#     return results

# def remove_diff_len(fullseqaff):
#     seql = []
#     for key, value in fullseqaff.items():
#         seql.append(len(key))
#     nseql = np.array(seql)
#     mostcommonlen = int(stats.mode(nseql)[0])
#     rm = []
#     for xid, dict in enumerate(fullseqaff.items()):
#         key, value = dict
#         if len(key) != mostcommonlen:
#             rm.append(key)
#     for x in rm:
#         fullseqaff.pop(x)
#     return fullseqaff
#
# def prune_alignment(seqs, simt=0.99, names=[]):
#     final_choice_names = []
#     final_choice_seqs = []
#     for sid, seq in enumerate(seqs):
#         print(sid)
#         append = True
#         seqoi = list(seq)
#         for existseq in final_choice_seqs:
#             es = list(existseq)
#             seq_similarity = 0.
#             for i in range(len(es)):
#                 if seqoi[i] == es[i]:
#                     seq_similarity += 1.
#             seq_similarity /= len(seq)
#             if seq_similarity >= simt:
#                 append = False
#                 break
#         if append:
#             if names:
#                 final_choice_names.append(names[sid])
#             final_choice_seqs.append(seq.upper())
#     print('INFO: reduced length of alignment from %d to %d due to sequence similarity' % (
#     len(seqs), len(final_choice_seqs)), file=sys.stderr),
#     if names:
#         return final_choice_names, final_choice_seqs
#     else:
#         return final_choice_seqs

def gap_adder(seqs, maxlen):
    nseqs = []
    for seq in seqs:
        nseqs.append(seq.replace("*", "-") + "".join(["-" for i in range(maxlen - len(seq))]))
    return nseqs


def extractor(seqs, cnum, lenindices, outdir, cpy_num, add_gaps=True):
    for i in range(cnum):
        print(i, lenindices[i][0], lenindices[i][1])
        c_adj, caffs = prep_data(seqs, lenindices[i][0], lenindices[i][1], cpy_num=cpy_num)
        if add_gaps:
            c_adj = gap_adder(c_adj, lenindices[i][1])
        write_fasta(c_adj, caffs, outdir + '_c' + str(i+1) + '.fasta')


focus = 'pig'
mfolder = '/mnt/D1/phage_display_analysis/'

# Fix this up (correct dirs)
if focus == 'pig':
    subdir = 'pig_tissue/fasta files/'
    odir = './pig_tissue/alignment/'  # out directory
    rounds = ['np1', 'np2', 'np3', 'n1', 'b3']
elif focus == 'invivo':
    subdir = 'in_vivo/fasta_files/'
    odir = './invivo/'  # out directory
    tissues = ['contra', 'heart', 'ipsi', 'spleen']
    injuries = ['acute1', 'acute2', 'chronic1', 'chronic2', 'sham1', 'sham2', 'subacute1', 'subacute2']
    rounds = []
    # very few sequences: sham1_spleen, subacute2_contra
    for i in range(len(injuries)):
        for j in range(len(tissues)):
            if i == 5 and j == 0: # no sham2_contra
                continue
            else:
                rounds.append(injuries[i]+"_"+tissues[j])
    startinglibraries = ['dab1', 'dab2']
elif focus == 'rod':
    subdir = 'rod_microglia/'
    odir = './rod/'  # out directory
    # Combine These
    rounds = ['r1a_r1', 'r1b_r1', 'r1b_r1', 'r1b_r2', 'r2a_r1', 'r2a_r2', 'r3a_r1', 'r3a_r2', 'r3b_r1', 'r3b_r2', 'r4a_r1', 'r4a_r2', 'r4b_r1']


cdrounds = [mfolder + subdir + x + '_cdr3.fasta' for x in rounds]


def initial_report(i):
    seqs = fasta_read(cdrounds[i])
    seqs, cpy_num, df = data_prop(seqs, rounds[i], outfile=odir+rounds[i]+'seq_len_report.txt')
    return df

def extract_data(i, cnum, c_indices, add_gaps=True):
    seqs = fasta_read(cdrounds[i])
    seqs, cpy_num, df = data_prop(seqs, rounds[i], outfile=odir + rounds[i] + 'seq_len_report.txt')
    ecseqs = corrector(seqs, cdr3_mut_possiblilites, lmin=80, lmax=80)
    seqs += ecseqs
    extractor(seqs, cnum, c_indices, odir + rounds[i], cpy_num, add_gaps=add_gaps)


### Nice Violin Plot of Data Lengths
# dfs = []

# get data into new files based on length
# for j in range(len(rounds)):
#     extract_data(j, 1, [[35, 45]], add_gaps=False)

#     df = initial_report(j)
#     dfs.append(df)

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

if focus == "pig":
    # path is from ProteinMotifRBM/ to /pig_tissue/trained_rbms/
    dest_path = "../pig_tissue/trained_rbms/"
    src_path = "../pig_tissue/"

    c1 = [x + '_c1.fasta' for x in rounds]
    c2 = [x + '_c2.fasta' for x in rounds]

    all_data_files = c1+c2

    rbm1 = [x + '_c1' for x in rounds]
    rbm2 = [x + '_c2' for x in rounds]

    all_rbm_names = rbm1 + rbm2

    script_names = ["pig"+str(i) for i in range(len(all_rbm_names))]

    paths_to_data = [src_path + x for x in all_data_files]

elif focus == "invivo":
    # path is from ProteinMotifRBM/agave_sbumit/ to /pig_tissue/trained_rbms/
    dest_path = "../invivo/trained_rbms/"
    src_path = "../invivo/"

    c1 = [x + '_c1.fasta' for x in rounds]
    c2 = [x + '_c2.fasta' for x in rounds]

    all_data_files = c1 + c2

    rbm1 = [x + '_c1' for x in rounds]
    rbm2 = [x + '_c2' for x in rounds]

    all_rbm_names = rbm1 + rbm2

    script_names = ["invivo" + str(i) for i in range(len(all_rbm_names))]

    paths_to_data = [src_path + x for x in all_data_files]


def write_submission_scripts(rbmnames, script_names, paths_to_data, destination, hiddenunits, focus, epochs, weights=False, gaps=True):
    # NAME DATA_PATH DESTINATION HIDDEN
    for i in range(len(rbmnames)):
        o = open('rbm_torch/submission_templates/rbm_train_htc.sh', 'r')
        filedata = o.read()
        o.close()

        if "c"+str(1) in rbmnames[i]: # cluster 1 has 22 visible units
            vis = 22
        elif "c"+str(2) in rbmnames[i]:# cluster 2 has 22 visible units
            vis = 45

        # python rbm_train.sh focus Data_Path Epochs Gpus Weights
        # Replace the Strings we want
        filedata = filedata.replace("NAME", rbmnames[i]+script_names[i])
        filedata = filedata.replace("FOCUS", focus)
        filedata = filedata.replace("DATA_PATH", paths_to_data[i])
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


write_submission_scripts(all_rbm_names, script_names, paths_to_data, dest_path, 20, focus, 200, weights=False, gaps=True)

w_script_names = [x+"_w" for x in script_names]

write_submission_scripts(all_rbm_names, w_script_names, paths_to_data, dest_path, 20, focus, 200, weights=True, gaps=True)
