# Script for Training RBMs using the provided RBM software
from __future__ import division, print_function, absolute_import
import sys,os,pickle
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import subprocess as sp
import argparse
sys.path.append('RBM/')
sys.path.append('utilities/')

try:
    import rbm
except:
    print('Compiling cy_utilities first') # the RBM package contains cython files that must be compiled first.
    curr_dir = os.getcwd()
    os.chdir('RBM/')
    sp.call("python setup.py build_ext --inplace", shell=True)
    # get_ipython().system('python setup.py build_ext --inplace')
    print('Compilation done')
    os.chdir(curr_dir)
    import rbm

import Proteins_utils, Proteins_RBM_utils, utilities, plots_utils, sequence_logo

key = {0:'A', 1:'C', 2:'G', 3:'T', 3:'U', 4:'-'}
keyr = {'A':0, 'C':1, 'G':2, 'T':3, 'U':3, '-':4}

rkey = {0:'A', 1:'C', 2:'G', 3:'U', 4:'-'}
rkeyr = {'A':0, 'C':1, 'G':2, 'U':3, '-':4}

dkey = {0:'A', 1:'C', 2:'G', 3:'T', 4:'-'}
dkeyr = {'A':0, 'C':1, 'G':2, 'T':3, '-':4}

def convert_genseqs(array):
    seq = []
    for x in array:
        seq.append(key[int(x)])
    return seq

def deconvert(array):
    seq = []
    for x in array:
        seq.append(key[int(x)])
    return ''.join(seq)

def print_weights(w8s, fp):
    o = open(fp, 'w')
    for i in range(len(w8s)):
        print("HIDDEN NODE", i+1, file=o)
        for j in range(len(w8s[i])):
            print("VISIBLE NODE", j+1, file=o)
            print(w8s[i][j], file=o)
    o.close()

def get_affinities(fastafile, alldata):
    afs = np.ndarray((len(all_data)), dtype='int')
    o = open(fastafile, 'r')
    c = 0
    for line in o:
        if line.startswith('>'):
            data = line.split('-')
            afs[c] = (float(data[1].rstrip()))
            c += 1
    o.close()
    return afs

def output_likelihoods(RBMin, data, outpath):
    RBM = Proteins_RBM_utils.loadRBM(RBMin)
    o = open(outpath, 'w')
    for xid, x in enumerate(data):
        if xid % 10000 == 0:
            print("Progress: ", xid, 'of', len(data))
        l = float(RBM.likelihood(x))
        print(deconvert(x), round(l,4), file=o)
    o.close()

def all_weights(RBMin, name, rows, columns, h, w, molecule='rna'):
    RBM = Proteins_RBM_utils.loadRBM(RBMin)
    beta = Proteins_RBM_utils.get_beta(RBM.weights)
    order = np.argsort(beta)[::-1]
    fig = sequence_logo.Sequence_logo_all(RBM.weights[order], name=name + '.pdf', nrows=rows, ncols=columns, figsize=(h,w) ,ticks_every=10,ticks_labels_size=10,title_size=12, dpi=400, molecule=molecule)


# parse arguments
parser = argparse.ArgumentParser(description="RBM Training on Provided Dataset")
parser.add_argument('RBMname', type=str, help="Name of Trained RBM, saved to destination/RBMname")
parser.add_argument('dataset', type=str, help="Relative Path + File Name of Sequence Data")
parser.add_argument('destination', type=str, help="Directory where RBM will be saved")
parser.add_argument('hiddenunits', type=str, help="Number of hidden units to use")
args = parser.parse_args()

# load and shuffle data
all_data = Proteins_utils.load_FASTA(args.dataset, drop_duplicates=True, type='protein')
nv = len(all_data[0])  # Number of sites in alignment
seed = utilities.check_random_state(69)
permutation = np.argsort(seed.rand(all_data.shape[0]))
# if using affinities as weights
# affs = get_affinities(g15p, all_data)
# affs = affs[permutation]
all_data = all_data[permutation] # Shuffle data.


dest_dir = args.destination
start_dir, _ = os.path.split(args.dataset)


#mu = utilities.average(all_data,c=4,weights=all_weights)
#sequence_logo.Sequence_logo(mu,ticks_every=5);

#PARAMETERS
make_training = True
n_h = int(args.hiddenunits)
n_v = nv # Number of visible units; = # sites in alignment.
visible = 'Potts' # Nature of visible units potential. Here, Potts states...
n_cv = 21 # With n_cv = 21 colors (all possible amino acids + gap)
n_cgv = 5 #MSA of DNA/RNA w/ gap
hidden = 'dReLU' # Nature of hidden units potential. Here, dReLU potential.
#seed = 0 # Random seed (optional)

# Directory RBM will be saved to
if dest_dir[-1] == '/':
    out = dest_dir + args.RBMname
else:
    out = dest_dir + "/" + args.RBMname

if make_training: # Make full training.
    batch_size = 300  # Size of mini-batches (and number of Markov chains used). Default: 100. Value for RBM shown in paper: 300
    n_iter = 1  # Number of epochs. Value for RBM shown in paper: 6000
    learning_rate = 0.1  # Initial learning rate (default: 0.1). Value for RBM shown in paper: 0.1
    decay_after = 0.5  # Decay learning rate after 50% of iterations (default: 0.5). Value for RBM shown in paper: 0.5
    l1b = 0.25  # L1b regularization. Default : 0. Value for RBM shown in paper: 0.25
    N_MC = 10  # Number of Monte Carlo steps between each update. Value for RBM shown in paper: 10

    RBM = rbm.RBM(visible=visible, hidden=hidden, n_v=n_v, n_h=n_h, n_cv=n_cv)
    RBM.fit(all_data, batch_size=batch_size, n_iter=n_iter, l1b=l1b, N_MC=N_MC, decay_after=decay_after, verbose=0)
    Proteins_RBM_utils.saveRBM(out, RBM)

else:
    print('This is literally the training script.. what are you doing?')


#############IMPORT AND MANAGE WEIGHTS
# rdata = Proteins_utils.load_FASTA(path+ribofile, drop_duplicates=True, type='rna')
# all_data = Proteins_utils.load_FASTA(path+filename,drop_duplicates=True, type='dna')
# test_data = Proteins_utils.load_FASTA(path+test,drop_duplicates=True)
# all_data = Proteins_utils.load_FASTA(g15p,drop_duplicates=True, type='dna')
# affs = get_affinities(g15p, all_data)
# seed = utilities.check_random_state(0)
# permutation = np.argsort(seed.rand(all_data.shape[0]))
# affs = affs[permutation]
# all_data = all_data[permutation] # Shuffle data.

# test = Proteins_utils.load_FASTA(g15v,drop_duplicates=True, type='dna')
# rdata = rdata[permutation]
# print(rdata[0], rdata[-2])
# print(len(rdata[0]))
# print(all_data[0], all_data[-1])
# print(affs[0], affs[-1])
# permutation = np.argsort(seed.rand(test_data.shape[0]))
# test_data = test_data[permutation]

############WEIGHTS
# num_neighbours= Proteins_utils.count_neighbours(all_data)
# all_weights = 1.0/num_neighbours
# # inv_weights = np.asarray(1.0/affs, dtype='float')
# # inv_norm_weights = np.asarray([float(i)/sum(affs) for i in inv_weights], dtype='float')
# # norm_weights = np.asarray([float(i)/sum(affs) for i in affs], dtype='float')
# weights_adj = np.asarray([float(i)/1000. for i in affs], dtype='float')
# # weights_adj = [0, 10, 100]
# # raw_weights = np.asarray(affs, dtype='float')

# RNA Aptamers Attempt 1

# ribofile = 'riboswitch.afa'
# filename = 's90.txt'
# path = ''
# test = 's10.txt'
#
# start = '../GunterAptamers/Selex_Data/v3_fullalign/'
# dest = '../GunterAptamers/Selex_Data/v3_fullalign/rbm_d/c0/rbms/'
# start_thc = '../../THC/v1_r15/'
# dest_thc = '../../THC/v1_r15/rbm_d/'
# cdest = '../Controls/rbm/'
# cstart = '../Controls/'
#
# gstart = '../gam/'
# gdest = '../gam/rbms/'
# g15p = gstart + 'c0_t.txt'
# g15p1 = start + 'c0_train.txt'
# g15v = gdir + 'v2_c1_test.txt'

