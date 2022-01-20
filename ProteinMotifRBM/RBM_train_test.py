from __future__ import division, print_function, absolute_import
import sys,os,pickle
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import subprocess as sp
sys.path.append('RBM/')
sys.path.append('utilities/')
build = True
try:
    import rbm
    if build:
        curr_dir = os.getcwd()
        sp.call("python torch_cython_setup.py build_ext --inplace", shell=True)
        # get_ipython().system('python setup.py build_ext --inplace')
        print('Compilation done')
        os.chdir(curr_dir)
except:
    print('Compiling cy_utilities first') # the RBM package contains cython files that must be compiled first.
    curr_dir = os.getcwd()
    sp.call("python torch_cython_setup.py build_ext --inplace", shell=True)
    # get_ipython().system('python setup.py build_ext --inplace')
    print('Compilation done')
    os.chdir(curr_dir)
    import rbm



import Proteins_utils, Proteins_RBM_utils, utilities, plots_utils, sequence_logo
# import torch
# import torch_utils as utilities
# import torch_seq_logo as sequence_logo
# import torch_seq_utils as Proteins_utils
# import torch_protein_rbm_utils as Proteins_RBM_utils

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
        print(deconvert(x), round(l,4), file=t)
    o.close()

def all_weights(RBMin, name, rows, columns, h, w, molecule='rna'):
    RBM = Proteins_RBM_utils.loadRBM(RBMin)
    beta = Proteins_RBM_utils.get_beta(RBM.weights)
    order = np.argsort(beta)[::-1]
    fig = sequence_logo.Sequence_logo_all(RBM.weights[order], name=name + '.pdf', nrows=rows, ncols=columns, figsize=(h,w) ,ticks_every=10,ticks_labels_size=10,title_size=12, dpi=400, molecule=molecule)




dest = './trial/'
g15p = './trial/v3_c0_all.txt'

pytorch_test = '../../phage_display_ML/invivo/sham2_ipsi_c1.fasta'
# pytorch_test = '../../phage_display_ML/invivo/chronic1_spleen_c1.fasta'

# import data and weights
all_data = Proteins_utils.load_FASTA(pytorch_test, drop_duplicates=True, type='protein')
print("dataset length: " + str(all_data.shape[0]))

# import data and weights
# all_data = Proteins_utils.load_FASTA(g15p,drop_duplicates=True, type='dna')
# affs = get_affinities(g15p, all_data)

# seed = utilities.init_random_state(0)
permutation = np.argsort(np.random.rand(all_data.shape[0]))

# affs = affs[permutation]
# all_data = all_data[permutation] # Shuffle data.

#WEIGHTS
num_neighbours= Proteins_utils.count_neighbours(all_data)
# all_weights = 1.0/num_neighbours
# weights = np.asarray([float(i)/1000. for i in affs], dtype='float')




#mu = utilities.average(all_data,c=4,weights=all_weights)

#sequence_logo.Sequence_logo(mu,ticks_every=5);

#PARAMETERS
make_training = True

n_h = 10
n_v = 22 # Number of visible units; = # sites in alignment.

visible = 'Potts' # Nature of visible units potential. Here, Potts states...
n_cv = 21 # With n_cv = 21 colors (all possible amino acids + gap)
hidden = 'dReLU' # Nature of hidden units potential. Here, dReLU potential.
#seed = 0 # Random seed (optional)


if make_training: # Make full training.

    batch_size = 10000 # Size of mini-batches (and number of Markov chains used). Default: 100. Value for RBM shown in paper: 300
    n_iter = 100 # Number of epochs. Value for RBM shown in paper: 6000
    learning_rate = 0.1 # Initial learning rate (default: 0.1). Value for RBM shown in paper: 0.1
    decay_after = 0.5 # Decay learning rate after 50% of iterations (default: 0.5). Value for RBM shown in paper: 0.5
    l1b = 0.1 # L1b regularization. Default : 0. Value for RBM shown in paper: 0.25
    N_MC = 6 # Number of Monte Carlo steps between each update. Value for RBM shown in paper: 10

    amber = rbm.RBM(visible=visible, hidden=hidden,n_v=n_v, n_h=n_h, n_cv=n_cv, random_state=38)

    amber.AIS(verbose=2)

    # start = time.time()
    # amber.fit(all_data, batch_size=batch_size, n_iter=n_iter, l1b=l1b, N_MC=N_MC, decay_after=decay_after, verbose=1)
    # end = time.time()
    #
    # tim = end-start
    # print("Trial took", tim, "seconds")


    # Proteins_RBM_utils.saveRBM("./long_trial", amber)

else:
    print('This is literally the training script.. what are you doing?')