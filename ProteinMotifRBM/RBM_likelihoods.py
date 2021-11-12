from __future__ import division, print_function, absolute_import
import sys,os,pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess as sp
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
    afs = np.ndarray((len(alldata)), dtype='int')
    o = open(fastafile, 'r')
    c = 0
    for line in o:
        if line.startswith('>'):
            data = line.split('-')
            afs[c] = (float(data[1]))
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


def all_weights(RBMin, name, rows, columns, h, w, rbm_loaded=False, molecule='rna'):
    if rbm_loaded:
        beta = Proteins_RBM_utils.get_beta(RBMin.weights)
        order = np.argsort(beta)[::-1]
        fig = sequence_logo.Sequence_logo_all(RBMin.weights[order], name=name + '.pdf', nrows=rows, ncols=columns, figsize=(h,w) ,ticks_every=10,ticks_labels_size=10,title_size=12, dpi=400, molecule=molecule)
    else:
        RBM = Proteins_RBM_utils.loadRBM(RBMin)
        beta = Proteins_RBM_utils.get_beta(RBM.weights)
        order = np.argsort(beta)[::-1]
        fig = sequence_logo.Sequence_logo_all(RBM.weights[order], name=name + '.pdf', nrows=rows, ncols=columns, figsize=(h,w) ,ticks_every=10,ticks_labels_size=10,title_size=12, dpi=400, molecule=molecule)


def get_likelihoods(rbm, datafile, outfile):
    data = Proteins_utils.load_FASTA(datafile,drop_duplicates=True, type='dna')
    afs = get_affinities(datafile, data)
    if len(afs) != len(data):
        print('Mismatching Affinities and Likelihoods')
        sys.exit()
    o = open(outfile, 'w')
    for xid, x in enumerate(data):
        if xid % 1000 == 0:
            print("Progress: ", round(xid/len(data)*100))
        l = float(rbm.likelihood(x))
        print(afs[xid], l, file=o)
    print('Wrote:', outfile)
    o.close()


ribofile = 'riboswitch.afa'
filename = 's90.txt'
path = ''
test = 's10.txt'

g15tr = 'r15_train.txt'
g15te = 'r15_test.txt'

gdir = '../GunterAptamers/Selex_Data/v3_fullalign/rbm_d/c0/rbms/'
dest = '../GunterAptamers/Selex_Data/v3_fullalign/rbm_f/'
datap = '../GunterAptamers/Selex_Data/v3_fullalign/'

datathc = '../../THC/v1_r15/'
rbmthc = '../../THC/v1_r15/rbm_d/'
destthc = '../../THC/v1_r15/rbm_f/'

g15p = datathc + 'thc_c0_t.txt'
g15v = datathc + 'thc_c0_v.txt'

cdir = '../Controls/'
crbm = '../Controls/rbm/'

gamdir = '../gam/'
gamrbms = '../gam/rbms/'
gamdest = '../gam/rbm_f/'
gtrain = gamdir + 'c0_t.txt'
gtest = gamdir +'c0_v.txt'

yrbm = '../thrombin_rbm/'
ytrain = yrbm + 's90.txt'
ytest = yrbm + 's10.txt'

make_training = False

if make_training: # Make full training.
   print('This is the Analysis Script.. make better choices!!!')

else:
    data = [[gtrain, gtest], [ytrain, ytest]]
    ranges = [[10, 110, 50], [2, 14, 4]]
    rbmdir = [gamrbms, yrbm]
    dests = [gamdest, yrbm]
    for i in range(0,1,1):
        print(i)
        #Load Data First
        datafiles = data[i]

        #Prep all RBMfiles
        rbmin_tmp = ['c0_' + str(x) + '_h.dat' for x in np.arange(ranges[i][0], ranges[i][1], ranges[i][2])]
        rbmwsq_tmp = [str(x.split('.')[0]) + '_w.dat' for x in rbmin_tmp]
        rbmins = [None]*(2*len(rbmin_tmp))
        rbmins[::2] = rbmin_tmp
        rbmins[1::2] = rbmwsq_tmp

        #Prep all Output Files
        weightout = [str(x.split('.')[0]) + '_weights' for x in rbmins]
        baserbmout = [str(x.split('.')[0]) + '_li.txt' for x in rbmins]
        trainrbmout = [str(x.split('.')[0]) + '_t.txt' for x in baserbmout]
        testrbmout = [str(x.split('.')[0]) + '_v.txt' for x in baserbmout]
        final_routs = [None]*(len(trainrbmout) + len(testrbmout))
        final_routs[::2] = trainrbmout
        final_routs[1::2] = testrbmout
        # print(final_routs)
        
        #RUN IT
        helper_indx = 0
        for rid, rbm in enumerate(rbmins):
            RBM = Proteins_RBM_utils.loadRBM(rbmdir[i] + str(rbm)) ## Alternative: Load previous model.
            # all_weights(RBM, destthc + weightout[rid], 5, 1, 10, 2, rbm_loaded=True, molecule='rna')
            for did, data in enumerate(datafiles):
                get_likelihoods(RBM, data, dests[i] + final_routs[rid+did+helper_indx])
            helper_indx += 1
    
    
    
   
   
   


####################EXTRA#########################

'''
# beta = Proteins_RBM_utils.get_beta(RBM.weights)
# order = np.argsort(beta)[::-1]
# fig = sequence_logo.Sequence_logo_multiple(RBM.weights[order], figsize=(5,2) ,ticks_every=5,ticks_labels_size=10,title_size=12)

top5 = order[[17, 29, 13, 18, 20, 81, 84, 23, 62, 53]]
interest = order[[17, 18, 84, 20, 81, 23]]
t10 = order[[17, 29, 13, 18, 20, 81, 84, 23, 62, 53]]
t20 = order[[50, 39, 28, 63, 85, 94,  8, 68, 21, 80]]
t30 = order[[67, 25, 35, 98, 76, 74, 26, 92, 32, 77]]
t40 = order[[43, 71,  6, 27, 91,  3, 65, 48, 0, 41]]
t50 = order[[82, 95, 30, 51, 93,  5, 88, 40, 15, 90]]
t60 = order[[45, 60, 72, 55, 89,  4, 59, 54, 57, 19]]
t70 = order[[69, 87, 11, 22, 37, 75, 10, 44, 70,  7]]
t80 = order[[14, 33, 31, 52, 97, 46, 49, 73, 78, 96]]
t90 = order[[56, 99, 16, 47,  2,  1, 61, 86, 36, 83]]
t100 = order[[38, 34, 24, 58, 66, 12, 9, 42, 79, 64]]

ALL = [t10, t20, t30, t40, t50, t60, t70, t80, t90, t100]
out = ['/home/jonah/rbm_rna_v1/' + str(10*(i+1)) + 'weights.png' for i in range(10)]
for i in range(10):
    fig = sequence_logo.Sequence_logo_multiple(RBM.weights[ALL[i]], figsize=(10,3) ,ticks_every=5,ticks_labels_size=20,title_size=24)
    plt.close()

###Generating New Sequences
# N_sequences, Nstep = 1000, 10
# datav1,datah1 = RBM.gen_data(Nchains = 100, Lchains = N_sequences/100,Nstep=Nstep,Nthermalize=500)
# data_regular_lowT, _ = Proteins_RBM_utils.gen_data_lowT(RBM, beta = 2, Nchains = 100, Lchains = N_sequences/100,Nstep=Nstep,Nthermalize=500)
# data_regular_zeroT, _ = Proteins_RBM_utils.gen_data_zeroT(RBM, Nchains = 100, Lchains = N_sequences/100,Nstep=Nstep,Nthermalize=500)

# outp = '/home/jonah/rbm_rna_v1/genseqsNormal.txt'
# o = open(outp, 'w')
# for i in range(len(datav1)):
    # norm = convert_genseqs(datav1[i])
    # zero = convert_genseqs(data_regular_zeroT[i])
    # seq = convert_genseqs(data_regular_lowT[i])
    # o.write(''.join(norm)+ '\n')
# o.close()

# print(datav1[1], datah1[1])
'''