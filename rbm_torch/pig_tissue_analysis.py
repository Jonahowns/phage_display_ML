from rbm import fasta_read, get_checkpoint, get_beta_and_W, all_weights, RBM
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt





# checkp = "/home/jonah/PycharmProjects/phage_display_ML/rbm_torch/ray_results/tune_pbt_rbm/train_rbm_d61f7_00003_3_batch_size=20000,h_num=40,mc_moves=8_2022-02-09_16-41-58/checkpoint_epoch=7-step=7/checkpoint"
# checkp2 = "/home/jonah/PycharmProjects/phage_display_ML/rbm_torch/ray_results/tune_rbm_asha/train_rbm_87421_00001_1_batch_size=20000,h_num=20,mc_moves=4_2022-02-10_09-43-09/checkpoint_epoch=9-step=9/checkpoint"
# checkp3 = "/home/jonah/PycharmProjects/phage_display_ML/rbm_torch/ray_results/tune_rbm_asha/train_rbm_87421_00001_1_batch_size=20000,h_num=20,mc_moves=4_2022-02-10_09-43-09/tb/checkpoints/epoch=8-step=8.ckpt"


# Directory of Stored RBMs
mdir = "/mnt/D1/globus/pig_trained_rbms/"
rounds = ["b3", "n1", "np1", "np2", "np3"]
c1_rounds = [x+"_c1" for x in rounds]
c2_rounds = [x+"_c2" for x in rounds]

# read in name of files (w/o file extension)
def fetch_data(fasta_names, dir=""):
    data_dict = {}
    for x in fasta_names:
        seqs = fasta_read(dir + "/" + x +".fasta", drop_duplicates=True, seq_read_counts=False)
        data = pd.DataFrame({"sequence": seqs})
        data_dict[x] = data
    return data_dict

def get_checkpoint_path(round, version=None):
    ndir = mdir + round + "/"
    if version:
        version_dir = ndir + f"version_{version}/"
    else:   # Get Most recent i.e. highest version number
        v_dirs = glob(ndir + "/*/", recursive=True)
        versions = [int(x[:-1].rsplit("_")[-1]) for x in v_dirs]  # extracted version numbers

        maxv = max(versions)  # get highest version number
        indexofinterest = versions.index(maxv)  # Get index of the highest version
        version_dir = v_dirs[indexofinterest]  # Access directory path of the highest version

    y = glob(version_dir + "checkpoints/*.ckpt", recursive=False)[0]
    return y

# for round in c1_rounds:
#     checkp = get_checkpoint_path(round)
#     rbm = RBM.load_from_checkpoint(checkp)
#     all_weights(rbm, mdir+round+"/"+round+"_weights", 5, 1, 10, 5, "protein")
    

data = fetch_data(c1_rounds, dir="../pig_tissue")
for rid, round in enumerate(c1_rounds):
    if rid > 0:
        break
    checkp = get_checkpoint_path("np3_c1")
    rbm = RBM.load_from_checkpoint(checkp)
    b3seqs, b3likeli = rbm.predict(data["b3_c1"])
    n1seqs, n1likeli = rbm.predict(data["n1_c1"])
    np1seqs, np1likeli = rbm.predict(data["np1_c1"])
    np2seqs, np2likeli = rbm.predict(data["np2_c1"])
    np3seqs, np3likeli = rbm.predict(data["np3_c1"])

fig, axs = plt.subplots(1, 5, sharey=True, tight_layout=True)
axs[0].hist(b3likeli, bins=100)
axs[1].hist(n1likeli, bins=100)
axs[2].hist(np1likeli, bins=100)
axs[3].hist(np2likeli, bins=100)
axs[4].hist(np3likeli, bins=100)
plt.savefig("Test_Likelihood.png")

def output_likelihoods(seqs, likeli, outpath):
    o = open(outpath, 'w')
    for xid, x in enumerate(likeli):
       o.write(seqs[xid]+ ", " +str(x))
    o.close()

output_likelihoods(b3seqs, b3likeli, "b3_from_np3_c1.txt")
output_likelihoods(n1seqs, n1likeli, "n1_from_np3_c1.txt")
output_likelihoods(np1seqs, np1likeli, "np1_from_np3_c1.txt")
output_likelihoods(np2seqs, np2likeli, "np2_from_np3_c1.txt")
output_likelihoods(np3seqs, np3likeli, "np3_from_np3_c1.txt")

# checkp = get_checkpoint_path('b3_c1')
# rbm = RBM.load_from_checkpoint(checkp)
# rbm.AIS()

# rbm.gen_data(N_PT=11, Nchains=20, Lchains=1, Nthermalize=200, update_betas=True)

# rbm._gen_data(200, 5, 1, N_PT=11, batches=1, reshape=False, beta=1,
#                 record_replica=True, record_acceptance=False, update_betas=True, record_swaps=False)

# rbm.gen_data(Nchains=10, Lchains=100, Nthermalize=0, Nstep=1, N_PT=11, beta=1, batches=None, reshape=False, record_replica=False, record_acceptance=False, update_betas=True, record_swaps=False)
# rbm.gen_data(Nchains=10, Lchains=1, Nthermalize=0, beta=0)





# rbm = RBM.load_from_checkpoint(checkp3)
# rbm.prepare_data()
#
# traind = rbm.train_dataloader(init_fields=False)
# vald = rbm.val_dataloader()
#
# for dataloader in [vald, traind]:
#     total = []
#     valdict = {}
#     for i, batch in enumerate(dataloader):
#     #     print(len(batch))
#     #     seqs, tens, weights = batch
#         # if i == 0:
#             # rbm.testing()
#         # rbm.training_step(batch, i)
#         valdict = rbm.validation_step(batch, i)
#         total.append(float(valdict["val_psuedolikelihood"]))
#     print(sum(total)/len(total))
# print("hi")