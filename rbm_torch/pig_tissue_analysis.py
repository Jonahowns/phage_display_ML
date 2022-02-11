from rbm_test import get_checkpoint, get_beta_and_W, all_weights, RBM





checkp = "/home/jonah/PycharmProjects/phage_display_ML/rbm_torch/ray_results/tune_pbt_rbm/train_rbm_d61f7_00003_3_batch_size=20000,h_num=40,mc_moves=8_2022-02-09_16-41-58/checkpoint_epoch=7-step=7/checkpoint"
checkp2 = "/home/jonah/PycharmProjects/phage_display_ML/rbm_torch/ray_results/tune_rbm_asha/train_rbm_87421_00001_1_batch_size=20000,h_num=20,mc_moves=4_2022-02-10_09-43-09/checkpoint_epoch=9-step=9/checkpoint"
checkp3 = "/home/jonah/PycharmProjects/phage_display_ML/rbm_torch/ray_results/tune_rbm_asha/train_rbm_87421_00001_1_batch_size=20000,h_num=20,mc_moves=4_2022-02-10_09-43-09/tb/checkpoints/epoch=8-step=8.ckpt"


rbm = RBM.load_from_checkpoint(checkp3)
rbm.prepare_data()

traind = rbm.train_dataloader(init_fields=False)
vald = rbm.val_dataloader()

for dataloader in [vald, traind]:
    total = []
    valdict = {}
    for i, batch in enumerate(dataloader):
    #     print(len(batch))
    #     seqs, tens, weights = batch
        # if i == 0:
            # rbm.testing()
        # rbm.training_step(batch, i)
        valdict = rbm.validation_step(batch, i)
        total.append(float(valdict["val_psuedolikelihood"]))
    print(sum(total)/len(total))
print("hi")