import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

# Project Dependencies
# from rbm_torch import crbm_configs
from rbm_torch.models.crbm import CRBM
import math
from torch.optim import SGD, AdamW
from pytorch_lightning import LightningModule, Trainer


class FitnessPredictor(LightningModule):
    def __init__(self, crbm_checkpoint_file, network_config, debug=False):
        super().__init__()

        self.crbm = self.load_crbm(crbm_checkpoint_file)

        self.network_epochs = network_config["network_epochs"]

        self.crbm.fasta_file = network_config["fasta_file"]
        self.crbm.weights = network_config["weights"]
        self.crbm.setup()


        # classifier optimization options
        self.network_lr = network_config["network_lr"]
        self.network_lrf = network_config["network_lr_final"]
        self.network_wd = network_config['network_weight_decay']  # Put into weight decay option in configure_optimizer, l2 regularizer
        self.network_decay_after = network_config['network_decay_after']  # hyperparameter for when the lr decay should occur
        network_optimizer = network_config['network_optimizer']  # hyperparameter for when the lr decay should occur

        # optimizer options
        if network_optimizer == "SGD":
            self.network_optimizer = SGD
        elif network_optimizer == "AdamW":
            self.network_optimizer = AdamW

        input_size = (self.crbm.batch_size, self.crbm.v_num, self.crbm.q)

        test_output = self.crbm.compute_output_v(torch.rand(input_size, device=self.device, dtype=torch.get_default_dtype()))

        # fully flatten each and concatenate along 0 dim
        out_flat = [torch.flatten(x, start_dim=1) for x in test_output]
        full_out = torch.cat(out_flat, dim=1)

        linear_size = full_out.shape[1]
        self.linear_size = linear_size

        network = [
            nn.Linear(linear_size, linear_size//3, dtype=torch.get_default_dtype()),
            nn.LeakyReLU(),
            nn.Linear(linear_size//3, linear_size // 2, dtype=torch.get_default_dtype()),
            nn.LeakyReLU(),
            nn.Linear(linear_size//2, 1, dtype=torch.get_default_dtype()),
            nn.Sigmoid()
        ]

        self.net = nn.Sequential(*network)
        self.netloss = nn.MSELoss(size_average=None, reduce=None, reduction='sum')

    def configure_optimizers(self):
        optim = self.network_optimizer(self.parameters(), lr=self.network_lr, weight_decay=self.network_wd)
        decay_gamma = (self.network_lrf / self.network_lr) ** (1 / (self.network_epochs * (1 - self.network_decay_after)))
        decay_milestone = math.floor(self.network_decay_after * self.network_epochs)
        my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=[decay_milestone], gamma=decay_gamma)
        # my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=decay_gamma)
        return optim

    def load_crbm(self, checkpoint_file):
        crbm = CRBM.load_from_checkpoint(checkpoint_file)
        crbm.eval()
        return crbm

    def train_dataloader(self):
        return self.crbm.train_dataloader()

    def val_dataloader(self):
        return self.crbm.val_dataloader()

    def training_step(self, batch, batch_idx):
        seqs, one_hot, fitness_targets = batch
        fitness_targets = fitness_targets.to(torch.get_default_dtype())
        preds = self(one_hot)
        train_loss = self.netloss(preds.squeeze(1), fitness_targets)

        self.log('train_MSE', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        seqs, one_hot, fitness_targets = batch
        fitness_targets = fitness_targets.to(torch.get_default_dtype())
        preds = self(one_hot)
        val_loss = self.netloss(preds.squeeze(1), fitness_targets)

        self.log('val_MSE', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return val_loss

    def forward(self, input):
        with torch.no_grad():
            # attempting to remove crbm_variables from gradient calculation
            h_pos = self.crbm.sample_from_inputs_h(self.crbm.compute_output_v(input))
        h_flattened = torch.cat([torch.flatten(x, start_dim=1) for x in h_pos], dim=1)
        return self.net(h_flattened)





if __name__ == '__main__':
    network_config = {"network_epochs": 100, "network_lr": 0.002, "network_lr_final": 0.00002, "network_weight_decay": 0.001,
                      "network_decay_after": 0.75, "network_optimizer": "AdamW", "fasta_file": "./datasets/cov/net_test_short.fasta", "weights": "fasta"}

    import rbm_torch.analysis.analysis_methods as am

    mdir = "/mnt/D1/globus/cov_trained_crbms/"
    r = "ultra_ut_enrich_1c_st"

    checkp, version_dir = am.get_checkpoint_path(r, rbmdir=mdir)

    fp = FitnessPredictor(checkp, network_config)

    logger = TensorBoardLogger("./datasets/cov/trained_crbms/", name="fitness_predictor_test")
    plt = Trainer(max_epochs=network_config['network_epochs'], logger=logger, accelerator="cpu")
    plt.fit(fp)





    # data_file = '../invivo/sham2_ipsi_c1.fasta'  # cpu is faster
    # large_data_file = '../invivo/chronic1_spleen_c1.fasta' # gpu is faster
    # lattice_data = '../datasets/lattice_proteins_verification/Lattice_Proteins_MSA.fasta'
    # # b3_c1 = "../pig/b3_c1.fasta"
    # # bivalent_data = "./bivalent_aptamers_verification/s100_8th.fasta"
    #
    # config = crbm_configs.lattice_default_config
    # # Edit config for dataset specific hyperparameters
    # config["fasta_file"] = lattice_data
    # config["sequence_weights"] = None
    # config["epochs"] = 100
    # config["sample_type"] = "gibbs"
    # config["l12"] = 25
    # config["lf"] = 10
    # config["ld"] = 5
    # # config["lr"] = 0.006
    # config["seed"] = 38
    #
    # # Training Code
    # rbm = ExpCRBM(config, debug=False)
    # logger = TensorBoardLogger('../tb_logs/', name="conv_lattice_trial_bn")
    # plt = Trainer(max_epochs=config['epochs'], logger=logger, gpus=1)  # gpus=1,
    # plt.fit(rbm)

    # checkp = "./tb_logs/lattice_crbm_exp/version_0/checkpoints/epoch=99-step=199.ckpt"
    # rbm = ExpCRBM.load_from_checkpoint(checkp)
    #
    # all_weights(rbm, name="./tb_logs/lattice_rbm/version_12/affine_batch_norm")

    #
    # # results = gen_data_lowT(rbm, which="marginal")
    # results = gen_data_zeroT(rbm, which="joint")
    # visible, hiddens = results
    #
    # E = rbm.energy(visible, hiddens)
    # print("E", E.shape)





    # import analysis.analysis_methods as am
    # Directory of Stored RBMs
    # mdir = "/mnt/D1/globus/exo_trained_rbms/"
    # rounds = ["exosome_st"]
    # data = ["exosome"]
    #
    # checkp, v_dir = am.get_checkpoint_path(rounds[0], rbmdir=mdir)
    # exo_rbm = RBM.load_from_checkpoint(checkp)
    #
    # exo_rbm.AIS()