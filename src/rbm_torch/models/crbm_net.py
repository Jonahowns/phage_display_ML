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
from rbm_torch.utils.utils import Categorical


class FitnessPredictor(LightningModule):
    def __init__(self, crbm_checkpoint_file, network_config, debug=False):
        super().__init__()

        self.crbm = self.load_crbm(crbm_checkpoint_file)

        self.network_epochs = network_config["network_epochs"]

        if debug:
            self.crbm.worker_num = 0

        self.crbm.fasta_file = network_config["fasta_file"]
        self.crbm.weights = network_config["weights"]
        self.crbm.setup()


        # classifier optimization options
        self.network_lr = network_config["network_lr"]
        self.network_dr = network_config["network_dr"]
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
        self.fcns = network_config["network_layers"]

        network = []

        fcn_size = [linear_size]
        if self.fcns > 1:
            fcn_size += [linear_size//i for i in range(2, self.fcns+1, 1)]

            for i in range(self.fcns-1):
                network.append(nn.Dropout(self.network_dr))
                network.append(nn.Linear(fcn_size[i], fcn_size[i+1], dtype=torch.get_default_dtype()))
                network.append(nn.BatchNorm1d(fcn_size[i+1]))
                network.append(nn.LeakyReLU())

        network.append(nn.Linear(fcn_size[-1], 1, dtype=torch.get_default_dtype()))
        network.append(nn.Sigmoid())

        # network = [
        #     nn.Linear(linear_size, linear_size//3, dtype=torch.get_default_dtype()),
        #     nn.LeakyReLU(),
        #     nn.Linear(linear_size//3, linear_size // 2, dtype=torch.get_default_dtype()),
        #     nn.LeakyReLU(),
        #     nn.Linear(linear_size//2, 1, dtype=torch.get_default_dtype()),
        #     nn.Sigmoid()
        # ]

        self.net = nn.Sequential(*network)
        self.netloss = nn.L1Loss(size_average=None, reduce=None, reduction='sum')
        self.save_hyperparameters()

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
            # h_pos = self.crbm.sample_from_inputs_h(self.crbm.compute_output_v(input))
            h_pos = self.crbm.compute_output_v(input)
        h_flattened = torch.cat([torch.flatten(x, start_dim=1) for x in h_pos], dim=1)
        return self.net(h_flattened)

    def predict(self, X):
        # Read in data
        reader = Categorical(X, self.crbm.q, weights=None, max_length=self.crbm.v_num, molecule=self.crbm.molecule, device=self.device, one_hot=True)
        data_loader = torch.utils.data.DataLoader(
            reader,
            batch_size=self.crbm.batch_size,
            num_workers=self.crbm.worker_num,  # Set to 0 if debug = True
            pin_memory=self.crbm.pin_mem,
            shuffle=False
        )
        self.eval()
        with torch.no_grad():
            fitness_vals = []
            for i, batch in enumerate(data_loader):
                seqs, one_hot, seq_weights = batch
                fitness_vals += self(one_hot).squeeze(1).detach().tolist()

        return X.sequence.tolist(), fitness_vals


class CRBM_net(CRBM):
    """ Pytorch Lightning Module of Cnvolutional Restricted Boltzmann Machine

    Parameters
    ----------
    config: dictionary,
        sets mandatory variables for model to run
    precision: str, optional, default="double"
        sets default dtype of torch tensors as torch.float32 or torch.float64
    debug: bool, optional, default=False
        sets dataworker number to 1, enables use of debugger during model training
        if False, tensor information won't be visible in debugger

    Notes
    -----
    Description of Mandatory Config Keys and their accepted values
        "fasta_file": str or list of strs,
        "h_num": int, number of hidden nodes
        "v_num": int, number of visible nodes, should be equal to sequence length of data
        "q": int, number of values each visible node can take on, ex. for dna with values {A, C, G, T, -}, q is 5
        "mc_moves": int, number of times each layer is sampled during each step
        "batch_size": int, number of sequences in a batch
        "epochs": int, number of iterations to run the model
        "molecule": str, type of sequence data can be {"dna", "rna", "protein"}
        "sample_type": str, {"gibbs", "pt", "pcd"}
        "loss_type": str, {"free_energy", "energy"}
        "optimizer": str, {"AdamW", "SGD", "Adagrad"}
        "lr": float, learning rate of model
        "l1_2": float, l1^2 penalty on the sum of the absolute value of the weights
        "lf": float, penalty on the sum of the absolute value of the visible biases 'fields'
        "seed": int, controls tensor generating random values for reproducable behavior



    Description of Optional Config Keys and their accepted values
        "data_worker_num": int, default=multiprocessing.cpu_count()
        "sequence_weights": str, torch.tensor, or np.array
            str values: "fasta", use values in fasta file
                        other str, name of a weight file in same directory as provided fasta file

        "lr_final"
    """


    def __init__(self, config, precision="double", debug=False):
        super().__init__(config, precision=precision, debug=debug)

        assert self.loss_type in ['free_energy']
        assert self.sample_type in ['gibbs']

        input_size = (config["batch_size"], self.v_num, self.q)

        test_output = self.compute_output_v(torch.rand(input_size, device=self.device, dtype=torch.get_default_dtype()))

        # fully flatten each and concatenate along 0 dim
        out_flat = [torch.flatten(x, start_dim=1) for x in test_output]
        full_out = torch.cat(out_flat, dim=1)

        linear_size = full_out.shape[1]
        self.linear_size = linear_size

        self.fcns = config["fully_connected_layers"]
        self.fcn_dr = config["fcn_dropout"]

        network = []

        fcn_size = [linear_size]
        if self.fcns > 1:
            fcn_size += [linear_size // i for i in range(2, self.fcns + 1, 1)]

            for i in range(self.fcns - 1):
                network.append(nn.Dropout(self.fcn_dr))
                network.append(nn.Linear(fcn_size[i], fcn_size[i + 1], dtype=torch.get_default_dtype()))
                network.append(nn.BatchNorm1d(fcn_size[i + 1]))
                network.append(nn.LeakyReLU())

        network.append(nn.Linear(fcn_size[-1], 1, dtype=torch.get_default_dtype()))
        network.append(nn.Sigmoid())

        if config["fcn_loss"] == "l1":
            self.netloss = nn.L1Loss(size_average=None, reduce=None, reduction='mean')
        elif config["fcn_loss"] == "mse":
            self.netloss = nn.MSELoss(size_average=None, reduce=None, reduction='mean')

        self.net = nn.Sequential(*network)

        torch.autograd.set_detect_anomaly(True)
        self.save_hyperparameters()


    def training_step_CD_free_energy(self, batch, batch_idx):
        seqs, one_hot, fitness_targets = batch

        fitness_targets = fitness_targets.to(torch.get_default_dtype())
        # if self.meminfo:
        #     print("GPU Allocated Training Step Start:", torch.cuda.memory_allocated(0))

        # forward function of CRBM
        V_neg_oh, h_neg, V_pos_oh, h_pos = self(one_hot)

        # print("GPU Allocated After Forward:", torch.cuda.memory_allocated(0))
        free_energy = self.free_energy(V_pos_oh)
        F_v = free_energy.sum() / V_pos_oh.shape[0]  # free energy of training data
        F_vp = self.free_energy(V_neg_oh).sum() / V_neg_oh.shape[0]  # free energy of gibbs sampled visible states
        cd_loss = F_v - F_vp

        # correlation coefficient between free energy and fitness values
        vx = -1 * (free_energy - torch.mean(free_energy))  # multiply be negative one so lowest free energy vals get paired with the highest copy number/fitness values
        vy = fitness_targets - torch.mean(fitness_targets)

        pearson_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + 1e-6) * torch.sqrt(torch.sum(vy ** 2) + 1e-6))
        pearson_loss = (1 - pearson_correlation) * (self.current_epoch/self.epochs + 1) * 5

        # Regularization Terms
        reg1 = self.lf/(2 * self.v_num * self.q) * getattr(self, "fields").square().sum((0, 1))
        reg2 = torch.zeros((1,), device=self.device)
        reg3 = torch.zeros((1,), device=self.device)
        for iid, i in enumerate(self.hidden_convolution_keys):
            W_shape = self.convolution_topology[i]["weight_dims"]  # (h_num,  input_channels, kernel0, kernel1)
            x = torch.sum(torch.abs(getattr(self, f"{i}_W")), (3, 2, 1)).square()
            reg2 += x.mean() * self.l1_2 / (2*W_shape[1]*W_shape[2]*W_shape[3])
            reg3 += self.ld / ((getattr(self, f"{i}_W").abs() - getattr(self, f"{i}_W").squeeze(1).abs()).abs().sum((1, 2, 3)).mean() + 1)

        # Network Loss on hidden unit input
        h_flattened = torch.cat([torch.flatten(x, start_dim=1) for x in self.compute_output_v(V_pos_oh)], dim=1)
        preds = self.net(h_flattened)

        raw_mse_loss = self.netloss(preds.squeeze(1), fitness_targets)
        net_loss = 10 * self.current_epoch/self.epochs * raw_mse_loss

        crbm_loss = (cd_loss + reg1 + reg2 + reg3) * (1.5 - self.current_epoch/self.epochs)

        # Calculate Loss
        loss = crbm_loss + net_loss + pearson_loss

        logs = {"loss": loss,
                "free_energy_diff": cd_loss.detach(),
                "train_mse_loss": net_loss.detach(),
                "train_free_energy": F_v.detach(),
                "train_pearson_corr": pearson_correlation.detach(),
                "train_pearson_loss": pearson_loss.detach(),
                "field_reg": reg1.detach(),
                "weight_reg": reg2.detach(),
                "distance_reg": reg3.detach()
                }

        self.log("ptl/train_free_energy", logs["train_free_energy"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_pearson_corr", logs["train_pearson_corr"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_fitness_mse", net_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # if self.meminfo:
        #     print("GPU Allocated Final:", torch.cuda.memory_allocated(0))

        return logs

    def training_epoch_end(self, outputs):
        # These are detached
        avg_loss = torch.stack([x['loss'].detach() for x in outputs]).mean()
        avg_dF = torch.stack([x["free_energy_diff"] for x in outputs]).mean()
        field_reg = torch.stack([x["field_reg"] for x in outputs]).mean()
        weight_reg = torch.stack([x["weight_reg"] for x in outputs]).mean()
        distance_reg = torch.stack([x["distance_reg"] for x in outputs]).mean()
        free_energy = torch.stack([x["train_free_energy"] for x in outputs]).mean()
        net_loss = torch.stack([x["train_mse_loss"] for x in outputs]).mean()
        pearson_corr = torch.stack([x["train_pearson_corr"] for x in outputs]).mean()
        pearson_loss = torch.stack([x["train_pearson_loss"] for x in outputs]).mean()
        # pseudo_likelihood = torch.stack([x['train_pseudo_likelihood'] for x in outputs]).mean()

        self.logger.experiment.add_scalars('Regularization', {'Field Reg':field_reg,
                                                              'Weight Reg': weight_reg,
                                                              'Distance Reg': distance_reg}, self.current_epoch)

        self.logger.experiment.add_scalars("Loss", {"Total": avg_loss,
                                                    "CD_Loss": avg_dF,
                                                    "Train Fitness MSE": net_loss,
                                                    "Pearson Loss": pearson_loss}, self.current_epoch)

        self.logger.log_metrics({"train_pearson_corr": pearson_corr, "train_fitness_mse": net_loss, "train_free_energy": free_energy}, self.current_epoch)

        for name, p in self.named_parameters():
            self.logger.experiment.add_histogram(name, p.detach(), self.current_epoch)


    def validation_step(self, batch, batch_idx):
        seqs, one_hot, fitness_targets = batch

        fitness_targets = fitness_targets.to(torch.get_default_dtype())

        h_input = self.compute_output_v(one_hot)
        h_flattened = torch.cat([torch.flatten(x, start_dim=1) for x in h_input], dim=1)
        preds = self.net(h_flattened)

        net_loss = self.netloss(preds.squeeze(1), fitness_targets)



        # pseudo_likelihood = (self.pseudo_likelihood(one_hot) * seq_weights).sum() / seq_weights.sum()
        free_energy = self.free_energy(one_hot)
        free_energy_avg = free_energy.sum() / one_hot.shape[0]

        # correlation coefficient between free energy and fitness values
        vx = -1*(free_energy - torch.mean(free_energy))  # multiply be negative one so lowest free energy vals get paired with the highest copy number/fitness values
        vy = fitness_targets - torch.mean(fitness_targets)

        pearson_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + 1e-6) * torch.sqrt(torch.sum(vy ** 2) + 1e-6))

        # pearson_loss = 1 - pearson_correlation

        batch_out = {
            # "val_pseudo_likelihood": pseudo_likelihood.detach()
            "val_free_energy": free_energy_avg.detach(),
            "val_fitness_mse": net_loss.detach(),
            "val_pearson_corr": pearson_correlation.detach()
        }

        # logging on step, for whatever reason allocates 512 bytes on gpu after every epoch.
        self.log("ptl/val_free_energy", batch_out["val_free_energy"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/val_fitness_mse", batch_out["val_fitness_mse"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/val_pearson_corr", batch_out["val_pearson_corr"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #
        return batch_out


    def validation_epoch_end(self, outputs):
        avg_fe = torch.stack([x['val_free_energy'] for x in outputs]).mean()
        avg_loss = torch.stack([x['val_fitness_mse'] for x in outputs]).mean()
        avg_pearson = torch.stack([x['val_pearson_corr'] for x in outputs]).mean()
        self.logger.log_metrics({"val_free_energy": avg_fe, "val_fitness_mse": avg_loss, "val_pearson_corr": avg_pearson}, self.current_epoch)
        # self.logger.experiment.add_scalar("Validation Free Energy", avg_fe, self.current_epoch)
        # self.logger.experiment.add_scalar("Validation MSE Loss", avg_loss, self.current_epoch)

    def on_before_zero_grad(self, optimizer):
        with torch.no_grad():
            for key in self.hidden_convolution_keys:
                for param in ["gamma+", "gamma-"]:
                    getattr(self, f"{key}_{param}").data.clamp_(0.05, 1.0)
                for param in ["theta+", "theta-"]:
                    getattr(self, f"{key}_{param}").data.clamp_(0.0, 1.0)

    def predict(self, X):
        # Read in data
        reader = Categorical(X, self.q, weights=None, max_length=self.v_num, molecule=self.molecule, device=self.device, one_hot=True)
        # reader.data.to(self.device)
        data_loader = torch.utils.data.DataLoader(
            reader,
            batch_size=self.batch_size,
            num_workers=self.worker_num,  # Set to 0 if debug = True
            pin_memory=self.pin_mem,
            shuffle=False
        )

        # reader.train_data = reader.train_data.to(device='cuda')
        # reader.train_weights = reader.train_weights.to(device='cuda')
        # reader.train_data.to()


        self.eval()
        with torch.no_grad():
            likelihood = []
            fitness_vals = []
            for i, batch in enumerate(data_loader):
                seqs, one_hot, seq_weights = batch
                one_hot_gpu = one_hot.to(device=self.device)

                likelihood += self.likelihood(one_hot_gpu).detach().tolist()

                h_input = self.compute_output_v(one_hot_gpu)
                h_flattened = torch.cat([torch.flatten(x, start_dim=1) for x in h_input], dim=1)
                preds = self.net(h_flattened)

                fitness_vals += preds.cpu().squeeze(1).tolist()

        return X.sequence.tolist(), likelihood, fitness_vals

    # def on_before_optimizer_step(self, optimizer, optimizer_idx):
    #     param_names = []
    #     for key in self.hidden_convolution_keys:
    #         param_names.append(f"{key}_gamma+")
    #         param_names.append(f"{key}_gamma-")
    #         param_names.append(f"{key}_theta+")
    #         param_names.append(f"{key}_theta-")
    #         param_names.append(f"{key}_W")
    #
    #     for param in param_names:
    #         p = getattr(self, f"{param}")
    #         if True in torch.isnan(p.grad):
    #             print(param)
    #             print(p.grad)


if __name__ == '__main__':
    network_config = {"network_epochs": 2000, "network_lr": 0.000005, "network_lr_final": 0.000005, "network_weight_decay": 0.001,
                      "network_decay_after": 0.75, "network_optimizer": "AdamW", "fasta_file": "./datasets/cov/en_fit.fasta", "weights": "fasta",
                      "network_layers": 1, "network_dr": 0.01}

    import rbm_torch.analysis.analysis_methods as am

    # mdir = "/mnt/D1/globus/cov_trained_crbms/"
    # r = "en_fit_w"
    # #
    # checkp, version_dir = am.get_checkpoint_path(r, rbmdir=mdir)
    # #
    # device = torch.device("cuda")
    # fp = FitnessPredictor(checkp, network_config, debug=False)
    # fp.to(device)
    #
    # logger = TensorBoardLogger("./datasets/cov/trained_crbms/", name="fitness_predictor_en_fit")
    # plt = Trainer(max_epochs=network_config['network_epochs'], logger=logger, accelerator="gpu", devices=1)
    # plt.fit(fp)


    # Evaluate how model did
    # check = "./datasets/cov/trained_crbms/fitness_predictor_en_fit/version_17/checkpoints/epoch=1999-step=16000.ckpt"
    # fp = FitnessPredictor.load_from_checkpoint(check)
    # fp.eval()
    #
    # from rbm_torch.utils.utils import fasta_read
    # import pandas as pd
    # seqs, folds, chars, q = fasta_read("./datasets/cov/en_fit.fasta", "dna", threads=4)
    #
    # df = pd.DataFrame({"sequence": seqs, "copy_num": folds})
    #
    # ss, fit_vals = fp.predict(df)
    #
    # # print(fit_vals)
    #
    # import matplotlib.pyplot as plt
    # plt.scatter(folds, fit_vals, alpha=0.3, s=2)
    # plt.show()

    check = "./datasets/cov/trained_crbms/en_net/version_1/checkpoints/epoch=1499-step=12000.ckpt"

    device = torch.device("cuda")

    cnet = CRBM_net.load_from_checkpoint(check)

    cnet.to(device)
    cnet.eval()

    from rbm_torch.utils.utils import fasta_read
    import pandas as pd
    seqs, folds, chars, q = fasta_read("./datasets/cov/en_fit.fasta", "dna", threads=4)

    df = pd.DataFrame({"sequence": seqs, "copy_num": folds})

    ss, likeli, fit_vals = cnet.predict(df)

    # print(fit_vals)

    import matplotlib.pyplot as plt
    plt.scatter(folds, fit_vals, alpha=0.3, s=2)
    plt.show()





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