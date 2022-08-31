import torch
import torch.nn.functional as F

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

# Project Dependencies
from rbm_torch.utils.utils import BatchNorm2D
# from rbm_torch import crbm_configs
from rbm_torch.models.crbm import CRBM



class ExpCRBM(CRBM):
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
        
        for key in self.hidden_convolution_keys:
            setattr(self, f"batch_norm_{key}", BatchNorm2D(affine=True, momentum=0.1))
        

    ## Compute Input for Hidden Layer from Visible Potts
    def compute_output_v(self, X):  # X is the one hot vector
        outputs = []
        hidden_layer_W = getattr(self, "hidden_layer_W")
        total_weights = hidden_layer_W.sum()
        for iid, i in enumerate(self.hidden_convolution_keys):
            # convx = self.convolution_topology[i]["convolution_dims"][2]
            outputs.append(F.conv2d(X.unsqueeze(1).type(torch.get_default_dtype()), getattr(self, f"{i}_W"), stride=self.convolution_topology[i]["stride"],
                                    padding=self.convolution_topology[i]["padding"],
                                    dilation=self.convolution_topology[i]["dilation"]).squeeze(3))
            outputs[-1] *= hidden_layer_W[iid] / total_weights
            batch_norm = getattr(self, f"batch_norm_{i}")  # get individual batch norm
            outputs[-1] = batch_norm(outputs[-1])  # apply batch norm
            # outputs[-1] *= convx
        return outputs


class pCRBM(CRBM):
    def __init__(self, config, precision="double", debug=False):
        super().__init__(config, precision=precision, debug=debug)

    def training_step_CD_free_energy(self, batch, batch_idx):
        seqs, one_hot, fitness_targets = batch

        fitness_targets = fitness_targets.to(torch.get_default_dtype())
        # if self.meminfo:
        #     print("GPU Allocated Training Step Start:", torch.cuda.memory_allocated(0))

        # forward function of CRBM
        V_neg_oh, h_neg, V_pos_oh, h_pos = self(one_hot)

        # print("GPU Allocated After Forward:", torch.cuda.memory_allocated(0))
        free_energy = self.free_energy(V_pos_oh)
        F_v = (free_energy * fitness_targets).sum() / fitness_targets.sum()  # free energy of training data
        F_vp = (self.free_energy(V_neg_oh) * fitness_targets).sum() / fitness_targets.sum()  # free energy of gibbs sampled visible states
        cd_loss = F_v - F_vp

        # correlation coefficient between free energy and fitness values
        vx = -1 * (free_energy - torch.mean(free_energy))  # multiply be negative one so lowest free energy vals get paired with the highest copy number/fitness values
        vy = fitness_targets - torch.mean(fitness_targets)

        pearson_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + 1e-6) * torch.sqrt(torch.sum(vy ** 2) + 1e-6))
        pearson_loss = (1 - pearson_correlation) * (self.current_epoch/self.epochs + 1.5) * 25

        # Regularization Terms
        reg1 = self.lf/(2 * self.v_num * self.q) * getattr(self, "fields").square().sum((0, 1))
        reg2 = torch.zeros((1,), device=self.device)
        reg3 = torch.zeros((1,), device=self.device)
        for iid, i in enumerate(self.hidden_convolution_keys):
            W_shape = self.convolution_topology[i]["weight_dims"]  # (h_num,  input_channels, kernel0, kernel1)
            x = torch.sum(torch.abs(getattr(self, f"{i}_W")), (3, 2, 1)).square()
            reg2 += x.mean() * self.l1_2 / (2*W_shape[1]*W_shape[2]*W_shape[3])
            reg3 += self.ld / ((getattr(self, f"{i}_W").abs() - getattr(self, f"{i}_W").squeeze(1).abs()).abs().sum((1, 2, 3)).mean() + 1)

        crbm_loss = (cd_loss + reg1 + reg2 + reg3) * (1.5 - self.current_epoch/self.epochs)

        # Calculate Loss
        loss = crbm_loss + pearson_loss

        logs = {"loss": loss,
                "free_energy_diff": cd_loss.detach(),
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
        pearson_corr = torch.stack([x["train_pearson_corr"] for x in outputs]).mean()
        pearson_loss = torch.stack([x["train_pearson_loss"] for x in outputs]).mean()
        # pseudo_likelihood = torch.stack([x['train_pseudo_likelihood'] for x in outputs]).mean()

        self.logger.experiment.add_scalars('Regularization', {'Field Reg':field_reg,
                                                              'Weight Reg': weight_reg,
                                                              'Distance Reg': distance_reg}, self.current_epoch)

        self.logger.experiment.add_scalars("Loss", {"Total": avg_loss,
                                                    "CD_Loss": avg_dF,
                                                    "Pearson Loss": pearson_loss}, self.current_epoch)

        self.logger.log_metrics({"train_pearson_corr": pearson_corr, "train_free_energy": free_energy}, self.current_epoch)

        for name, p in self.named_parameters():
            self.logger.experiment.add_histogram(name, p.detach(), self.current_epoch)

    def validation_step(self, batch, batch_idx):
        seqs, one_hot, fitness_targets = batch

        fitness_targets = fitness_targets.to(torch.get_default_dtype())

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
            "val_pearson_corr": pearson_correlation.detach()
        }

        # logging on step, for whatever reason allocates 512 bytes on gpu after every epoch.
        self.log("ptl/val_free_energy", batch_out["val_free_energy"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/val_pearson_corr", batch_out["val_pearson_corr"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #
        return batch_out

    def validation_epoch_end(self, outputs):
        avg_fe = torch.stack([x['val_free_energy'] for x in outputs]).mean()
        avg_pearson = torch.stack([x['val_pearson_corr'] for x in outputs]).mean()
        self.logger.log_metrics({"val_free_energy": avg_fe, "val_pearson_corr": avg_pearson}, self.current_epoch)




# if __name__ == '__main__':
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