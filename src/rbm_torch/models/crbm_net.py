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
from rbm_torch.utils.utils import FDS, BatchNorm2D


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

        self.use_pearson = config["use_pearson"]
        self.use_lst_sqrs = config["use_lst_sqrs"]
        self.use_network = config["use_network"]
        self.use_batch_norm = config["use_batch_norm"]


        input_size = (config["batch_size"], self.v_num, self.q)

        if self.use_batch_norm:
            for key in self.hidden_convolution_keys:
                setattr(self, f"batch_norm_{key}", BatchNorm2D(affine=False, momentum=0.1))

        test_output = self.compute_output_v(torch.rand(input_size, device=self.device, dtype=torch.get_default_dtype()))

        # fully flatten each and concatenate along 0 dim
        out_flat = [torch.flatten(x, start_dim=1) for x in test_output]
        full_out = torch.cat(out_flat, dim=1)

        linear_size = full_out.shape[1]
        self.linear_size = linear_size

        if self.use_network:
            self.network_type = config["predictor_network"]
            self.network_layers = config["network_layers"]
            self.network_delay = 0.5
            if "network_delay" in config.keys():
                self.network_delay = config["network_delay"]

            network = []
            network.append(nn.BatchNorm1d(linear_size))
            if self.network_type == "fcn":
                self.fcn_dr = config["fcn_dropout"]
                network.append(nn.BatchNorm1d(linear_size))

                if "fcn_start_size" in config.keys():
                    fcn_start_size = config["fcn_start_size"]
                    network.append(nn.Dropout(self.fcn_dr))
                    network.append(nn.Linear(linear_size, fcn_start_size, dtype=torch.get_default_dtype()))
                    network.append(nn.BatchNorm1d(fcn_start_size))
                    network.append(nn.LeakyReLU())
                    linear_size = config["fcn_start_size"]
                    self.network_layers -= 1

                fcn_size = [linear_size]
                if self.network_layers > 1:
                    fcn_size += [linear_size // i for i in range(2, self.network_layers + 1, 1)]

                    for i in range(self.network_layers - 1):
                        network.append(nn.Dropout(self.fcn_dr))
                        network.append(nn.Linear(fcn_size[i], fcn_size[i + 1], dtype=torch.get_default_dtype()))
                        network.append(nn.BatchNorm1d(fcn_size[i + 1]))
                        network.append(nn.LeakyReLU())

                network.append(nn.Linear(fcn_size[-1], 1, dtype=torch.get_default_dtype()))
                network.append(nn.Sigmoid())

            elif self.network_type == "conv":

                conv_start_channels = config["conv_start_channels"]
                conv_end_channels = config["conv_end_channels"]

                kernel_sizes, output_sizes, channels = [], [], []

                channels = [x for x in range(conv_start_channels, conv_end_channels-1, -int((conv_start_channels-conv_end_channels)//(self.network_layers-1)))]

                input_size = linear_size
                for i in range(self.network_layers):
                    kernel_sizes.append(int(input_size // 1.5))
                    output_sizes.append(input_size - kernel_sizes[-1] + 1)
                    input_size = output_sizes[-1]

                network = []
                for i in range(self.network_layers):
                    if i == 0:
                        network.append(nn.Sequential(
                            nn.Conv1d(1, channels[i], kernel_size=kernel_sizes[i]),
                            nn.BatchNorm1d(channels[i]),
                            # nn.LeakyReLU()))
                            nn.Tanh()))
                    else:
                        network.append(nn.Sequential(
                            nn.Conv1d(channels[i-1], channels[i], kernel_size=kernel_sizes[i]),
                            nn.BatchNorm1d(channels[i]),
                            nn.Tanh()))

                self.final = nn.Sequential(nn.Linear(channels[-1] * output_sizes[-1], 1),
                                           nn.Sigmoid())

            self.net = nn.Sequential(*network)

        self.use_fds = False
        if "fds_kernel" in config.keys():
            self.use_fds = True

            self.fds_kernel = config["fds_kernel"]  # gaussian, laplace, triang
            self.fds_ks = config["fds_ks"]  # 5, FDS kernel size: should be odd number
            self.fds_sigma = config ["fds_sigma"]  # 2, FDS gaussian/laplace kernel sigma
            self.fds_start_update = config["fds_start_update"]  # 0, which epoch to start FDS updating
            self.fds_start_smooth = config["fds_start_smooth"]  # 1, which epoch to start using FDS to smooth features
            self.fds_bucket_num = config["fds_bucket_num"]  # 50, maximum bucket considered for FDS
            self.fds_bucket_start = config["fds_bucket_start"]  # 0, minimum(starting) bucket for FDS
            self.fds_momentum = config["fds_momentum"]  # 0.9, FDS momentum

            self.FDS = FDS(feature_dim=linear_size, bucket_num=self.fds_bucket_num, bucket_start=self.fds_bucket_start,
                       start_update=self.fds_start_update, start_smooth=self.fds_start_smooth,
                       kernel=self.fds_kernel, ks=self.fds_ks, sigma=self.fds_sigma, momentum=self.fds_momentum, device=self.device)


        self.network_loss_type = config["network_loss"]
        if self.network_loss_type == "l1":
            self.netloss = nn.L1Loss(size_average=None, reduce=None, reduction='mean')
        elif self.network_loss_type == "mse":
            self.netloss = nn.MSELoss(size_average=None, reduce=None, reduction='mean')

        # self.net = nn.Sequential(*network)

        # torch.autograd.set_detect_anomaly(True)
        self.save_hyperparameters()

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
            if self.use_batch_norm:
                batch_norm = getattr(self, f"batch_norm_{i}")  # get individual batch norm
                outputs[-1] = batch_norm(outputs[-1])  # apply batch norm
            # outputs[-1] *= convx
        return outputs

    # def on_before_zero_grad(self, optimizer):
    #     with torch.no_grad():
    #         for key in self.hidden_convolution_keys:
    #             for param in ["gamma+", "gamma-"]:
    #                 getattr(self, f"{key}_{param}").data.clamp_(0.05, 1.0)
    #             for param in ["theta+", "theta-"]:
    #                 getattr(self, f"{key}_{param}").data.clamp_(0.0, 1.0)
    #             if self.use_batch_norm:
    #                 getattr(self, f"batch_norm_{key}").bias.clamp(-1.0, 1.0)
    #                 getattr(self, f"batch_norm_{key}").weight.clamp(0.0, 1.5)

    def network(self, x, fitness_targets):
        # Enable FDS
        if self.use_fds:
            if self.current_epoch >= self.fds_start_smooth:
                x = self.FDS.smooth(x, fitness_targets, self.current_epoch)

        if self.network_type == "fcn":
            return self.net(x)
        elif self.network_type == "conv":
            y = self.net(x.unsqueeze(1))
            y = y.view(y.size(0), -1)  # flatten
            return self.final(y)

    # def conv_net(self, x):
    #     # input is of fcn_size[-1]
    #     y = self.conv1(x.unsqueeze(1))
    #     y = self.conv2(y)
    #     y = y.view(y.size(0), -1)  # flatten
    #     pred = self.final(y)
    #     return pred


    def on_train_start(self):
        super().on_train_start()
        # only way I can get this thing on the correct device
        if self.use_fds:
            self.FDS.device = self.device

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


        if self.use_pearson:
            # correlation coefficient between free energy and fitness values
            vx = -1 * (free_energy - torch.mean(free_energy))  # multiply be negative one so lowest free energy vals get paired with the highest copy number/fitness values
            vy = fitness_targets - torch.mean(fitness_targets)

            pearson_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + 1e-6) * torch.sqrt(torch.sum(vy ** 2) + 1e-6))
            pearson_loss = (1 - pearson_correlation) # * (self.current_epoch/self.epochs + 1) * 10


        if self.use_lst_sqrs:
            # Using residuals from least square fitting as loss, not sure why but this seems to not work at all
            a = torch.vstack([-1*free_energy, torch.ones(free_energy.shape[0], device=self.device)]).T
            # sol = torch.linalg.lstsq(a, vy, driver="gels").solution   # This errors out in the backward pass
            sol = torch.linalg.pinv(a) @ fitness_targets
            residuals = torch.abs(fitness_targets - (-1*free_energy * sol[0] + sol[1]))
            lst_sq_loss = residuals.sum()


        # lst_sq_loss = max((self.current_epoch/self.epochs - 0.25), 0.) * residuals.mean()*100

        pearson_multiplier = 1.
        # Regularization Terms
        reg1 = self.lf/(2 * self.v_num * self.q) * getattr(self, "fields").square().sum((0, 1))
        reg2 = torch.zeros((1,), device=self.device)
        reg3 = torch.zeros((1,), device=self.device)
        for iid, i in enumerate(self.hidden_convolution_keys):
            W_shape = self.convolution_topology[i]["weight_dims"]  # (h_num,  input_channels, kernel0, kernel1)
            pearson_multiplier *= W_shape[0] / 2
            x = torch.sum(torch.abs(getattr(self, f"{i}_W")), (3, 2, 1)).square()
            reg2 += x.mean() * self.l1_2 / (2*W_shape[1]*W_shape[2]*W_shape[3])
            reg3 += self.ld / ((getattr(self, f"{i}_W").abs() - getattr(self, f"{i}_W").squeeze(1).abs()).abs().sum((1, 2, 3)).mean() + 1)

        if self.use_network:
            # Network Loss on hidden unit input
            h_flattened = torch.cat([torch.flatten(x, start_dim=1) for x in self.compute_output_v(V_pos_oh)], dim=1)

            preds = self.network(h_flattened, fitness_targets)
            raw_mse_loss = self.netloss(preds.squeeze(1), fitness_targets)
            net_loss = raw_mse_loss  # * max((self.current_epoch/self.epochs - 0.5), 0.)

        # loss calculation
        # if self.use_batch_norm:
        #     cd_loss /= pearson_multiplier


        crbm_loss = (cd_loss + reg1 + reg2 + reg3)  # * (1.2 - self.current_epoch/self.epochs)

        loss = crbm_loss

        if self.use_pearson:
            loss += pearson_loss * pearson_multiplier

        if self.use_lst_sqrs:
            loss += lst_sq_loss

        if self.use_network:
            if self.self.current_epoch/self.epochs > self.network_delay:
                loss += 5 * net_loss * max((self.current_epoch / self.epochs - 0.25), 0.)
            else:
                loss += net_loss*0.



        # Calculate Loss
        # loss = crbm_loss + net_loss * 5 + pearson_loss * pearson_multiplier + lst_sq_loss

        logs = {"loss": loss,
                "free_energy_diff": cd_loss.detach(),
                # f"train_{self.network_loss_type}_loss": net_loss.detach(),
                "train_free_energy": F_v.detach(),
                # "train_pearson_corr": pearson_correlation.detach(),
                # "train_pearson_loss": pearson_loss.detach(),
                # "train_residuals": residuals.mean().detach(),
                "field_reg": reg1.detach(),
                "weight_reg": reg2.detach(),
                "distance_reg": reg3.detach()
                }

        if self.use_pearson:
            logs["train_pearson_corr"] = pearson_correlation.detach()
            logs["train_pearson_loss"] = pearson_loss.detach()
            self.log("ptl/train_pearson_corr", logs["train_pearson_corr"], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if self.use_network:
            logs[f"train_{self.network_loss_type}_loss"] = net_loss.detach()
            self.log("ptl/train_residuals", residuals.sum().detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"ptl/train_fitness_{self.network_loss_type}", raw_mse_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log("ptl/train_free_energy", logs["train_free_energy"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("ptl/train_pearson_corr", logs["train_pearson_corr"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log("ptl/train_residuals", residuals.sum().detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log(f"ptl/train_fitness_{self.network_loss_type}", raw_mse_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

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


        # pseudo_likelihood = torch.stack([x['train_pseudo_likelihood'] for x in outputs]).mean()

        self.logger.experiment.add_scalars('Regularization', {'Field Reg':field_reg,
                                                              'Weight Reg': weight_reg,
                                                              'Distance Reg': distance_reg}, self.current_epoch)

        loss_scalars = {"Total": avg_loss, "CD_Loss": avg_dF}
        metric_scalars = {"train_free_energy": free_energy}

        if self.use_network:
            net_loss = torch.stack([x[f"train_{self.network_loss_type}_loss"] for x in outputs]).mean()
            loss_scalars[f"Train Fitness {self.network_loss_type}"] = net_loss
            metric_scalars[f"Train Fitness {self.network_loss_type}"] = net_loss

        if self.use_pearson:
            pearson_corr = torch.stack([x["train_pearson_corr"] for x in outputs]).mean()
            pearson_loss = torch.stack([x["train_pearson_loss"] for x in outputs]).mean()
            loss_scalars[f"Pearson Loss"] = pearson_loss
            metric_scalars["Train Pearson Corr"] = pearson_corr

        self.logger.experiment.add_scalars("Loss", loss_scalars, self.current_epoch)

        self.logger.log_metrics(metric_scalars, self.current_epoch)

        for name, p in self.named_parameters():
            self.logger.experiment.add_histogram(name, p.detach(), self.current_epoch)


    def validation_step(self, batch, batch_idx):
        seqs, one_hot, fitness_targets = batch

        fitness_targets = fitness_targets.to(torch.get_default_dtype())

        if self.use_network:
            h_input = self.compute_output_v(one_hot)
            h_flattened = torch.cat([torch.flatten(x, start_dim=1) for x in h_input], dim=1)
            preds = self.network(h_flattened, fitness_targets)

            net_loss = self.netloss(preds.squeeze(1), fitness_targets)

        # pseudo_likelihood = (self.pseudo_likelihood(one_hot) * seq_weights).sum() / seq_weights.sum()
        free_energy = self.free_energy(one_hot)
        free_energy_avg = free_energy.sum() / one_hot.shape[0]

        if self.use_pearson:
            # correlation coefficient between free energy and fitness values
            vx = -1*(free_energy - torch.mean(free_energy))  # multiply be negative one so lowest free energy vals get paired with the highest copy number/fitness values
            vy = fitness_targets - torch.mean(fitness_targets)

            pearson_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + 1e-6) * torch.sqrt(torch.sum(vy ** 2) + 1e-6))

        # pearson_loss = 1 - pearson_correlation

        batch_out = {
            # "val_pseudo_likelihood": pseudo_likelihood.detach()
            "val_free_energy": free_energy_avg.detach(),
            # f"val_fitness_{self.network_loss_type}": net_loss.detach(),
            # "val_pearson_corr": pearson_correlation.detach()
        }

        if self.use_network:
            batch_out[f"val_fitness_{self.network_loss_type}"] = net_loss.detach()
            self.log(f"ptl/val_fitness_{self.network_loss_type}", batch_out[f"val_fitness_{self.network_loss_type}"],
                     on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.use_pearson:
            batch_out["val_pearson_corr"] = pearson_correlation.detach()
            self.log("ptl/val_pearson_corr", batch_out["val_pearson_corr"], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)

        # logging on step, for whatever reason allocates 512 bytes on gpu after every epoch.
        self.log("ptl/val_free_energy", batch_out["val_free_energy"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log(f"ptl/val_fitness_{self.network_loss_type}", batch_out[f"val_fitness_{self.network_loss_type}"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("ptl/val_pearson_corr", batch_out["val_pearson_corr"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #
        return batch_out


    def validation_epoch_end(self, outputs):
        avg_fe = torch.stack([x['val_free_energy'] for x in outputs]).mean()
        metrics = {"val_free_energy": avg_fe}

        if self.use_network:
            avg_loss = torch.stack([x[f'val_fitness_{self.network_loss_type}'] for x in outputs]).mean()
            metrics[f"val_fitness_{self.network_loss_type}"] = avg_loss

        if self.use_pearson:
            avg_pearson = torch.stack([x['val_pearson_corr'] for x in outputs]).mean()
            metrics["val_pearson_corr"] = avg_pearson

        self.logger.log_metrics(metrics, self.current_epoch)
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
                preds = self.network(h_flattened, seq_weights)

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
    network_config = {"network_epochs": 1000, "network_lr": 0.000005, "network_lr_final": 0.000005, "network_weight_decay": 0.001,
                      "network_decay_after": 0.75, "network_optimizer": "AdamW", "fasta_file": "./datasets/cov/en_fit.fasta", "weights": "fasta",
                      "network_layers": 2, "network_dr": 0.01}

    import rbm_torch.analysis.analysis_methods as am

    mdir = "/mnt/D1/globus/cov_trained_crbms/"
    r = "pcrbm_en_net"
    #
    checkp, version_dir = am.get_checkpoint_path(r, rbmdir=mdir)
    #
    device = torch.device("cuda")
    fp = FitnessPredictor(checkp, network_config, debug=False)
    fp.to(device)

    logger = TensorBoardLogger("./datasets/cov/trained_crbms/", name="fitness_predictor_pcrbm_en_fit")
    plt = Trainer(max_epochs=network_config['network_epochs'], logger=logger, accelerator="gpu", devices=1)
    plt.fit(fp)


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

    # check = "./datasets/cov/trained_crbms/en_net/version_1/checkpoints/epoch=1499-step=12000.ckpt"
    #
    # device = torch.device("cuda")
    #
    # cnet = CRBM_net.load_from_checkpoint(check)
    #
    # cnet.to(device)
    # cnet.eval()
    #
    # from rbm_torch.utils.utils import fasta_read
    # import pandas as pd
    # seqs, folds, chars, q = fasta_read("./datasets/cov/en_fit.fasta", "dna", threads=4)
    #
    # df = pd.DataFrame({"sequence": seqs, "copy_num": folds})
    #
    # ss, likeli, fit_vals = cnet.predict(df)
    #
    # # print(fit_vals)
    #
    # import matplotlib.pyplot as plt
    # plt.scatter(folds, fit_vals, alpha=0.3, s=2)
    # plt.show()





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