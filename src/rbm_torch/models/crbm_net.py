import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

# Project Dependencies
# from rbm_torch import crbm_configs
from rbm_torch.models.crbm import CRBM
from rbm_torch.models.pool_crbm import pool_CRBM
import numpy as np
import math
from torch.optim import SGD, AdamW
from pytorch_lightning import LightningModule, Trainer
from rbm_torch.utils.utils import Categorical, HiddenInputs
from rbm_torch.utils.utils import FDS, BatchNorm2D, pool1d_dim


class FitnessPredictor(LightningModule):
    def __init__(self, crbm_checkpoint_file, network_config, debug=False):
        super().__init__()

        self.crbm = self.load_crbm(crbm_checkpoint_file, crbm_type="pool")

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

        self.network_type = network_config["predictor_network"]
        self.network_layers = network_config["network_layers"]
        self.network_objective = network_config["network_objective"]

        assert self.network_objective in ["classification", "regression"]
        if self.network_objective =="classification":
            self.class_number = network_config["network_class_number"]
            self.label_spacing = network_config["label_spacing"]

            training_fitness_values = self.crbm.training_data.seq_count.to_numpy()
            val_fitness_values = self.crbm.validation_data.seq_count.to_numpy()

            combined = np.concatenate([training_fitness_values, val_fitness_values])

            if type(self.label_spacing) is list:
                bin_edges = self.label_spacing
            else:
                if self.label_spacing == "log":
                    bin_edges = np.geomspace(np.min(combined), np.max(combined), self.class_number+1)
                elif self.label_spacing == "lin":
                    bin_edges = np.linspace(np.min(combined), np.max(combined), self.class_number + 1)
            bin_edges = bin_edges[1:]

            def assign_label(x):
                bin_edge = bin_edges[0]
                idx = 0
                while x > bin_edge:
                    idx += 1
                    bin_edge = bin_edges[idx]

                return idx

            train_labels = list(map(assign_label, training_fitness_values))
            self.class_weights = [1-train_labels.count(x)/len(train_labels) for x in range(0, self.class_number)]

            val_labels = list(map(assign_label, val_fitness_values))

            self.crbm.training_data["labels"] = train_labels
            self.crbm.validation_data["labels"] = val_labels


        # self.network_delay = 0.5
        # if "network_delay" in network_config.keys():
        #     self.network_delay = network_config["network_delay"]

        network = []
        network.append(nn.BatchNorm1d(linear_size))
        if self.network_type == "fcn":
            self.fcn_dr = network_config["fcn_dropout"]

            if "fcn_start_size" in network_config.keys():
                fcn_start_size = network_config["fcn_start_size"]
                network.append(nn.Dropout(self.fcn_dr))
                network.append(nn.Linear(linear_size, fcn_start_size, dtype=torch.get_default_dtype()))
                network.append(nn.BatchNorm1d(fcn_start_size))
                network.append(nn.LeakyReLU())
                linear_size = network_config["fcn_start_size"]
                self.network_layers -= 1

            fcn_size = [linear_size]
            if self.network_layers > 1:
                fcn_size += [linear_size // i for i in range(2, self.network_layers + 1, 1)]

                for i in range(self.network_layers - 1):
                    network.append(nn.Dropout(self.fcn_dr))
                    network.append(nn.Linear(fcn_size[i], fcn_size[i + 1], dtype=torch.get_default_dtype()))
                    network.append(nn.BatchNorm1d(fcn_size[i + 1]))
                    network.append(nn.LeakyReLU())

            if self.network_objective == "regression":
                network.append(nn.Linear(fcn_size[-1], 1, dtype=torch.get_default_dtype()))
                network.append(nn.Sigmoid())
            elif self.network_objective == "classification":
                network.append(nn.Linear(fcn_size[-1], self.class_number, dtype=torch.get_default_dtype()))
                network.append(nn.LogSoftmax())

        elif self.network_type == "conv":

            conv_start_channels = network_config["conv_start_channels"]
            conv_end_channels = network_config["conv_end_channels"]

            kernel_sizes, output_sizes, channels = [], [], []

            channels = [x for x in range(conv_start_channels, conv_end_channels - 1, -int((conv_start_channels - conv_end_channels) // (self.network_layers - 1)))]

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
                        nn.Conv1d(channels[i - 1], channels[i], kernel_size=kernel_sizes[i]),
                        nn.BatchNorm1d(channels[i]),
                        nn.Tanh()))

            if self.network_objective == "regression":
                self.final = nn.Sequential(nn.Linear(channels[-1] * output_sizes[-1], 1),
                                           nn.Sigmoid())
            elif self.network_objective == "classification":
                self.final = nn.Sequential(nn.Linear(channels[-1] * output_sizes[-1], self.class_number),
                                           nn.LogSoftmax())


        self.net = nn.Sequential(*network)

        self.network_loss_type = network_config["network_loss"]

        if self.network_objective == "regression":
            if self.network_loss_type == "l1":
                self.netloss = nn.L1Loss(size_average=None, reduce=None, reduction='mean')
            elif self.network_loss_type == "mse":
                self.netloss = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        elif self.network_objective == "classification":
            if self.network_loss_type == "nll":
                self.netloss = nn.NLLLoss(weight=torch.tensor(self.class_weights, device=self.device), reduction='mean')

        self.use_fds = network_config["use_fds"]
        if self.use_fds:
            self.fds_kernel = network_config["fds_kernel"]  # gaussian, laplace, triang
            self.fds_ks = network_config["fds_ks"]  # 5, FDS kernel size: should be odd number
            self.fds_sigma = network_config["fds_sigma"]  # 2, FDS gaussian/laplace kernel sigma
            self.fds_start_update = network_config["fds_start_update"]  # 0, which epoch to start FDS updating
            self.fds_start_smooth = network_config["fds_start_smooth"]  # 1, which epoch to start using FDS to smooth features
            self.fds_bucket_num = network_config["fds_bucket_num"]  # 50, maximum bucket considered for FDS
            self.fds_bucket_start = network_config["fds_bucket_start"]  # 0, minimum(starting) bucket for FDS
            self.fds_momentum = network_config["fds_momentum"]  # 0.9, FDS momentum

            self.FDS = FDS(feature_dim=self.linear_size, bucket_num=self.fds_bucket_num, bucket_start=self.fds_bucket_start,
                           start_update=self.fds_start_update, start_smooth=self.fds_start_smooth,
                           kernel=self.fds_kernel, ks=self.fds_ks, sigma=self.fds_sigma, momentum=self.fds_momentum, device=self.device)


        self.save_hyperparameters()

    def on_train_start(self):
        super().on_train_start()
        # only way I can get this thing on the correct device
        if self.use_fds:
            self.FDS.device = self.device

    def network(self, hidden_inputs, fitness_targets, eval=False):
        x = hidden_inputs

        # Enable FDS
        if self.use_fds or eval:
            if self.current_epoch >= self.fds_start_smooth:
                x = self.FDS.smooth(x, fitness_targets, self.current_epoch)

        if self.network_type == "fcn":
            return self.net(x)
        elif self.network_type == "conv":
            y = self.net(x.unsqueeze(1))
            y = y.view(y.size(0), -1)  # flatten
            return self.final(y)

    def configure_optimizers(self):
        optim = self.network_optimizer(self.parameters(), lr=self.network_lr, weight_decay=self.network_wd)
        decay_gamma = (self.network_lrf / self.network_lr) ** (1 / (self.network_epochs * (1 - self.network_decay_after)))
        decay_milestone = math.floor(self.network_decay_after * self.network_epochs)
        my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=[decay_milestone], gamma=decay_gamma)
        # my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=decay_gamma)
        return optim

    def load_crbm(self, checkpoint_file, crbm_type="crbm"):
        if crbm_type == "crbm":
            crbm = CRBM.load_from_checkpoint(checkpoint_file)
        elif crbm_type == "pool":
            crbm = pool_CRBM.load_from_checkpoint(checkpoint_file)
        crbm.eval()
        return crbm


    def train_dataloader(self, init_fields=True):
        # Get Correct Weights
        if self.network_objective == "regression":
            if "seq_count" in self.crbm.training_data.columns:
                training_weights = self.crbm.training_data["seq_count"].tolist()
            else:
                training_weights = None
        elif self.network_objective == "classification":
            if "labels" in self.crbm.training_data.columns:
                training_weights = self.crbm.training_data["labels"].tolist()

        train_reader = HiddenInputs(self.crbm, self.crbm.training_data, self.crbm.q, training_weights, max_length=self.crbm.v_num,
                                   molecule=self.crbm.molecule, device=self.crbm.device, one_hot=True)

        return torch.utils.data.DataLoader(
            train_reader,
            batch_size=self.crbm.batch_size,
            num_workers=self.crbm.worker_num,  # Set to 0 if debug = True
            pin_memory=self.crbm.pin_mem,
            shuffle=True
        )

    def val_dataloader(self):
        # Get Correct Validation weights
        if self.network_objective == "regression":
            if "seq_count" in self.crbm.validation_data.columns:
                validation_weights = self.crbm.validation_data["seq_count"].tolist()
            else:
                validation_weights = None
        elif self.network_objective == "classification":
            if "labels" in self.crbm.validation_data.columns:
                validation_weights = self.crbm.validation_data["labels"].tolist()

        val_reader = HiddenInputs(self.crbm, self.crbm.validation_data, self.crbm.q, validation_weights, max_length=self.crbm.v_num,
                                   molecule=self.crbm.molecule, device=self.crbm.device, one_hot=True)

        return torch.utils.data.DataLoader(
            val_reader,
            batch_size=self.crbm.batch_size,
            num_workers=self.crbm.worker_num,  # Set to 0 to view tensors while debugging
            pin_memory=self.crbm.pin_mem,
            shuffle=False
        )

    def training_step(self, batch, batch_idx):
        hidden_inputs, fitness_targets = batch
        fitness_targets = fitness_targets.to(torch.get_default_dtype())
        preds = self.network(hidden_inputs, fitness_targets)

        if self.network_objective == "regression":
            train_loss = self.netloss(preds.squeeze(1), fitness_targets)
            self.log('train_MSE', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        elif self.network_objective == "classification":
            train_loss = self.netloss(preds, fitness_targets.long())
            class_pred = torch.argmax(preds, 1)
            acc = (class_pred == fitness_targets).double().mean()
            self.log('train_Loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("train Acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return train_loss

    def classification_tables(self):
        from sklearn.metrics import classification_report
        print('\nClassification Report\n')

        train_seqs, train_class_preds = self.predict(self.crbm.training_data)
        val_seqs, val_class_preds = self.predict(self.crbm.validation_data)

        train_true_labels = self.crbm.training_data.labels.tolist()
        val_true_labels = self.crbm.validation_data.labels.tolist()

        print('\nTraining Classification Report\n')
        print(classification_report(train_class_preds, train_true_labels, target_names=[f'Class {i}' for i in range(self.class_number)]))
        print('\nValidation Classification Report\n')
        print(classification_report(val_class_preds, val_true_labels, target_names=[f'Class {i}' for i in range(self.class_number)]))

    def validation_step(self, batch, batch_idx):
        hidden_inputs, fitness_targets = batch
        fitness_targets = fitness_targets.to(torch.get_default_dtype())
        preds = self.network(hidden_inputs, fitness_targets)

        if self.network_objective == "regression":
            val_loss = self.netloss(preds.squeeze(1), fitness_targets)
            self.log('val_MSE', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        elif self.network_objective == "classification":
            val_loss = self.netloss(preds, fitness_targets.long())
            class_pred = torch.argmax(preds, 1)
            acc = (class_pred == fitness_targets).double().mean()
            self.log('val_Loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("val Acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return val_loss

    # def forward(self, input):
    #     with torch.no_grad():
    #         # attempting to remove crbm_variables from gradient calculation
    #         # h_pos = self.crbm.sample_from_inputs_h(self.crbm.compute_output_v(input))
    #         h_pos = self.crbm.compute_output_v(input)
    #     h_flattened = torch.cat([torch.flatten(x, start_dim=1) for x in h_pos], dim=1)
    #     return self.net(h_flattened)

    def predict(self, X):
        # Read in data
        # reader = Categorical(X, self.crbm.q, weights=None, max_length=self.crbm.v_num, molecule=self.crbm.molecule, device=self.device, one_hot=True)

        reader = HiddenInputs(self.crbm, X, self.crbm.q, None, max_length=self.crbm.v_num,
                                   molecule=self.crbm.molecule, device=self.crbm.device, one_hot=True)

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
                one_hot, fitness_targets = batch
                if self.network_objective == "regression":
                    fitness_vals += self.network(one_hot, fitness_targets, eval=False).squeeze(1).detach().tolist()
                elif self.network_objective == "classification":
                    fitness_vals += self.network(one_hot, fitness_targets, eval=False).argmax(1).detach().tolist()

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

        self.use_cd = config["use_cd"]
        self.use_pearson = config["use_pearson"]
        self.use_batch_norm = config["use_batch_norm"]
        # self.use_spearman = config["use_spearman"]
        self.use_lst_sqrs = config["use_lst_sqrs"]
        self.use_network = config["use_network"]

        input_size = (config["batch_size"], self.v_num, self.q)


        if self.use_batch_norm:
            for kid, key in enumerate(self.hidden_convolution_keys):
                setattr(self, f"batch_norm_{key}", BatchNorm2D(affine=False, momentum=0.1))
                setattr(self, f"batch_norm_{key}", BatchNorm2D(affine=False, momentum=0.1))

        #### for max and min pool version
        # self.pool_topology = config["pool_topology"]
        # # self.pool_kernels = config["pool_kernels"]
        # # self.pool_strides = config["pool_strides"]
        # self.pools = []
        # self.unpools = []
        # for kid, key in enumerate(self.hidden_convolution_keys):
        #     pool_input_size = self.convolution_topology[key]["convolution_dims"][:-1]
        #
        #     pool_top = self.pool_topology[key]
        #     # determines output size of pool, and reconstruction size. Checks it will work
        #     pool_dims = pool1d_dim(pool_input_size, pool_top)
        #
        #     self.pools.append(nn.MaxPool1d(pool_top["kernel"], stride=pool_top["stride"], return_indices=True, padding=pool_top["padding"]))
        #     self.unpools.append(nn.MaxUnpool1d(pool_top["kernel"], stride=pool_top["stride"], padding=pool_top["padding"]))

        # self.pool = nn.MaxPool2d(self.pool_kernel, return_indices=True)
        # self.unpool = nn.MaxUnpool2d(self.pool_kernel)

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


    ### Version with max and min pools
    # def compute_output_v(self, X):  # X is the one hot vector
    #     outputs = []
    #     hidden_layer_W = getattr(self, "hidden_layer_W")
    #     total_weights = hidden_layer_W.sum()
    #     for iid, i in enumerate(self.hidden_convolution_keys):
    #         # convx = self.convolution_topology[i]["convolution_dims"][2]
    #         conv = F.conv2d(X.unsqueeze(1).type(torch.get_default_dtype()), getattr(self, f"{i}_W"), stride=self.convolution_topology[i]["stride"],
    #                                 padding=self.convolution_topology[i]["padding"],
    #                                 dilation=self.convolution_topology[i]["dilation"]).squeeze(3)
    #
    #         max_pool, max_inds = self.pools[iid](conv)
    #         min_pool, min_inds = self.pools[iid](-1*conv)
    #
    #         max_reconst = self.unpools[iid](max_pool, max_inds)
    #         min_reconst = self.unpools[iid](-1*min_pool, min_inds)
    #
    #         outputs.append(max_reconst + min_reconst)
    #
    #         outputs[-1] *= hidden_layer_W[iid] / total_weights
    #         if self.use_batch_norm:
    #             batch_norm = getattr(self, f"batch_norm_{i}")  # get individual batch norm
    #             outputs[-1] = batch_norm(outputs[-1])  # apply batch norm
    #         # outputs[-1] *= convx
    #     return outputs
    #
    #     ## Compute Input for Visible Layer from Hidden dReLU
    #
    # def compute_output_h(self, Y):  # from h_uk (B, hidden_num, convx_num)
    #     outputs = []
    #     nonzero_masks = []
    #     hidden_layer_W = getattr(self, "hidden_layer_W")
    #     total_weights = hidden_layer_W.sum()
    #     for iid, i in enumerate(self.hidden_convolution_keys):
    #         # convx = self.convolution_topology[i]["convolution_dims"][2]
    #         outputs.append(F.conv_transpose2d(Y[iid].unsqueeze(3), getattr(self, f"{i}_W"),
    #                                           stride=self.convolution_topology[i]["stride"],
    #                                           padding=self.convolution_topology[i]["padding"],
    #                                           dilation=self.convolution_topology[i]["dilation"],
    #                                           output_padding=self.convolution_topology[i]["output_padding"]).squeeze(1))
    #         outputs[-1] *= hidden_layer_W[iid] / total_weights
    #         nonzero_masks.append((outputs[-1] != 0.).type(torch.get_default_dtype()) * getattr(self, "hidden_layer_W")[iid])  # Used for calculating mean of outputs, don't want zeros to influence mean
    #         # outputs[-1] /= convx  # multiply by 10/k to normalize by convolution dimension
    #     if len(outputs) > 1:
    #         # Returns mean output from all hidden layers, zeros are ignored
    #         mean_denominator = torch.sum(torch.stack(nonzero_masks), 0)
    #         return torch.sum(torch.stack(outputs), 0) / mean_denominator
    #     else:
    #         return outputs[0]

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

        # print("GPU Allocated After Forward:", torch.cuda.memory_allocated(0))
        free_energy = self.free_energy(one_hot)

        if self.use_cd:
            V_neg_oh, h_neg, V_pos_oh, h_pos = self(one_hot)
            F_v = (free_energy * fitness_targets).sum() / fitness_targets.sum()  # free energy of training data
            F_vp = (self.free_energy(V_neg_oh) * fitness_targets).sum() / fitness_targets.sum()  # free energy of gibbs sampled visible states
            cd_loss = F_v - F_vp

        if self.use_pearson:
            # correlation coefficient between free energy and fitness values
            vx = -1 * (free_energy - torch.mean(free_energy))  # multiply be negative one so lowest free energy vals get paired with the highest copy number/fitness values
            vy = fitness_targets - torch.mean(fitness_targets)

            pearson_correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + 1e-6) * torch.sqrt(torch.sum(vy ** 2) + 1e-6))
            pearson_loss = (1 - pearson_correlation) # * (self.current_epoch/self.epochs + 1) * 10

        # if self.use_spearman:
        #     spearman_correlation = spearman((-1 * free_energy).unsqueeze(0).cpu(), (fitness_targets).unsqueeze(0).cpu())
        #     spearman_loss = 1 - spearman_correlation

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
        if self.use_batch_norm:
            cd_loss /= pearson_multiplier

        loss = torch.zeros((1,), device=self.device)

        if self.use_cd:
            loss_term = cd_loss + reg1 + reg2 + reg3
            loss += loss_term  # * (1.2 - self.current_epoch/self.epochs)

        if self.use_pearson:
            loss += pearson_loss * 20  # * pearson_multiplier

        # if self.use_spearman:
        #     loss += spearman_loss  # * pearson_multiplier

        if self.use_lst_sqrs:
            loss += lst_sq_loss

        if self.use_network:
            if self.current_epoch/self.epochs > self.network_delay:
                loss += 5 * net_loss * self.current_epoch / self.epochs
            else:
                loss += net_loss*0.

        # Calculate Loss
        # loss = crbm_loss + net_loss * 5 + pearson_loss * pearson_multiplier + lst_sq_loss

        logs = {"loss": loss,
                # "free_energy_diff": cd_loss.detach(),
                # # f"train_{self.network_loss_type}_loss": net_loss.detach(),
                # "train_free_energy": F_v.detach(),
                # # "train_pearson_corr": pearson_correlation.detach(),
                # # "train_pearson_loss": pearson_loss.detach(),
                # # "train_residuals": residuals.mean().detach(),
                # "field_reg": reg1.detach(),
                # "weight_reg": reg2.detach(),
                # "distance_reg": reg3.detach()
                }

        if self.use_cd:
            logs["free_energy_diff"] = cd_loss.detach()
            logs["train_free_energy"] = F_v.detach()
            logs["field_reg"] = reg1.detach()
            logs["weight_reg"] = reg2.detach()
            logs["distance_reg"] = reg3.detach()
            self.log("ptl/train_free_energy", logs["train_free_energy"], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if self.use_pearson:
            logs["train_pearson_corr"] = pearson_correlation.detach()
            logs["train_pearson_loss"] = pearson_loss.detach()
            self.log("ptl/train_pearson_corr", logs["train_pearson_corr"], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # if self.use_spearman:
        #     logs["train_spearman_corr"] = spearman_correlation.detach()
        #     logs["train_spearman_loss"] = spearman_loss.detach()
        #     self.log("ptl/train_spearman_corr", logs["train_spearman_corr"], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if self.use_network:
            logs[f"train_{self.network_loss_type}_loss"] = net_loss.detach()
            self.log(f"ptl/train_fitness_{self.network_loss_type}", raw_mse_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if self.use_lst_sqrs:
            self.log("ptl/train_residuals", residuals.sum().detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # self.log("ptl/train_free_energy", logs["train_free_energy"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
        # pseudo_likelihood = torch.stack([x['train_pseudo_likelihood'] for x in outputs]).mean()

        loss_scalars = {"Total": avg_loss}
        metric_scalars = {}

        if self.use_cd:
            avg_dF = torch.stack([x["free_energy_diff"] for x in outputs]).mean()
            field_reg = torch.stack([x["field_reg"] for x in outputs]).mean()
            weight_reg = torch.stack([x["weight_reg"] for x in outputs]).mean()
            distance_reg = torch.stack([x["distance_reg"] for x in outputs]).mean()
            free_energy = torch.stack([x["train_free_energy"] for x in outputs]).mean()
            self.logger.experiment.add_scalars('Regularization', {'Field Reg': field_reg,
                                                                  'Weight Reg': weight_reg,
                                                                  'Distance Reg': distance_reg}, self.current_epoch)
            loss_scalars[f"CD_Loss"] = avg_dF
            metric_scalars["train_free_energy"] = free_energy

        if self.use_network:
            net_loss = torch.stack([x[f"train_{self.network_loss_type}_loss"] for x in outputs]).mean()
            loss_scalars[f"Train Fitness {self.network_loss_type}"] = net_loss
            metric_scalars[f"Train Fitness {self.network_loss_type}"] = net_loss

        if self.use_pearson:
            pearson_corr = torch.stack([x["train_pearson_corr"] for x in outputs]).mean()
            pearson_loss = torch.stack([x["train_pearson_loss"] for x in outputs]).mean()
            loss_scalars[f"Pearson Loss"] = pearson_loss
            metric_scalars["Train Pearson Corr"] = pearson_corr

        # if self.use_spearman:
        #     spearman_corr = torch.stack([x["train_spearman_corr"] for x in outputs]).mean()
        #     spearman_loss = torch.stack([x["train_spearman_loss"] for x in outputs]).mean()
        #     loss_scalars[f"Spearman Loss"] = spearman_loss
        #     metric_scalars["Train Spearman Corr"] = spearman_corr

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

        # if self.use_spearman:
        #     spearman_correlation = spearman((-1 * free_energy).unsqueeze(0).cpu(), (fitness_targets).unsqueeze(0).cpu())

        # pearson_loss = 1 - pearson_correlation

        batch_out = {}
            # "val_pseudo_likelihood": pseudo_likelihood.detach()

            # f"val_fitness_{self.network_loss_type}": net_loss.detach(),
            # "val_pearson_corr": pearson_correlation.detach()
        # }

        if self.use_cd:
            batch_out["val_free_energy"] = free_energy_avg.detach()
            self.log("ptl/val_free_energy", batch_out["val_free_energy"], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.use_network:
            batch_out[f"val_fitness_{self.network_loss_type}"] = net_loss.detach()
            self.log(f"ptl/val_fitness_{self.network_loss_type}", batch_out[f"val_fitness_{self.network_loss_type}"],
                     on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.use_pearson:
            batch_out["val_pearson_corr"] = pearson_correlation.detach()
            self.log("ptl/val_pearson_corr", batch_out["val_pearson_corr"], on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)

        # if self.use_spearman:
        #     batch_out["val_spearman_corr"] = spearman_correlation.detach()
        #     self.log("ptl/val_spearman_corr", batch_out["val_spearman_corr"], on_step=False, on_epoch=True, prog_bar=True,
        #              logger=True)


        # logging on step, for whatever reason allocates 512 bytes on gpu after every epoch.
        # self.log("ptl/val_free_energy", batch_out["val_free_energy"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log(f"ptl/val_fitness_{self.network_loss_type}", batch_out[f"val_fitness_{self.network_loss_type}"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("ptl/val_pearson_corr", batch_out["val_pearson_corr"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #
        return batch_out


    def validation_epoch_end(self, outputs):
        # avg_fe = torch.stack([x['val_free_energy'] for x in outputs]).mean()
        # metrics = {"val_free_energy": avg_fe}
        metrics = {}

        if self.use_cd:
            avg_fe = torch.stack([x['val_free_energy'] for x in outputs]).mean()
            metrics["val_free_energy"] = avg_fe

        if self.use_network:
            avg_loss = torch.stack([x[f'val_fitness_{self.network_loss_type}'] for x in outputs]).mean()
            metrics[f"val_fitness_{self.network_loss_type}"] = avg_loss

        if self.use_pearson:
            avg_pearson = torch.stack([x['val_pearson_corr'] for x in outputs]).mean()
            metrics["val_pearson_corr"] = avg_pearson

        # if self.use_spearman:
        #     avg_spearman = torch.stack([x['val_spearman_corr'] for x in outputs]).mean()
        #     metrics["val_spearman_corr"] = avg_spearman

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

                if self.use_network:
                    h_input = self.compute_output_v(one_hot_gpu)
                    h_flattened = torch.cat([torch.flatten(x, start_dim=1) for x in h_input], dim=1)
                    preds = self.network(h_flattened, seq_weights)

                    fitness_vals += preds.cpu().squeeze(1).tolist()

        if self.use_network:
            return X.sequence.tolist(), likelihood, fitness_vals
        else:
            return X.sequence.tolist(), likelihood
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
    network_config = {"network_epochs": 200, "network_lr": 0.0005, "network_lr_final": 0.0005, "network_weight_decay": 0.001,
                      "network_decay_after": 0.75, "network_optimizer": "AdamW", "fasta_file": "./datasets/exo/caris_train.fasta", "weights": "fasta",
                    "network_dr": 0.01, "network_layers": 3, "predictor_network": "fcn", "fcn_start_size": 500,
                      "conv_start_channels": 16, "conv_end_channels": 4, "fcn_dropout": 0.05, "network_loss": "nll", "network_objective": "classification",
                      "network_class_number": 2, "label_spacing": [0., 0.15, 1.0],
                      "use_fds": False,
                      "fds_kernel": "gaussian",
                      "fds_ks": 5,
                      "fds_sigma": 2,
                      "fds_start_update": 0,
                      "fds_start_smooth": 1,
                      "fds_bucket_num": 10,
                      "fds_bucket_start": 0,
                      "fds_momentum": 0.9
                      }


    train = False
    # r = "g3_t5"
    r = "crbm_en_exo"

    if train:
        import rbm_torch.analysis.analysis_methods as am

        # mdir = "/mnt/D1/globus/cov_trained_crbms/"

        mdir = f"./datasets/exo/trained_crbms/"
        # r = "g3_t5"
        #
        checkp, version_dir = am.get_checkpoint_path(r, rbmdir=mdir, version=182)
        #
        device = torch.device("cuda")
        fp = FitnessPredictor(checkp, network_config, debug=True)
        fp.to(device)

        logger = TensorBoardLogger("./datasets/exo/trained_crbms/", name=f"fitness_predictor_{r}")
        plt = Trainer(max_epochs=network_config['network_epochs'], logger=logger, accelerator="cuda", devices=1)
        plt.fit(fp)

        fp.classification_tables()
    else:
        # Evaluate how model did
        check = f"./datasets/exo/trained_crbms/fitness_predictor_{r}/version_34/checkpoints/epoch=199-step=800.ckpt"
        fp = FitnessPredictor.load_from_checkpoint(check)
        fp.eval()

        from rbm_torch.utils.utils import fasta_read
        import pandas as pd
        # seqs, folds, chars, q = fasta_read("./datasets/exo/en_avg_g3.fasta", "dna", threads=4)
        dataset_file = "./datasets/exo/caris_train.fasta"
        ev = "./datasets/exo/pev_n.fasta"
        el = "./datasets/exo/pel_n.fasta"

        ds_seqs, ds_folds, ds_chars, ds_q = fasta_read(dataset_file, "dna", threads=4)
        ev_seqs, ev_folds, ev_chars, ev_q = fasta_read(ev, "dna", threads=4)
        el_seqs, el_folds, el_chars, el_q = fasta_read(el, "dna", threads=4)

        ds_df = pd.DataFrame({"sequence": ds_seqs, "copy_num": ds_folds})
        ev_df = pd.DataFrame({"sequence": ev_seqs, "copy_num": ev_folds})
        el_df = pd.DataFrame({"sequence": el_seqs, "copy_num": el_folds})

        ds_ss, ds_fit_vals = fp.predict(ds_df)
        ev_ss, ev_fit_vals = fp.predict(ev_df)
        el_ss, el_fit_vals = fp.predict(el_df)

        import matplotlib.pyplot as plt

        plt.scatter(ds_folds, ds_fit_vals, alpha=0.3, s=2, c="blue")
        ev_folds = [0.05 for x in ev_fit_vals]
        el_folds = [0.9 for x in el_fit_vals]
        plt.scatter(ev_folds, ev_fit_vals, alpha=0.3, s=2, c="green")
        plt.scatter(el_folds, el_fit_vals, alpha=0.3, s=2, c="red")
        plt.show()

        print(ev_fit_vals.count(0)/len(ev_fit_vals), ev_fit_vals.count(1)/len(ev_fit_vals))
        print(el_fit_vals.count(0)/len(el_fit_vals), el_fit_vals.count(1)/len(el_fit_vals))

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
    # #
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