from rbm_torch.crbm import CRBM

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.profiler import SimpleProfiler, PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.optim import SGD, AdamW
from sklearn.metrics import balanced_accuracy_score

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import randint
import numpy as np

from torchvision import transforms
from torchvision.datasets import MNIST
import torchmetrics
import math

from torch.utils.data import Dataset, DataLoader

# q should be 1 on config
class BinaryCRBM(CRBM):
    def __init__(self, config, dataset="mnist", debug=False):
        super().__init__(config, debug=debug)
        self.dataset = dataset
        self.v_num = config["v_num"]


    def transform_v(self, I):
        return (I+getattr(self, "fields").squeeze(-1))>0

    def energy_v(self, config, remove_init=False):
        if remove_init:
            return -1.*(config * ((getattr(self, "fields") - getattr(self, "fields0")).unsqueeze(0).squeeze(-1)))
        else:
            # return -torch.mm(config, getattr(self, "fields").squeeze(-1))
            return -1.*(config * getattr(self, "fields").unsqueeze(0).squeeze(-1)).sum((2, 1))

    def random_init_config_v(self, custom_size=False, zeros=False):
        if custom_size:
            size = (*custom_size, self.v_num)
        else:
            size = (self.batch_size, self.v_num)

        if zeros:
            return torch.zeros(size, device=self.device)
        else:
            ### might need rewriting
            return self.sample_from_inputs_v(torch.zeros(size, device=self.device).flatten(0, -3), beta=0).reshape(size)

    def logpartition_v(self, inputs, beta=1):
        if beta == 1:
            return torch.log(1 + torch.exp(getattr(self, "fields").unsqueeze(0).squeeze(-1) + inputs)).sum(1)
        else:
            return torch.log(1 + torch.exp((beta * getattr(self, "fields")).unsqueeze(0).squeeze(-1) + (1 - beta) * getattr(self, "fields0")).unsqueeze(0).squeeze(-1) + beta * inputs).sum(1)

    def logpartition_h(self, inputs, beta=1):
        # Input is list of matrices I_uk
        marginal = torch.zeros((len(self.hidden_convolution_keys), inputs[0].shape[0]), device=self.device)
        for iid, i in enumerate(self.hidden_convolution_keys):
            if beta == 1:
                a_plus = (getattr(self, f'{i}_gamma+')).unsqueeze(0).unsqueeze(2).unsqueeze(3)
                a_minus = (getattr(self, f'{i}_gamma-')).unsqueeze(0).unsqueeze(2).unsqueeze(3)
                theta_plus = (getattr(self, f'{i}_theta+')).unsqueeze(0).unsqueeze(2).unsqueeze(3)
                theta_minus = (getattr(self, f'{i}_theta-')).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            else:
                theta_plus = (beta * getattr(self, f'{i}_theta+') + (1 - beta) * getattr(self, f'{i}_0theta+')).unsqueeze(0).unsqueeze(2).unsqueeze(3)
                theta_minus = (beta * getattr(self, f'{i}_theta-') + (1 - beta) * getattr(self, f'{i}_0theta-')).unsqueeze(0).unsqueeze(2).unsqueeze(3)
                a_plus = (beta * getattr(self, f'{i}_gamma+') + (1 - beta) * getattr(self, f'{i}_0gamma+')).unsqueeze(0).unsqueeze(2).unsqueeze(3)
                a_minus = (beta * getattr(self, f'{i}_gamma-') + (1 - beta) * getattr(self, f'{i}_0gamma-')).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            y = torch.logaddexp(self.log_erf_times_gauss((-inputs[iid] + theta_plus) / torch.sqrt(a_plus)) -
                                0.5 * torch.log(a_plus), self.log_erf_times_gauss((inputs[iid] + theta_minus) / torch.sqrt(a_minus)) - 0.5 * torch.log(a_minus)).sum(
                1) + 0.5 * np.log(2 * np.pi) * inputs[iid].shape[1]
            marginal[iid] = y.sum((2, 1)) # sum over uk
            # marginal[iid] /= self.convolution_topology[i]["convolution_dims"][2]
        return marginal.sum(0)

    def compute_output_v(self, X):
        # X input size 1000, 1, 28, 28 for mnist data
        # Might need rewriting
        outputs = []
        hidden_layer_W = getattr(self, "hidden_layer_W")
        total_weights = hidden_layer_W.sum()
        for iid, i in enumerate(self.hidden_convolution_keys):
            # convx = self.convolution_topology[i]["convolution_dims"][2]
            outputs.append(F.conv2d(X.unsqueeze(1).type(torch.get_default_dtype()), getattr(self, f"{i}_W"), stride=self.convolution_topology[i]["stride"],
                                    padding=self.convolution_topology[i]["padding"],
                                    dilation=self.convolution_topology[i]["dilation"]).squeeze(3))
            outputs[-1] *= hidden_layer_W[iid] / total_weights
            # outputs[-1] *= convx
        return outputs

    ## Compute Input for Visible Layer from Hidden dReLU
    def compute_output_h(self, Y):  # from h_uk (B, hidden_num, convx_num)
        outputs = []
        nonzero_masks = []
        hidden_layer_W = getattr(self, "hidden_layer_W")
        total_weights = hidden_layer_W.sum()
        for iid, i in enumerate(self.hidden_convolution_keys):
            # convx = self.convolution_topology[i]["convolution_dims"][2]
            outputs.append(F.conv_transpose2d(Y[iid], getattr(self, f"{i}_W"),
                                              stride=self.convolution_topology[i]["stride"],
                                              padding=self.convolution_topology[i]["padding"],
                                              dilation=self.convolution_topology[i]["dilation"],
                                              output_padding=self.convolution_topology[i]["output_padding"]).squeeze(1))
            outputs[-1] *= hidden_layer_W[iid] / total_weights
            nonzero_masks.append((outputs[-1] != 0.).type(torch.get_default_dtype()) * getattr(self, "hidden_layer_W")[iid])  # Used for calculating mean of outputs, don't want zeros to influence mean
            # outputs[-1] /= convx  # multiply by 10/k to normalize by convolution dimension
        if len(outputs) > 1:
            # Returns mean output from all hidden layers, zeros are ignored
            mean_denominator = torch.sum(torch.stack(nonzero_masks), 0)
            return torch.sum(torch.stack(outputs), 0) / mean_denominator
        else:
            return outputs[0]

    def mean_v(self, psi, beta=1):
        if beta == 1:
            return torch.special.expit(psi + getattr(self, "fields").unsqueeze(0).squeeze(-1))
        else:
            return torch.special.expit(beta * psi + getattr(self, "fields0").unsqueeze(0).squeeze(-1) + beta * (getattr(self, "fields").unsqueeze(0).squeeze(-1) - getattr(self, "fields0").unsqueeze(0).squeeze(-1)))

    def sample_from_inputs_v(self, psi, beta=1):
        return torch.randn(psi.shape, device=self.device) < self.mean_v(psi, beta=beta)

    def setup(self, stage=None):
        if self.dataset == "mnist":
            PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
            self.train_reader = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
            self.val_reader = MNIST(PATH_DATASETS, train=False, download=True, transform=transforms.ToTensor())

    def on_train_start(self):
        pass

    def train_dataloader(self, init_fields=True):
        if init_fields:
            with torch.no_grad():
                tmp_fields = torch.randn((*self.v_num, self.q), device=self.device)
                self.fields += tmp_fields
                self.fields0 += tmp_fields


        return torch.utils.data.DataLoader(
            self.train_reader,
            batch_size=self.batch_size,
            num_workers=self.worker_num,  # Set to 0 if debug = True
            pin_memory=self.pin_mem,
            shuffle=True
        )

    def val_dataloader(self, init_fields=True):
        return torch.utils.data.DataLoader(
            self.val_reader,
            batch_size=self.batch_size,
            num_workers=self.worker_num,  # Set to 0 if debug = True
            pin_memory=self.pin_mem,
            shuffle=False
        )

    def training_step(self, batch, batch_idx):

        x, labels = batch # we ignore the labels for now as this is an unsupervised model
        x_neg, h_neg, x_pos, h_pos = self(x.squeeze(1))

        F_v = self.free_energy(x_pos).mean(0)  # free energy of training data
        F_vp = self.free_energy(x_neg).mean(0) # free energy of gibbs sampled visible states

        cd_loss = F_v - F_vp

        # Regularization Terms
        reg1 = self.lf / (2 * math.prod(self.v_num)) * getattr(self, "fields").square().sum((0, 1))
        reg2 = torch.zeros((1,), device=self.device)
        # reg3 = torch.zeros((1,), device=self.device)
        for iid, i in enumerate(self.hidden_convolution_keys):
            W_shape = self.convolution_topology[i]["weight_dims"]  # (h_num,  input_channels, kernel0, kernel1)
            x = torch.sum(torch.abs(getattr(self, f"{i}_W")), (3, 2, 1)).square()
            reg2 += x.mean() * self.l1_2 / (2 * W_shape[1] * W_shape[2] * W_shape[3])
            # reg3 += self.ld / ((getattr(self, f"{i}_W").abs() - getattr(self, f"{i}_W").squeeze(1).abs()).abs().sum((1, 2, 3)).mean() + 1)

        # Calculate Loss
        loss = cd_loss + reg1 + reg2  # + reg3

        logs = {"loss": loss,
                "free_energy_diff": cd_loss.detach(),
                "train_free_energy": F_v.detach(),
                "field_reg": reg1.detach(),
                "weight_reg": reg2.detach(),
                # "distance_reg": reg3.detach()
                }

        self.log("ptl/train_free_energy", logs["train_free_energy"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return logs

    def validation_step(self, batch, batch_idx):

        x, labels = batch  # we ignore the labels for now as this is an unsupervised model
        x_neg, h_neg, x_pos, h_pos = self(x.squeeze(1))

        F_v = self.free_energy(x_pos).mean(0)  # free energy of training data
        F_vp = self.free_energy(x_neg).mean(0)  # free energy of gibbs sampled visible states

        cd_loss = F_v - F_vp

        # Calculate Loss
        loss = cd_loss  # + reg1 + reg2 + reg3

        logs = {"loss": loss,
                "free_energy_diff": cd_loss.detach(),
                "val_free_energy": F_v.detach(),
                # "field_reg": reg1.detach(),
                # "weight_reg": reg2.detach(),
                # "distance_reg": reg3.detach()
                }

        self.log("ptl/val_free_energy", logs["val_free_energy"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return logs

    def forward(self, X):
        with torch.no_grad():  # only use last sample for gradient calculation, Enabled to minimize memory usage, hopefully won't have much effect on performance
            fantasy_v, fantasy_h = self.markov_step(X)
            for _ in range(self.mc_moves - 2):
                fantasy_v, fantasy_h = self.markov_step(fantasy_v)

        V_neg, fantasy_h = self.markov_step(fantasy_v)

        # V_neg, h_neg, V_pos, h_pos
        return V_neg, self.sample_from_inputs_h(self.compute_output_v(V_neg)), X, self.sample_from_inputs_h(self.compute_output_v(X))

    ## Gibbs Sampling of dReLU hidden layer
    def sample_from_inputs_h(self, psi, nancheck=False, beta=1):  # psi is a list of hidden Iuks
        h_uks = []
        for iid, i in enumerate(self.hidden_convolution_keys):
            if beta == 1:
                a_plus = getattr(self, f'{i}_gamma+').unsqueeze(0).unsqueeze(2).unsqueeze(3)
                a_minus = getattr(self, f'{i}_gamma-').unsqueeze(0).unsqueeze(2).unsqueeze(3)
                theta_plus = getattr(self, f'{i}_theta+').unsqueeze(0).unsqueeze(2).unsqueeze(3)
                theta_minus = getattr(self, f'{i}_theta-').unsqueeze(0).unsqueeze(2).unsqueeze(3)
            else:
                theta_plus = (beta * getattr(self, f'{i}_theta+') + (1 - beta) * getattr(self, f'{i}_0theta+')).unsqueeze(0).unsqueeze(2).unsqueeze(3)
                theta_minus = (beta * getattr(self, f'{i}_theta-') + (1 - beta) * getattr(self, f'{i}_0theta-')).unsqueeze(0).unsqueeze(2).unsqueeze(3)
                a_plus = (beta * getattr(self, f'{i}_gamma+') + (1 - beta) * getattr(self, f'{i}_0gamma+')).unsqueeze(0).unsqueeze(2).unsqueeze(3)
                a_minus = (beta * getattr(self, f'{i}_gamma-') + (1 - beta) * getattr(self, f'{i}_0gamma-')).unsqueeze(0).unsqueeze(2).unsqueeze(3)
                psi[iid] *= beta

            if nancheck:
                nans = torch.isnan(psi[iid])
                if nans.max():
                    nan_unit = torch.nonzero(nans.max(0))[0]
                    print('NAN IN INPUT')
                    print('Hidden units', nan_unit)

            psi_plus = (-psi[iid]).add(theta_plus).div(torch.sqrt(a_plus))
            psi_minus = psi[iid].add(theta_minus).div(torch.sqrt(a_minus))

            etg_plus = self.erf_times_gauss(psi_plus)  # Z+ * sqrt(a+)
            etg_minus = self.erf_times_gauss(psi_minus)  # Z- * sqrt(a-)

            p_plus = 1 / (1 + (etg_minus / torch.sqrt(a_minus)) / (etg_plus / torch.sqrt(a_plus)))  # p+ 1 / (1 +( (Z-/sqrt(a-))/(Z+/sqrt(a+))))    =   (Z+/(Z++Z-)
            nans = torch.isnan(p_plus)

            if True in nans:
                p_plus[nans] = torch.tensor(1.) * (torch.abs(psi_plus[nans]) > torch.abs(psi_minus[nans]))
            p_minus = 1 - p_plus

            is_pos = torch.rand(psi[iid].shape, device=self.device) < p_plus

            rmax = torch.zeros(p_plus.shape, device=self.device)
            rmin = torch.zeros(p_plus.shape, device=self.device)
            rmin[is_pos] = torch.erf(psi_plus[is_pos].mul(self.invsqrt2))  # Part of Phi(x)
            rmax[is_pos] = 1  # pos values rmax set to one
            rmin[~is_pos] = -1  # neg samples rmin set to -1
            rmax[~is_pos] = torch.erf((-psi_minus[~is_pos]).mul(self.invsqrt2))  # Part of Phi(x)

            h = torch.zeros(psi[iid].shape, dtype=torch.float64, device=self.device)
            tmp = (rmax - rmin > 1e-14)
            h = self.sqrt2 * torch.erfinv(rmin + (rmax - rmin) * torch.rand(h.shape, device=self.device))
            h[is_pos] -= psi_plus[is_pos]
            h[~is_pos] += psi_minus[~is_pos]
            h /= torch.sqrt(is_pos * a_plus + ~is_pos * a_minus)
            h[torch.isinf(h) | torch.isnan(h) | ~tmp] = 0
            h_uks.append(h)
        return h_uks

    def training_epoch_end(self, outputs):
        # These are detached
        avg_loss = torch.stack([x['loss'].detach() for x in outputs]).mean()
        avg_dF = torch.stack([x["free_energy_diff"] for x in outputs]).mean()
        field_reg = torch.stack([x["field_reg"] for x in outputs]).mean()
        weight_reg = torch.stack([x["weight_reg"] for x in outputs]).mean()
        # free_energy = torch.stack([x["train_free_energy"] for x in outputs]).mean()
        # pseudo_likelihood = torch.stack([x['train_pseudo_likelihood'] for x in outputs]).mean()

        self.logger.experiment.add_scalars("All Scalars", {"Loss": avg_loss,
                                                           "CD_Loss": avg_dF,
                                                           "W_reg": weight_reg,
                                                           "field_reg": field_reg,
                                                           # "Train_pseudo_likelihood": pseudo_likelihood,
                                                           # "Train Free Energy": free_energy,
                                                           }, self.current_epoch)

        for name, p in self.named_parameters():
            self.logger.experiment.add_histogram(name, p.detach(), self.current_epoch)

    def validation_epoch_end(self, outputs):
        # avg_pl = torch.stack([x['val_pseudo_likelihood'] for x in outputs]).mean()
        # self.logger.experiment.add_scalar("Validation pseudo_likelihood", avg_pl, self.current_epoch)
        avg_fe = torch.stack([x['val_free_energy'] for x in outputs]).mean()
        val_loss = torch.stack([x['loss'].detach() for x in outputs]).mean()
        self.logger.experiment.add_scalar("Validation Free Energy", avg_fe, self.current_epoch)
        self.logger.experiment.add_scalar("Validation Loss", val_loss, self.current_epoch)

class BinaryClassifier(LightningModule):
    def __init__(self, crbm_config, dataset="mnist", train_crbm=False, debug=False):
        super().__init__()

        self.binary_crbm = BinaryCRBM(crbm_config, debug=debug)
        self.crbm_epochs = crbm_config["epochs"]
        self.classifier_epochs = crbm_config["classifier_epochs"]
        log_dir = crbm_config["crbm_log_dir"]

        # classifier optimization options
        self.classifier_lr = config["classifier_lr"]
        self.classifier_lrf = config["classifier_lr_final"]
        self.classifier_wd = config['classifier_weight_decay']  # Put into weight decay option in configure_optimizer, l2 regularizer
        self.classifier_decay_after = config['classifier_decay_after']  # hyperparameter for when the lr decay should occur
        classifier_optimizer = config['classifier_optimizer']  # hyperparameter for when the lr decay should occur

        # optimizer options
        if classifier_optimizer == "SGD":
            self.classifier_optimizer = SGD
        elif classifier_optimizer == "AdamW":
            self.classifier_optimizer = AdamW

        if dataset == "mnist":
            input_size = (crbm_config["batch_size"], 28, 28)

        torch.set_default_dtype(torch.float64)

        output = self.get_test_rbm_output(input_size)

        # fully flatten each and concatenate along 0 dim
        out_flat = [torch.flatten(x, start_dim=1) for x in output]
        full_out = torch.cat(out_flat, dim=1)

        linear_size = full_out.shape[1]
        self.linear_size = linear_size

        classifier = [
            nn.Linear(linear_size, 10), # 10 possible digits
            nn.Softmax(dim=1)
        ]


        self.classifier = nn.Sequential(*classifier)

        if train_crbm:
            # Train RBM
            self.train_crbm(gpus=0, logger_dir=log_dir+"/crbm/", name="mnist_crbm")

    def configure_optimizers(self):
        optim = self.classifier_optimizer(self.parameters(), lr=self.classifier_lr, weight_decay=self.classifier_wd)
        decay_gamma = (self.classifier_lrf / self.classifier_lr) ** (1 / (self.classifier_epochs * (1 - self.classifier_decay_after)))
        decay_milestone = math.floor(self.classifier_decay_after * self.classifier_epochs)
        my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=[decay_milestone], gamma=decay_gamma)
        # my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=decay_gamma)
        return optim

    def load_crbm(self, checkpoint_file, load_data=True):
        self.binary_crbm = BinaryCRBM.load_from_checkpoint(checkpoint_file)
        if load_data:
            self.binary_crbm.setup()
        self.binary_crbm.eval()

    # def initialize_classifier

    def get_test_rbm_output(self, input_size):
        return self.binary_crbm.compute_output_v(torch.rand(input_size, device=self.device))

    def train_crbm(self, gpus=0, logger_dir="./tb_logs/", name="mnist_crbm"):
        logger = TensorBoardLogger(logger_dir, name=name)
        plt = Trainer(max_epochs=self.crbm_epochs, logger=logger, gpus=gpus, accelerator="ddp")  # distributed data-parallel
        plt.fit(self.binary_crbm)

    def train_dataloader(self):
        return self.binary_crbm.train_dataloader()

    def val_dataloader(self):
        return self.binary_crbm.val_dataloader()

    def training_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x.squeeze(1))
        preds = probs.argmax(1)
        train_loss = F.cross_entropy(probs, y)

        train_accuracy = balanced_accuracy_score(y.cpu(), preds.detach().cpu())
        self.log('train_accuracy', train_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x.squeeze(1))
        preds = probs.argmax(1)
        val_loss = F.cross_entropy(probs, y)

        val_accuracy = balanced_accuracy_score(y.cpu(), preds.detach().cpu())
        self.log('val_accuracy', val_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return val_loss

    def forward(self, input):
        with torch.no_grad():
            # attempting to remove crbm_variables from gradient calculation
            h_pos = self.binary_crbm.sample_from_inputs_h(self.binary_crbm.compute_output_v(input))
        h_flattened = torch.cat([torch.flatten(x, start_dim=1) for x in h_pos], dim=1)
        return self.classifier(h_flattened)







mnist_default_config = {"v_num": (28, 28), "q": 1, "epochs": 100, "classifier_epochs": 100, "crbm_log_dir": "", "seed": randint(0, 100000, 1)[0], "batch_size": 1000, "mc_moves": 4, "lr": 0.0001,
                        "lr_final": None, "decay_after": 0.75, "sequence_weights": None, "optimizer": "AdamW", "weight_decay": 0.02, "data_worker_num": 4, "fasta_file": "", "molecule": "dna",
                        "loss_type": "free_energy", "sample_type": "gibbs", "l1_2": 1000000.0, "lf": 5000.0, "ld": 10.0, "classifier_lr": 0.005, "classifier_lr_final": 0.0005, "classifier_decay_after": 0.75,
                        "classifier_weight_decay": 0.02, "classifier_optimizer": "AdamW",
                        "convolution_topology": {
                            "hidden20x20": {"number": 30, "kernel": (20, 20), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0}
                        }}

if __name__ == "__main__":
    # binary_crbm = BinaryCRBM(mnist_default_config, dataset="mnist")

    config = mnist_default_config
    config["epochs"] = 100
    config["classifier_epochs"] = 50

    binary_classifier = BinaryClassifier(mnist_default_config, dataset="mnist", train_crbm=False, debug=True)
    binary_classifier.train_crbm(1)

    # binary_classifier.load_crbm("./tb_logs/mnist_crbm/version_7/checkpoints/epoch=9-step=599.ckpt")

    # logger = TensorBoardLogger('./tb_logs/', name="mnist_classifier")
    # plt = Trainer(max_epochs=config['classifier_epochs'], logger=logger, gpus=1)  # gpus=1,
    # plt.fit(binary_classifier)

    # binary_crbm.setup()
    # td = binary_crbm.train_dataloader()

    # for i, batch in enumerate(td):
    #     if i > 0:
    #         break
    #     out = binary_crbm.training_step(batch, i)
    #     print("hi")

    # logger = TensorBoardLogger('./tb_logs/', name="mnist_crbm")
    # plt = Trainer(max_epochs=config['epochs'], logger=logger, gpus=1)  # gpus=1,
    # plt.fit(binary_crbm)