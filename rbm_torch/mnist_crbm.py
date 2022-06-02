from crbm import CRBM

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.profiler import SimpleProfiler, PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.datasets import MNIST

from torch.utils.data import Dataset, DataLoader

# q should be 1 on config
class BinaryCRBM(CRBM):
    def __init__(self, config):
        super().__init__(config)


    def transform_v(self, I):
        return (I+getattr(self, "fields"))>0

    def energy_v(self, config, remove_init=False):
        if remove_init:
            return -torch.dot(config, getattr(self, "fields") - getattr(self, "fields0"))
        else:
            return -torch.dot(config, getattr(self, "fields"))

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
            return torch.log(1 + torch.exp(getattr(self, "fields").unsqueeze(0) + inputs)).sum(1)
        else:
            return torch.log(1 + torch.exp((beta * getattr(self, "fields")).unsqueeze(0) + (1 - beta) * getattr(self, "fields0")).unsqueeze(0) + beta * inputs).sum(1)

    def compute_output_v(self, X):
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

    def mean_v(self, psi, beta=1):
        if beta == 1:
            return torch.special.expit(psi + getattr(self, "fields").unsqueeze(0))
        else:
            return torch.special.expit(beta * psi + getattr(self, "fields0").unsqueeze(0) + beta * (getattr(self, "fields").unsqueeze(0) - getattr(self, "fields0").unsqueeze(0)))

    def sample_from_inputs_v(self, psi, beta=1):
        return torch.randn(psi.shape) < self.mean_v(psi, beta=beta)

    def setup(self, stage=None):
        PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
        self.train_reader = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
        self.val_reader = MNIST(PATH_DATASETS, train=False, download=True, transform=transforms.ToTensor())

    def train_dataloader(self, init_fields=True):
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

        x, y = batch
        output = self(x)
        loss = F.cross_entropy(self(x), y)
        return loss



    def forward(self, X):
        with torch.no_grad():  # only use last sample for gradient calculation, Enabled to minimize memory usage, hopefully won't have much effect on performance
            fantasy_v, fantasy_h = self.markov_step(X)
            for _ in range(self.mc_moves - 2):
                fantasy_v, fantasy_h = self.markov_step(fantasy_v)

        V_neg, fantasy_h = self.markov_step(fantasy_v)

        # V_neg, h_neg, V_pos, h_pos
        return V_neg, self.sample_from_inputs_h(self.compute_output_v(V_neg)), V_pos_ohe, self.sample_from_inputs_h(self.compute_output_v(V_pos_ohe))





