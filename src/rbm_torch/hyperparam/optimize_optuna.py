# import ray.tune as tune
# from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
# from ray_lightning import RayPlugin
# from ray_lightning.tune import TuneReportCallback

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import os
# import numpy as np
# import math
# from ray import air, tune
# from ray.air import session
# import torch
# import argparse
# from copy import deepcopy

# local files
from rbm_torch.models.rbm import RBM
from rbm_torch.models.crbm import CRBM
from rbm_torch.models.crbm_experimental import ExpCRBM
from rbm_torch.models.crbm_net import CRBM_net
from rbm_torch.models.rbm_experimental import ExpRBM

from contextlib import contextmanager
import multiprocessing
# from rbm_torch.utils.utils import load_run_file
# from rbm_torch.hyperparam.hyp_configs import hconfigs


import optuna
from optuna.integration import PyTorchLightningPruningCallback


def objective(trial, hyperparams_of_interest, config, epochs, device, postfix=None):
    hyper_params = {}
    for key, value in hyperparams_of_interest.items():
        assert key in config.keys()
        assert key not in ["sequence_weights", "seed", "q", "v_num", "raytune", "fasta_file",
                           "molecule"]  # these you can't really change for now
        # This dictionary contains type of hyperparameter it is and the parameters associated with each type
        for subkey, subval in value.items():
            if subkey == "uniform" :
                config[key] = trial.suggest_uniform(key, subval[0], subval[1])
                # config[key] = tune.uniform(subval[0], subval[1])
            elif subkey == "loguniform":
                config[key] = trial.suggest_loguniform(key, subval[0], subval[1])
                # hyper_param_mut[key] = lambda: np.exp(np.random.uniform(subval[0], subval[1]))
            elif subkey == "choice":
                config[key] = trial.suggest_categorical(key, subval)
                # hyper_param_mut[key] = subval
            elif subkey == "grid":
                config[key] = trial.suggest_categorical(key, subval)
                # hyper_param_mut[key] = subval
            hyper_params[key] = config[key]

    model = config["model_type"]
    assert model in ["rbm", "crbm", "net_crbm", "exp_rbm", "exp_crbm"]

    if model == "rbm" or model == "exp_rbm":
        # metric_cols = ["training_iteration", "train_loss", "train_pseudo_likelihood", "val_pseudo_likelihood", ]
        metric = "ptl/val_pseudo_likelihood"
        # metric_mode = "max"
    elif model == "net_crbm":
        # metric_cols = ["training_iteration", "train_free_energy", "train_mse", "val_free_energy", "val_mse"]
        metric = "ptl/val_fitness_mse"
        # metric_mode = "min"
    elif model == "crbm" or model == "exp_crbm":
        # metric_cols = ["training_iteration", "train_free_energy", "val_free_energy"]
        metric = "ptl/val_free_energy"
        # metric_mode = "min"

    if postfix:
        config["model_name"] = config["model_name"] + f"_{postfix}"

    if model == "rbm":
        mod = RBM(config, precision=config["precision"])
    elif model == "exp_rbm":
        mod = ExpRBM(config, precision=config["precision"])
    elif model == "crbm":
        mod = CRBM(config, precision=config["precision"])
    elif model == "exp_crbm":
        mod = ExpCRBM(config, precision=config["precision"])
    elif model == "net_crbm":
        mod = CRBM_net(config, precision=config["precision"])

    num_gpus = config["gpus"]
    if num_gpus == 0:
        device_num = [os.cpu_count()]
        acc = "cpu"
    elif num_gpus == 1:
        device_num = [device]
        acc = "gpu"
    else:
        device_num = [device]
        acc = "ddp"

    trainer = Trainer(
        # default_root_dir="./checkpoints/",
        max_epochs=epochs,
        devices=device_num,
        accelerator=acc,
        enable_progress_bar=False,
        enable_checkpointing=True,
        logger=TensorBoardLogger(
            save_dir=os.path.join(os.getcwd(), config["server_model_dir"]), name=config["model_name"], version=trial.number),
        callbacks=[PyTorchLightningPruningCallback(trial, monitor=metric)],
    )

    # hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims)
    trainer.logger.log_hyperparams(hyper_params)
    trainer.fit(mod)

    return trainer.callback_metrics[metric].item()


# Simple implementation for multi gpu optimization
# adapted from https://github.com/optuna/optuna/issues/1365
class Objective:
    def __init__(self, gpu_queue, hyperparams_of_interest, config, epochs, postfix=None):
        # Shared queue to manage GPU IDs.
        self.gpu_queue = gpu_queue
        self.hyperparams_of_interest = hyperparams_of_interest
        self.config = config
        self.epochs = epochs
        self.postfix = postfix

    def __call__(self, trial):
        # Fetch GPU ID for this trial.
        gpu_id = self.gpu_queue.get()

        # Please write actual objective function here
        value = objective(trial, self.hyperparams_of_interest, self.config, self.epochs, gpu_id, postfix=self.postfix)
        # Return GPU ID to the queue.
        self.gpu_queue.put(gpu_id)

        # return metric
        return value
