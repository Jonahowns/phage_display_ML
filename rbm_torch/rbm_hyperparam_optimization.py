import sklearn
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.suggest.bayesopt import BayesOptSearch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import os
import numpy as np
import multiprocessing as mp
import math

from rbm_test import RBM


# fasta_file #
def pbt_rbm(fasta_file, hyperparams_of_interest, num_samples=10, num_epochs=10, gpus_per_trial=0, cpus_per_trial=1, data_workers_per_trial=None):

    '''
    Launches Population Based Hyperparameter Optimization

    :param fasta_file: is the data file, must provide absolute path because of ray tune's nonsense
    :param hyperparams_of_interest: dictionary providing the hyperparameters and values will be altered
    during this hyperparameter optimization run. The hyperparmater name which must match the config exactly
    are the keys of the dictionary. The values are the corresponding tune distribution type with the corresponding range
    :param num_samples: How many trials will be run
    :param num_epochs: How many training iterations
    :param gpus_per_trial: Number of gpus to be dedicated PER trial (usually 0 or 1)
    :param cpus_per_trial: Number of cpus to be dedicated PER trial
    :return: Nothing
    '''

    # Default Values
    config = {"fasta_file": fasta_file,
              "molecule": "protein",
              "h_num": 10,  # number of hidden units, can be variable
              "v_num": 27,
              "q": 21,
              "batch_size": 10000,
              "mc_moves": 6,
              "seed": 38,
              "lr": 0.0065,
              "lr_final": None,
              "decay_after": 0.75,
              "loss_type": "free_energy",
              "sample_type": "gibbs",
              "sequence_weights": None,
              "optimizer": "AdamW",
              "epochs": num_epochs,
              "weight_decay": 0.001,  # l2 norm on all parameters
              "l1_2": 0.185,
              "lf": 0.002,
              }

    if data_workers_per_trial:
        config['data_worker_num'] = data_workers_per_trial

    hyper_param_mut = {}

    for key, value in hyperparams_of_interest.items():
        assert key in config.keys()
        assert key not in ["sequence_weights", "seed", "q", "v_num", "raytune", "fasta_file", "molecule"] # these you can't really change for now
        # This dictionary contains type of hyperparameter it is and the parameters associated with each type
        for subkey, subval in value.items():
            if subkey == "uniform":
                config[key] = tune.uniform(subval[0], subval[1])
                hyper_param_mut[key] = lambda: np.random.uniform(subval[0], subval[1])
            elif subkey == "choice":
                config[key] = tune.choice(subval)
                hyper_param_mut[key] = subval
            elif subkey == "grid":
                config[key] = tune.grid_search(subval)
                hyper_param_mut[key] = subval


    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=10,
        hyperparam_mutations=hyper_param_mut)

    reporter = CLIReporter(
        parameter_columns=list(hyper_param_mut.keys()),
        metric_columns=["train_loss", "train_psuedolikelihood", "val_psuedolikelihood", "training_iteration"])

    stopper = tune.stopper.MaximumIterationStopper(num_epochs)

    analysis = tune.run(
        tune.with_parameters(
            train_rbm,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial
        },
        metric="val_psuedolikelihood",
        mode="max",
        local_dir="./ray_results/",
        config=config,
        num_samples=num_samples,
        name="tune_pbt_rbm",
        scheduler=scheduler,
        progress_reporter=reporter,
        verbose=1,
        stop=stopper,
        # export_formats=[ExportFormat.MODEL],
        checkpoint_score_attr="val_psuedolikelihood",
        keep_checkpoints_num=4)

    print("Best hyperparameters found were: ", analysis.get_best_config(metric="val_psuedolikelihood", mode="max"))



def train_rbm(config, checkpoint_dir=None, num_epochs=10, num_gpus=0):
    trainer = pl.Trainer(
        # default_root_dir="./checkpoints/",
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=num_gpus,
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="tb", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCheckpointCallback(
                metrics={
                    "train_loss": "ptl/train_loss",
                    "val_psuedolikelihood": "ptl/val_psuedolikelihood",
                    "train_psuedolikelihood": "ptl/train_psuedolikelihood"
                },
                filename="checkpoint",
                on="validation_end")
        ]
    )

    if checkpoint_dir:
        # Currently, this leads to errors:
        # model = LightningMNISTClassifier.load_from_checkpoint(
        #     os.path.join(checkpoint, "checkpoint"))
        # Workaround:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        rbm = RBM.load_from_checkpoint(checkpoint)
    else:
        rbm = RBM(config)

    trainer.fit(rbm)
    
    
    
if __name__ == '__main__':

    # All hyperparameters that can be optimized via population based Tuning
    sample_hyperparams_of_interest = {
        "h_num": {"grid": [10, 120, 250, 1000]},  # number of hidden units, can be variable
        "batch_size": {"choice": [5000, 10000, 20000]},
        "mc_moves": {"grid": [4, 8]},
        "lr": {"uniform": [1e-5, 1e-1]},
        "decay_after": {"grid": [0.5, 0.75, 0.9]},  # Fraction of epochs to have exponential decay after
        "loss_type": {"choice": ["free_energy", "energy"]},
        "sample_type": {"choice": ["gibbs", "pt"]},
        "optimizer": {"choice": ["AdamW", "SGD", "Adagrad"]},
        "epochs": {"choice": [100, 200, 1000]},
        "weight_decay": {"uniform": [1e-5, 1e-1]},
        "l1_2": {"uniform": [0.15, 0.6]},
        "lf": {"uniform": [1e-5, 1e-2]},
    }

    # hyperparams of interest
    hidden_opt = {
        "h_num": {"grid": [60, 120, 250, 500]},
        # "h_num": {"grid": [10, 20, 30, 40]},
        "batch_size": {"choice": [5000, 10000, 20000]},
        "mc_moves": {"choice": [4, 8]},
    }

    reg_opt = {
        "weight_decay": {"uniform": [1e-5, 1e-1]},
        "l1_2": {"uniform": [0.15, 0.6]},
        "lf": {"uniform": [1e-5, 1e-2]},
        "lr": {"uniform": [1e-5, 1e-1]}
    }

    ### If using Grid
    # Number of Trials = Number of Samples * Number of Grid Parameter Combinations

    ### If using Random Choice Only
    # Number of Trials = Number of Samples


    # local Test
    os.environ["SLURM_JOB_NAME"] = "bash"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # For debugging of random cuda errors

    # pbt_rbm("/home/jonah/PycharmProjects/phage_display_ML/rbm_torch/lattice_proteins_verification/Lattice_Proteins_MSA.fasta",
    #         hidden_opt, 1, 20, 1, 1)

    # pbt_rbm("/scratch/jprocyk/machine_learning/phage_display_ML/invivo/sham2_ipsi_c1.fasta",
    #         hidden_opt,
    #         num_samples=1,
    #         num_epochs=30,
    #         gpus_per_trial=1,
    #         cpus_per_trial=12,
    #         data_workers_per_trial=6)

    # pbt_rbm("/home/jonah/PycharmProjects/phage_display_ML/pig_tissue/b3_c1.fasta",
    #         hidden_opt, num_samples=1, num_epochs=2, gpus_per_trial=1, cpus_per_trial=1, data_workers_per_trial=3)
    # Server Run

    # pbt_rbm("/scratch/jprocyk/machine_learning/phage_display_ML/rbm_torch/lattice_proteins_verification/Lattice_Proteins_MSA.fasta",
    #         hidden_opt, 1, 150, 1, 2)
    # pbt_rbm("/scratch/jprocyk/machine_learning/phage_display_ML/pig_tissue/b3_c1.fasta",
    #         hidden_opt, num_samples=1, num_epochs=150, gpus_per_trial=1, cpus_per_trial=12, data_workers_per_trial=12)

    pbt_rbm("/scratch/jprocyk/machine_learning/phage_display_ML/pig_tissue/b3_c1.fasta",
            hidden_opt,
            num_samples=1,
            num_epochs=150,
            gpus_per_trial=1,
            cpus_per_trial=12,
            data_workers_per_trial=12)
