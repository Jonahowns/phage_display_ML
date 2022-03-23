# import sklearn
import ray.tune as tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
# from ray.tune.suggest.bayesopt import BayesOptSearch
# import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import os
import numpy as np
# import multiprocessing as mp
import math
import argparse

from rbm import RBM


def tune_asha_search(config, hyperparams_of_interest, num_samples=10, num_epochs=10, gpus_per_trial=0, cpus_per_trial=1):
    hyper_param_mut = {}
    for key, value in hyperparams_of_interest.items():
        assert key in config.keys()
        assert key not in ["sequence_weights", "seed", "q", "v_num", "raytune", "fasta_file", "molecule"]  # these you can't really change for now
        # This dictionary contains type of hyperparameter it is and the parameters associated with each type
        for subkey, subval in value.items():
            if subkey == "uniform":
                config[key] = tune.uniform(subval[0], subval[1])
                hyper_param_mut[key] = lambda: np.random.uniform(subval[0], subval[1])
            elif subkey == "loguniform":
                config[key] = tune.loguniform(subval[0], subval[1])
                hyper_param_mut[key] = lambda: np.exp(np.random.uniform(subval[0], subval[1]))
            elif subkey == "choice":
                config[key] = tune.choice(subval)
                hyper_param_mut[key] = subval
            elif subkey == "grid":
                config[key] = tune.grid_search(subval)
                hyper_param_mut[key] = subval

    scheduler = tune.schedulers.ASHAScheduler(
        max_t=num_epochs,
        grace_period=math.floor(num_epochs/2),
        reduction_factor=2)

    reporter = tune.CLIReporter(
        parameter_columns=list(hyper_param_mut.keys()),
        metric_columns=["train_loss", "train_psuedolikelihood", "val_psuedolikelihood", "training_iteration"])

    # bayesopt = BayesOptSearch(metric="mean_loss", mode="min")
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
        local_dir="../ray_results/",
        config=config,
        num_samples=num_samples,
        # search_alg=bayesopt,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_rbm_asha",
        checkpoint_score_attr="val_psuedolikelihood",
        keep_checkpoints_num=1)

    print("Best hyperparameters found were: ", analysis.best_config)

def pbt_rbm(config, hyperparams_of_interest, num_samples=10, num_epochs=10, gpus_per_trial=0, cpus_per_trial=1):

    '''
    Launches Population Based Hyperparameter Optimization

    :param config: Holds Hyperparameter Values of RBM
    :param hyperparams_of_interest: dictionary providing the hyperparameters and values to be altered
    during this hyperparameter optimization run. The hyperparmater name which must match the config exactly
    are the keys of the dictionary. The values are the corresponding tune distribution type with the corresponding range
    :param num_samples: How many trials will be run
    :param num_epochs: How many training iterations
    :param gpus_per_trial: Number of gpus to be dedicated PER trial (usually 0 or 1)
    :param cpus_per_trial: Number of cpus to be dedicated PER trial
    :return: Nothing
    '''

    hyper_param_mut = {}

    for key, value in hyperparams_of_interest.items():
        assert key in config.keys()
        assert key not in ["sequence_weights", "seed", "q", "v_num", "raytune", "fasta_file", "molecule"] # these you can't really change for now
        # This dictionary contains type of hyperparameter it is and the parameters associated with each type
        for subkey, subval in value.items():
            if subkey == "uniform":
                config[key] = tune.uniform(subval[0], subval[1])
                hyper_param_mut[key] = lambda: np.random.uniform(subval[0], subval[1])
            elif subkey == "loguniform":
                config[key] = tune.loguniform(subval[0], subval[1])
                hyper_param_mut[key] = lambda: np.exp(np.random.uniform(subval[0], subval[1]))
            elif subkey == "choice":
                config[key] = tune.choice(subval)
                hyper_param_mut[key] = subval
            elif subkey == "grid":
                config[key] = tune.grid_search(subval)
                hyper_param_mut[key] = subval


    scheduler = tune.schedulers.PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=10,
        hyperparam_mutations=hyper_param_mut)

    reporter = tune.CLIReporter(
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
        local_dir="../ray_results/",
        config=config,
        num_samples=num_samples,
        name="tune_pbt_rbm",
        scheduler=scheduler,
        progress_reporter=reporter,
        verbose=1,
        stop=stopper,
        # export_formats=[ExportFormat.MODEL],
        checkpoint_score_attr="val_psuedolikelihood",
        keep_checkpoints_num=1)

    print("Best hyperparameters found were: ", analysis.get_best_config(metric="val_psuedolikelihood", mode="max"))

def train_rbm(config, checkpoint_dir=None, num_epochs=10, num_gpus=0):
    trainer = Trainer(
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
    ### If using Grid
    # Number of Trials = Number of Samples * Number of Grid Parameter Combinations
    ### If using Random Choice Only
    # Number of Trials = Number of Samples

    ######## Define Hyperparameters Here ########
    # All hyperparameters that can be optimized via population based Tuning

    ### All Options that are currently supported
    # sample_hyperparams_of_interest = {
    #     "h_num": {"grid": [10, 120, 250, 1000]},  # number of hidden units, can be variable
    #     "batch_size": {"choice": [5000, 10000, 20000]},
    #     "mc_moves": {"grid": [4, 8]},
    #     "lr": {"uniform": [1e-5, 1e-1]},
    #     "decay_after": {"grid": [0.5, 0.75, 0.9]},  # Fraction of epochs to have exponential decay after
    #     "loss_type": {"choice": ["free_energy", "energy"]},
    #     "sample_type": {"choice": ["gibbs", "pt"]},
    #     "optimizer": {"choice": ["AdamW", "SGD", "Adagrad"]},
    #     "epochs": {"choice": [100, 200, 1000]},
    #     "weight_decay": {"uniform": [1e-5, 1e-1]},
    #     "l1_2": {"uniform": [0.15, 0.6]},
    #     "lf": {"uniform": [1e-5, 1e-2]},
    # }

    # Some of the more pertinent hyper parameters with direct effect on weights, loss, and gradient
    # reg_opt = {
    #     "weight_decay": {"uniform": [1e-5, 1e-1]},
    #     "l1_2": {"uniform": [0.15, 0.6]},
    #     "lf": {"uniform": [1e-5, 1e-2]},
    #     "lr": {"uniform": [1e-5, 1e-1]}
    # }

    # hyperparams of interest
    hidden_opt = {
        "h_num": {"grid": [20, 60, 100, 200]},
        # "h_num": {"grid": [5, 10, 15, 20]},
        # "batch_size": {"choice": [10000, 20000]},
        "l1_2": {"uniform": [0.15, 0.6]},
        "lr": {"choice": [1e-3, 6e-3]},
        "lf": {"uniform": [1e-4, 1e-2]}
        # "mc_moves": {"choice": [4, 8]},
    }


    # local Test
    os.environ["SLURM_JOB_NAME"] = "bash"   # server runs crash without this line (yay raytune)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # For debugging of cuda errors

    # Parse arguments
    parser = argparse.ArgumentParser(description="RBM Training on Provided Dataset")
    parser.add_argument('dataset_fullpath', type=str, help="Full Path (not relative) of the fasta file used for training")
    parser.add_argument('visible_num', type=int, help="Number of Visible Nodes")
    parser.add_argument('molecule', type=str, help="DNA, RNA, or Protein")
    parser.add_argument('samples', type=int, help="Number of Ray Tune Samples")
    parser.add_argument('epochs', type=int, help="Number of Training Iterations")
    parser.add_argument('gpus', type=int, help="Number of gpus per trial")
    parser.add_argument('cpus', type=int, help="Number of cpus per trial")
    parser.add_argument('data_workers', type=int, help="Number of data workers ")
    parser.add_argument('weights', type=bool, help="Weight Sequences by their count?")
    parser.add_argument('gaps', type=bool, help="Gaps in the alignment?")
    args = parser.parse_args()

    search = "asha"  # must be either pbt or asha, the optimization method
    optimization = hidden_opt  # which hyperparameter dictionary to use for actual run

    if args.gaps is True:
        molecule_states = {"dna": 5, "rna": 5, "protein": 21}  # with gaps
    else:
        molecule_states = {"dna": 4, "rna": 4, "protein": 20}

    # Default Values, the optimization dictionary replaces the default values
    config = {"fasta_file": args.dataset_fullpath,
              "molecule": args.molecule,
              "h_num": 10,  # number of hidden units, can be variable
              "v_num": args.visible_num,
              "q": molecule_states[args.molecule],
              "batch_size": 20000,
              "mc_moves": 6,
              "seed": 38,
              "lr": 0.0065,
              "lr_final": None,
              "decay_after": 0.75,
              "loss_type": "free_energy",
              "sample_type": "gibbs",
              "sequence_weights": None,
              "optimizer": "AdamW",
              "epochs": args.epochs,
              "weight_decay": 0.001,  # l2 norm on all parameters
              "l1_2": 0.185,
              "lf": 0.002,
              "data_worker_num": args.data_workers
              }

    if search == "pbt":
        pbt_rbm(config,
                optimization,
                num_samples=args.samples,
                num_epochs=args.epochs,
                gpus_per_trial=args.gpus,
                cpus_per_trial=args.cpus)

    elif search == 'asha':
        tune_asha_search(config,
                         optimization,
                         num_samples=args.samples,
                         num_epochs=args.epochs,
                         gpus_per_trial=args.gpus,
                         cpus_per_trial=args.cpus)
