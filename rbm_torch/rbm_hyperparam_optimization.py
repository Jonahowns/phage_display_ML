import ray.tune as tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import os
import numpy as np
import math
import argparse

# local files
from rbm import RBM
import rbm_configs
import rbm_hyper_configs


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
        metric_columns=["train_loss", "train_pseudo_likelihood", "val_pseudo_likelihood", "training_iteration"])

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
        metric="val_pseudo_likelihood",
        mode="max",
        local_dir="../ray_results/",
        config=config,
        num_samples=num_samples,
        # search_alg=bayesopt,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_rbm_asha",
        checkpoint_score_attr="val_pseudo_likelihood",
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
        assert key not in ["sequence_weights", "seed", "q", "v_num", "fasta_file", "molecule"] # these you can't really change for now
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
        metric_columns=["train_loss", "train_pseudo_likelihood", "val_pseudo_likelihood", "training_iteration"])

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
        metric="val_pseudo_likelihood",
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
        checkpoint_score_attr="val_pseudo_likelihood",
        keep_checkpoints_num=1)

    print("Best hyperparameters found were: ", analysis.get_best_config(metric="val_pseudo_likelihood", mode="max"))

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
                    "val_pseudo_likelihood": "ptl/val_pseudo_likelihood",
                    "train_pseudo_likelihood": "ptl/train_pseudo_likelihood"
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
    os.environ["SLURM_JOB_NAME"] = "bash"   # server runs crash without this line (yay raytune)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # For debugging of cuda errors

    # Parse arguments
    parser = argparse.ArgumentParser(description="RBM Training on Provided Dataset")
    parser.add_argument('focus', type=str, help="Short string identifier for setting default options")
    parser.add_argument('dataset_fullpath', type=str, help="Full Path (not relative) of the fasta file used for training")
    parser.add_argument('samples', type=int, help="Number of Ray Tune Samples")
    parser.add_argument('epochs', type=int, help="Number of Training Iterations")
    parser.add_argument('gpus', type=int, help="Number of gpus per trial")
    parser.add_argument('cpus', type=int, help="Number of cpus per trial")
    parser.add_argument('data_workers', type=int, help="Number of data workers ")
    parser.add_argument('weights', type=str, help="Weight Sequences by their count?")
    args = parser.parse_args()

    # Set weights argument, used by RBM's config
    weights = None
    if args.weights in ["True", "true", "TRUE", "yes"]:
        weights = "fasta"  # All weights are already in the processed files

    file = os.path.basename(args.dataset_fullpath)
    name = file.split(".")[0]  # short specifier of round etc.

    # Weighted RBMS are put into separate tensorboard folders
    if weights is not None:
        name += "_w"

    # set default config
    if args.focus == "pig":
        if "c1" in name:
            config = rbm_configs.pig_c1_default_config
        elif "c2" in name:
            config = rbm_configs.pig_c2_default_config
    elif args.focus == "cov":
        config = rbm_configs.cov_default_config
    else:
        print("Focus Not Supported. Please Add focus to this script and add default config to configs.py")
        exit(-1)

    # Assign Run specific parameters
    config["fasta_file"] = args.dataset_fullpath
    config["sequence_weights"] = weights
    config["epochs"] = args.epochs
    config["data_worker_num"] = args.data_workers

    # Set search Parameters
    search = "asha"
    optimization = rbm_hyper_configs.hidden_opt  # From hyper_configs, Sets which hyperparameters are optimized

    # Launch Hyperparameter Optimization Task
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
