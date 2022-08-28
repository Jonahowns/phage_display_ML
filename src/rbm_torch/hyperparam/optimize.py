# import ray.tune as tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
# from ray_lightning import RayPlugin
from ray_lightning.tune import TuneReportCallback


from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import os
import numpy as np
import math
from ray import air, tune
# from ray.air import session
import torch
import argparse
from copy import deepcopy

# local files
from rbm_torch.models.rbm import RBM
from rbm_torch.models.crbm import CRBM
from rbm_torch.models.crbm_experimental import ExpCRBM
from rbm_torch.models.crbm_net import CRBM_net
from rbm_torch.models.rbm_experimental import ExpRBM

from rbm_torch.utils.utils import load_run_file
from rbm_torch.hyperparam.hyp_configs import hconfigs


def optimize(config, hyperparams_of_interest, method="asha", num_samples=10, num_epochs=10, gpus_per_trial=0, cpus_per_trial=1):
    """
       Launches ASHA Grid search or Population Based Hyperparameter Optimization
       :param config: Holds Hyperparameter Values of RBM
       :param hyperparams_of_interest: dictionary providing the hyperparameters and values to be altered
       during this hyperparameter optimization run. The hyperparmater name which must match the config exactly
       are the keys of the dictionary. The values are the corresponding tune distribution type with the corresponding range
       :param num_samples: How many trials will be run
       :param num_epochs: How many training iterations
       :param gpus_per_trial: Number of gpus to be dedicated PER trial (usually 0 or 1)
       :param cpus_per_trial: Number of cpus to be dedicated PER trial
       :return: Nothing
    """

    hyper_param_mut = {}
    for key, value in hyperparams_of_interest.items():
        assert key in config.keys()
        assert key not in ["sequence_weights", "seed", "q", "v_num", "raytune", "fasta_file",
                           "molecule"]  # these you can't really change for now
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

    model = config["model_type"]
    assert model in ["rbm", "crbm", "net_crbm", "exp_rbm", "exp_crbm"]

    if model == "rbm" or model == "exp_rbm":
        metric_cols = ["training_iteration", "train_loss", "train_pseudo_likelihood", "val_pseudo_likelihood", ]
        metric = "val_pseudo_likelihood"
        metric_mode = "max"
    elif model == "net_crbm":
        metric_cols = ["training_iteration", "train_free_energy", "train_mse", "val_free_energy", "val_mse"]
        metric = "val_mse"
        metric_mode = "min"
    elif model == "crbm" or model == "exp_crbm":
        metric_cols = ["training_iteration", "train_free_energy", "val_free_energy"]
        metric = "val_free_energy"
        metric_mode = "min"

    reporter = tune.CLIReporter(
        parameter_columns=list(hyper_param_mut.keys()),
        metric_columns=metric_cols)

    stopper = tune.stopper.MaximumIterationStopper(num_epochs)

    if method == "asha":
        scheduler = tune.schedulers.ASHAScheduler(
            max_t=num_epochs,
            grace_period=math.floor(num_epochs / 2),
            reduction_factor=2)

    elif method == "pbt":
        scheduler = tune.schedulers.PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=10,
            hyperparam_mutations=hyper_param_mut)

    # bayesopt = BayesOptSearch(metric="mean_loss", mode="min")
    train_with_params = tune.with_parameters(
                            ray_train,
                            num_epochs=num_epochs,
                            num_gpus=gpus_per_trial,
                            num_cpus=cpus_per_trial)

    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}

    tuner = tune.Tuner(
        tune.with_resources(train_with_params, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            metric=metric,
            mode=metric_mode,
            num_samples=num_samples),
        run_config=air.RunConfig(
            progress_reporter=reporter,
            name=f"tune_{model}_{method}",
            stop=stopper,
        ),
        param_space=config
    )

    results = tuner.fit()

    # print("Best hyperparameters found were: ", results.get_best_result(metric=metric, mode=metric_mode))


def ray_train(config, num_epochs=10, num_gpus=0, num_cpus=0):
    model = config["model_type"]
    assert model in ["rbm", "crbm", "net_crbm", "exp_rbm", "exp_crbm"]

    if model == "rbm" or model == "exp_rbm":
        mets = {"train_loss": "ptl/train_loss",
                "val_pseudo_likelihood": "ptl/val_pseudo_likelihood",
                "train_pseudo_likelihood": "ptl/train_pseudo_likelihood"
                }
    elif model == "net_crbm":
        mets = {"train_free_energy": "ptl/train_free_energy",
                "train_mse": "ptl/train_fitness_mse",
                "val_mse": "ptl/val_fitness_mse",
                "val_free_energy": "ptl/val_free_energy"
                }
    elif model == "crbm" or model == "exp_crbm":
        mets = {"train_free_energy": "ptl/train_free_energy",
                "val_free_energy": "ptl/val_free_energy"
                }

    if num_gpus == 0:
        device_num = num_cpus
        acc = "cpu"
    elif num_gpus == 1:
        device_num = num_gpus
        acc = "gpu"
    else:
        device_num = num_gpus
        acc = "ddp"

    trainer = Trainer(
        # default_root_dir="./checkpoints/",
        max_epochs=num_epochs,
        devices=device_num,
        accelerator=acc,
        enable_progress_bar=False,
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="tb", version="."),
        callbacks=[
            TuneReportCheckpointCallback(mets, filename="checkpoint", on="validation_end")
        ]
    )

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

    # if session.get_checkpoint():
    #     print("Loading from checkpoint.")
    #     loaded_checkpoint = session.get_checkpoint()
    #     with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
    #         path = os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
    #         checkpoint = torch.load(path)
    #         mod.load_state_dict(checkpoint["model_state_dict"])
    #         step = checkpoint["step"]

    trainer.fit(mod)


# if __name__ == '__main__':
    # os.environ["SLURM_JOB_NAME"] = "bash"   # server runs crash without this line (yay raytune)
    # # os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # For debugging of cuda errors
    #
    # # Parse arguments
    # parser = argparse.ArgumentParser(description="Hyperparameter Optimization on Provided Dataset")
    # parser.add_argument('runfile', type=str, help="File holding all the necessary info for training the model")
    # parser.add_argument('hparam_config_name', type=str, help="Name of hyperparameter optimization dictionary in hyp_configs.py")
    # parser.add_argument('optimization_method', type=str, help="Which hparam optimization method to use, asha or pbt?")
    # parser.add_argument('samples', type=int, help="Number of Ray Tune Samples")
    # parser.add_argument('epochs', type=int, help="Number of Training Iterations")
    # parser.add_argument('cpus', type=int, help="Number of CPUs PER Trial")
    # parser.add_argument('gpus', type=int, help="Number of GPUs PER Trial")
    #
    # args = parser.parse_args()
    #
    # run_data, config = load_run_file(args.runfile)
    # config["model_type"] = run_data["model_type"]
    #
    # # Set search Parameters
    # search_method = args.optimization_method
    # optimization_dict = hconfigs[run_data["model_type"]][args.hparam_config_name]  # From hyper_configs, Sets which hyperparameters are optimized
    #
    # # Take our very compressed version of the convolution topology and write it out in a form the model will understand
    # full_convolution_topology = []
    # if "convolution_topology" in optimization_dict.keys():
    #     for convolution_top in optimization_dict["convolution_topology"]["choice"]:
    #         convolution_top_expanded = {}  # temporary convolution topology that is stored as a choice in full_convolution_topology
    #         for convolution in convolution_top:
    #             hidden_number, kernelx = convolution
    #             convolution_top_expanded[f"hidden_{kernelx}"] = {"number": hidden_number, "kernel": (kernelx. config["q"]), "stride": (1, 1),
    #                                                              "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0}
    #         full_convolution_topology.append(deepcopy(convolution_top_expanded))
    #
    # # Write out our expanded convolution topology to the optimization dictionary
    # optimization_dict["convolution_topology"]["choice"] = full_convolution_topology
    #
    # # Launch Hyperparameter Optimization Task
    # optimize(config,
    #          optimization_dict,
    #          num_samples=args.samples,
    #          num_epochs=args.epochs,
    #          gpus_per_trial=args.gpus,
    #          cpus_per_trial=args.cpus)
