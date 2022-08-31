# import ray.tune as tune
# from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
# from pytorch_lightning import Trainer
# from pytorch_lightning.loggers import TensorBoardLogger
import os
# import numpy as np
# import math
import argparse
from copy import deepcopy
import optuna

from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
from multiprocessing import Manager
from joblib import parallel_backend

# # local files
# from rbm_torch.models.rbm import RBM
# from rbm_torch.models.crbm import CRBM
# from rbm_torch.models.crbm_experimental import ExpCRBM
# from rbm_torch.models.crbm_net import CRBM_net
# from rbm_torch.models.rbm_experimental import ExpRBM

from rbm_torch.utils.utils import load_run_file
from rbm_torch.hyperparam.hyp_configs import hconfigs
# from rbm_torch.hyperparam.optimize import optimize
from rbm_torch.hyperparam.optimize_optuna import Objective, directions

if __name__ == '__main__':
    os.environ["SLURM_JOB_NAME"] = "bash"   # server runs crash without this line (yay raytune)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # For debugging of cuda errors

    # Parse arguments
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization on Provided Dataset")
    parser.add_argument('runfile', type=str, help="File holding all the necessary info for training the model")
    parser.add_argument('hparam_config_name', type=str, help="Name of hyperparameter optimization dictionary in hyp_configs.py")
    # parser.add_argument('optimization_method', type=str, help="Which hparam optimization method to use, asha or pbt?")
    parser.add_argument('trials', type=int, help="Number of Optuna trials")
    parser.add_argument('epochs', type=int, help="Number of Training Iterations")
    parser.add_argument('gpus', type=int, help="Number of GPUs Total, each trial trained on separate gpu")

    args = parser.parse_args()

    run_data, config = load_run_file(args.runfile)
    config["model_type"] = run_data["model_type"]
    config["fasta_file"] = os.path.join(os.getcwd(), config["fasta_file"])
    config["model_name"] = f"{args.hparam_config_name}_{run_data['model_name']}"
    config["server_model_dir"] = run_data["server_model_dir"]
    config["epochs"] = args.epochs

    # Set search Parameters
    optimization_dict = hconfigs[run_data["model_type"]][args.hparam_config_name]  # From hyper_configs, Sets which hyperparameters are optimized

    # Take our very compressed version of the convolution topology and write it out in a form the model will understand
    full_convolution_topology = []
    if "convolution_topology" in optimization_dict.keys():
        tune_key = list(optimization_dict["convolution_topology"].keys())[0]
        for convolution_top in optimization_dict["convolution_topology"][tune_key]:
            convolution_top_expanded = {}  # temporary convolution topology that is stored as a choice in full_convolution_topology
            for convolution in convolution_top:
                hidden_number, kernelx = convolution
                convolution_top_expanded[f"hidden_{kernelx}"] = {"number": hidden_number, "kernel": (kernelx, config["q"]), "stride": (1, 1),
                                                                 "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0}
            full_convolution_topology.append(deepcopy(convolution_top_expanded))

        # Write out our expanded convolution topology to the optimization dictionary
        optimization_dict["convolution_topology"][tune_key] = full_convolution_topology

    # if model directory already exists, increment by one and add to new directory name
    if os.path.isdir(os.path.join(os.getcwd(), config["server_model_dir"], config["model_name"])):
        i = 1
        postfix = f"_{i}"
        while os.path.isdir(os.path.join(os.getcwd(), config["server_model_dir"], config["model_name"] + postfix)):
            i += 1
            postfix = f"_{i}"

        config["model_name"] = config["model_name"] + postfix

    pruner_warm_up_steps = 200

    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=pruner_warm_up_steps, interval_steps=100, n_min_trials=8)

    study = optuna.create_study(
        study_name=config["model_name"],
        direction=directions[config["model_type"]],
        pruner=pruner
    )

    n_gpu = int(args.gpus)
    # config["gpus"] = n_gpu

    cluster = LocalCUDACluster()
    client = Client(cluster)

    with Manager() as manager:

        # Initialize the queue by adding available GPU IDs.
        gpu_queue = manager.Queue()
        for i in range(n_gpu):
            gpu_queue.put(i)
        with parallel_backend("dask", n_jobs=n_gpu):
            study.optimize(Objective(gpu_queue, optimization_dict, config, args.epochs), n_trials=args.trials, n_jobs=n_gpu)

    # study.optimize(lambda trial: Objective(gpu_queue, trial, optimization_dict, config, args.epochs), n_trials=args.trials, n_jobs=config["gpus"])

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))