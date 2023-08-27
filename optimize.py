# import ray.tune as tune
# from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
# from pytorch_lightning import Trainer
# from pytorch_lightning.loggers import TensorBoardLogger
import os
# import numpy as np
# import math
import argparse
from copy import deepcopy

# # local files
# from rbm_torch.models.rbm import RBM
# from rbm_torch.models.crbm import CRBM
# from rbm_torch.models.crbm_experimental import ExpCRBM
# from rbm_torch.models.crbm_net import CRBM_net
# from rbm_torch.models.rbm_experimental import ExpRBM

from rbm_torch.utils.utils import load_run
from rbm_torch.hyperparam.hyp_configs import hconfigs
from rbm_torch.hyperparam.optimize import optimize

if __name__ == '__main__':
    os.environ["SLURM_JOB_NAME"] = "bash"   # server runs crash without this line (yay raytune)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # For debugging of cuda errors

    # Parse arguments
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization on Provided Dataset")
    parser.add_argument('runfile', type=str, help="File holding all the necessary info for training the model")
    parser.add_argument('hparam_config_name', type=str, help="Name of hyperparameter optimization dictionary in hyp_configs.py")
    parser.add_argument('optimization_method', type=str, help="Which hparam optimization method to use, asha or pbt?")
    parser.add_argument('samples', type=int, help="Number of Ray Tune Samples")
    parser.add_argument('epochs', type=int, help="Number of Training Iterations")
    parser.add_argument('cpus', type=int, help="Number of CPUs PER Trial")
    parser.add_argument('gpus', type=int, help="Number of GPUs PER Trial")

    args = parser.parse_args()

    run_data, config = load_run(args.runfile)
    config["model_type"] = run_data["model_type"]
    config["fasta_file"] = os.path.join(os.getcwd(), config["fasta_file"])

    # Set search Parameters
    search_method = args.optimization_method
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

    # Launch Hyperparameter Optimization Task
    optimize(config,
             optimization_dict,
             num_samples=args.samples,
             num_epochs=args.epochs,
             gpus_per_trial=args.gpus,
             cpus_per_trial=args.cpus)