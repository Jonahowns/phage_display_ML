from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

import argparse
# import json
# import numpy as np
# from numpy.random import randint
import os

from rbm_torch.models.rbm import RBM
from rbm_torch.models.crbm import CRBM
from rbm_torch.models.crbm_experimental import ExpCRBM, pCRBM
from rbm_torch.models.crbm_net import CRBM_net
from rbm_torch.models.rbm_experimental import ExpRBM
from rbm_torch.models.pool_crbm import pool_CRBM
from rbm_torch.models.pool_crbm_classification import pool_class_CRBM
# from rbm_torch.models.pool_crbm_regression import pool_regression_CRBM
from rbm_torch.models.composition_crbm import comp_CRBM

from rbm_torch.utils.utils import load_run_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runs Training Procedure from a json run file")
    parser.add_argument('runfile', type=str, help="json file containing all info needed to run models")
    parser.add_argument('-dg', type=str, nargs="?", help="debug flag for gpu, pass true to be able to inspect tensor values", default="False")
    parser.add_argument('-dc', type=str, nargs="?", help="debug flag for cpu, pass true to be able to inspect tensor values", default="False")
    parser.add_argument('-s', type=str, nargs="?", help="seed number", default=None)
    args = parser.parse_args()

    os.environ["SLURM_JOB_NAME"] = "bash"  # server runs crash without this line (yay raytune)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # For debugging of cuda errors

    run_data, config = load_run_file(args.runfile)
    model_type = run_data["model_type"]
    server_model_dir = run_data["server_model_dir"]

    debug_flag = False
    if args.dg in ["true", "True"]:
        debug_flag = True
        run_data["gpus"] = 1

    if args.dc in ["true", "True"]:
        debug_flag = True
        run_data["gpus"] = 0

    if args.s is not None:
        config["seed"] = int(args.s)

    # Training Code
    if model_type == "rbm":
        model = RBM(config, debug=debug_flag, precision=config["precision"])
    elif model_type == "exp_rbm":
        model = ExpRBM(config, debug=debug_flag, precision=config["precision"])
    elif model_type == "crbm":
        model = CRBM(config, debug=debug_flag, precision=config["precision"])
    elif model_type == "exp_crbm":
        model = ExpCRBM(config, debug=debug_flag, precision=config["precision"])
    elif model_type == 'pcrbm':
        model = pCRBM(config, debug=debug_flag, precision=config["precision"])
    elif model_type == "net_crbm":
        model = CRBM_net(config, debug=debug_flag, precision=config["precision"])
    elif model_type == "pool_crbm":
        model = pool_CRBM(config, debug=debug_flag, precision=config["precision"])
    elif model_type == "pool_class_crbm":
        model = pool_class_CRBM(config, debug=debug_flag, precision=config["precision"])
    # elif model_type == "pool_regression_crbm":
    #     model = pool_regression_CRBM(config, debug=debug_flag, precision=config["precision"])
    elif model_type == "comp_crbm":
        model = comp_CRBM(config, debug=debug_flag, precision=config["precision"])
    else:
        print(f"Model Type {model_type} is not supported")
        exit(1)

    logger = TensorBoardLogger(server_model_dir, name=run_data["model_name"])

    if debug_flag:
        if run_data["gpus"] == 0:
            plt = Trainer(max_epochs=config['epochs'], logger=logger, accelerator="cpu")
        else:
            plt = Trainer(max_epochs=config['epochs'], logger=logger, accelerator="cuda", devices=run_data["gpus"])
    elif run_data["gpus"] > 1:
        # distributed data parallel, multi-gpus on single machine or across multiple machines
        plt = Trainer(max_epochs=config['epochs'], logger=logger, gpus=run_data["gpus"], accelerator="cuda", strategy="ddp")  # distributed data-parallel
    else:
        plt = Trainer(max_epochs=config['epochs'], logger=logger, devices=run_data["gpus"], accelerator="cuda")  # gpus=1,
    plt.fit(model)