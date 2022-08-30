from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

import argparse
# import json
# import numpy as np
# from numpy.random import randint
import os

from rbm_torch.models.rbm import RBM
from rbm_torch.models.crbm import CRBM
from rbm_torch.models.crbm_experimental import ExpCRBM
from rbm_torch.models.crbm_net import CRBM_net
from rbm_torch.models.rbm_experimental import ExpRBM

from rbm_torch.utils.utils import load_run_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runs Training Procedure from a json run file")
    parser.add_argument('runfile', type=str, help="json file containing all info needed to run models")
    parser.add_argument('-d', type=str, nargs="?", help="debug flag, pass true to be able to inspect tensor values", default="False")
    args = parser.parse_args()

    os.environ["SLURM_JOB_NAME"] = "bash"  # server runs crash without this line (yay raytune)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # For debugging of cuda errors

    run_data, config = load_run_file(args.runfile)
    model_type = run_data["model_type"]
    server_model_dir = run_data["server_model_dir"]

    debug_flag = False
    if args.d in ["true", "True"]:
        debug_flag = True
        run_data["gpus"] = 0

    # Training Code
    if model_type == "rbm":
        model = RBM(config, debug=debug_flag, precision=config["precision"])
    elif model_type == "exp_rbm":
        model = ExpRBM(config, debug=debug_flag, precision=config["precision"])
    elif model_type == "crbm":
        model = CRBM(config, debug=debug_flag, precision=config["precision"])
    elif model_type == "exp_crbm":
        model = ExpCRBM(config, debug=debug_flag, precision=config["precision"])
    elif model_type == "net_crbm":
        model = CRBM_net(config, debug=debug_flag, precision=config["precision"])
    else:
        print(f"Model Type {model_type} is not supported")
        exit(1)

    logger = TensorBoardLogger(server_model_dir, name=run_data["model_name"])

    if debug_flag:
        plt = Trainer(max_epochs=config['epochs'], logger=logger, accelerator="cpu")
    elif run_data["gpus"] > 1:
        # distributed data parallel, multi-gpus on single machine or across multiple machines
        plt = Trainer(max_epochs=config['epochs'], logger=logger, gpus=run_data["gpus"], accelerator="cuda", strategy="ddp")  # distributed data-parallel
    else:
        plt = Trainer(max_epochs=config['epochs'], logger=logger, devices=run_data["gpus"], accelerator="cuda")  # gpus=1,
    plt.fit(model)