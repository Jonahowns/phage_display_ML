from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

import argparse
import os
from rbm_torch.models.crbm_base import CRBM
from rbm_torch.models.pool_crbm_base import pool_CRBM
from rbm_torch.models.pool_crbm_relu_base import pool_CRBM_relu
from rbm_torch.models.pool_crbm_classification import pool_class_CRBM
from rbm_torch.models.pool_crbm_cluster import pcrbm_cluster

from rbm_torch.utils.utils import load_run


def get_model(model_type, config, debug_flag=False):
    if model_type == "crbm":
        return CRBM(config, debug=debug_flag, precision=config["precision"])
    elif model_type == "pool_crbm":
        return pool_CRBM(config, debug=debug_flag, precision=config["precision"])
    elif model_type == "pool_crbm_relu":
        return pool_CRBM_relu(config, debug=debug_flag, precision=config["precision"])
    elif model_type == "pool_class_crbm":
        return pool_class_CRBM(config, debug=debug_flag, precision=config["precision"])
    elif model_type == "pcrbm_cluster":
        return pcrbm_cluster(config, debug=debug_flag, precision=config["precision"])
    else:
        print(f"Model Type {model_type} is not supported")
        exit(1)



# callable from jupyter notebooks etc.
def train(run_data_dict, debug_flag=False):
    logger = TensorBoardLogger(run_data_dict['server_model_dir'], name=run_data_dict["model_name"])
    config = run_data_dict['config']

    model = get_model(run_data_dict["model_type"], config, debug_flag=debug_flag)

    if run_data_dict["gpus"] > 1:
        # distributed data parallel, multi-gpus on single machine or across multiple machines
        plt = Trainer(max_epochs=config['epochs'], logger=logger, gpus=run_data_dict["gpus"], accelerator="cuda",
                      strategy="ddp")  # distributed data-parallel
    else:
        if run_data_dict['gpus'] == 0:
            plt = Trainer(max_epochs=config['epochs'], logger=logger, accelerator="cpu")
        else:
            plt = Trainer(max_epochs=config['epochs'], logger=logger, devices=run_data_dict["gpus"],
                          accelerator="cuda")  # gpus=1,
    plt.fit(model)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runs Training Procedure from a json run file")
    parser.add_argument('runfile', type=str, help="json file containing all info needed to run models")
    parser.add_argument('-dg', type=str, nargs="?", help="debug flag for gpu, pass true to be able to inspect tensor values", default="False")
    parser.add_argument('-dc', type=str, nargs="?", help="debug flag for cpu, pass true to be able to inspect tensor values", default="False")
    parser.add_argument('-s', type=str, nargs="?", help="seed number", default=None)
    args = parser.parse_args()

    os.environ["SLURM_JOB_NAME"] = "bash"  # server runs crash without this line (yay raytune)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # For debugging of cuda errors

    run_data, config = load_run(args.runfile)

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
    model = get_model(run_data["model_type"], config, debug_flag=debug_flag)

    logger = TensorBoardLogger(run_data["server_model_dir"], name=run_data["model_name"])

    if run_data["gpus"] > 1:
        # distributed data parallel, multi-gpus on single machine or across multiple machines
        plt = Trainer(max_epochs=config['epochs'], logger=logger, gpus=run_data["gpus"], accelerator="cuda", strategy="ddp")  # distributed data-parallel
    else:
        if run_data['gpus'] == 0:
            plt = Trainer(max_epochs=config['epochs'], logger=logger, accelerator="cpu")
        else:
            plt = Trainer(max_epochs=config['epochs'], logger=logger, devices=run_data["gpus"], accelerator="cuda")  # gpus=1,
    plt.fit(model)