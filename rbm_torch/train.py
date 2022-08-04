from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

import argparse
import json
import numpy as np
from numpy.random import randint

from models.rbm import RBM
from models.crbm import CRBM
from models.crbm_experimental import ExpCRBM
from models.rbm_experimental import ExpRBM


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training from json run file")
    parser.add_argument('runfile', type=str, help="json file containing all info needed to run models")
    args = parser.parse_args()

    with open(args.runfile, "r") as f:
        run_data = json.load(f)

    # Get info needed for all models
    model_type = run_data["model_type"]  # rbm, crbm, exp_rbm, exp_crbm
    assert model_type in ["rbm", "crbm", "exp_rbm", "exp_crbm"]

    config = run_data["config"]
    model_name = run_data["model_name"]

    data_dir = run_data["data_dir"]
    fasta_file = run_data["fasta_file"]

    server_model_dir = run_data["server_model_dir"]

    # Deal with weights
    weights = None
    if run_data["weights"] == "fasta":
        weights = "fasta"  # All weights are already in the processed fasta files
        # model_name += "_f"
    elif run_data["weights"] is None or run_data["weights"] == "None":
        pass
    else:
        ## Assumes weight file to be in same directory as our data files.
        try:
            with open(data_dir + run_data["weights"]) as f:
                data = json.load(f)
            weights = np.asarray(data["weights"])
            # model_name += f"_{data['extension']}"
        except IOError:
            print(f"Could not load provided weight file {data_dir + run_data['weights']}")
            exit(-1)

    # Edit config for dataset specific hyperparameters
    config["fasta_file"] = data_dir + fasta_file
    config["sequence_weights"] = weights
    seed = randint(0, 10000, 1)[0]
    config["seed"] = seed
    if config["lr_final"] == "None":
        config["lr_final"] = None

    if "crbm" in model_type:
        # added since json files don't support tuples
        for key, val in config["convolution_topology"].items():
            for attribute in ["kernel", "dilation", "padding", "stride", "output_padding"]:
                val[f"{attribute}"] = (val[f"{attribute}x"], val[f"{attribute}y"])


    # Training Code
    if model_type == "rbm":
        model = RBM(config, debug=False, precision=config["precision"])
    elif model_type == "exp_rbm":
        model = ExpRBM(config, debug=False, precision=config["precision"])
    elif model_type == "crbm":
        model = CRBM(config, debug=False, precision=config["precision"])
    elif model_type == "exp_crbm":
        model = ExpCRBM(config, debug=False, precision=config["precision"])

    logger = TensorBoardLogger('../' + server_model_dir, name=model_name)
    if run_data["gpus"] > 1:
        # distributed data parallel, multi-gpus on single machine or across multiple machines
        plt = Trainer(max_epochs=config['epochs'], logger=logger, gpus=run_data["gpus"], accelerator="ddp")  # distributed data-parallel
    else:
        plt = Trainer(max_epochs=config['epochs'], logger=logger, gpus=run_data["gpus"])  # gpus=1,
    plt.fit(model)