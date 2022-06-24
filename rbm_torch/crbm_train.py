from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
import argparse
import os
import sys
import json
import numpy as np
# Local imports
from crbm import CRBM
import crbm_configs
sys.path.insert(1, './analysis/')
from global_info import get_global_info

if __name__ == '__main__':
    # Example Usage
    # rbm_train.py pig ../pig/b3_c1.fasta protein 200 1 False
    parser = argparse.ArgumentParser(description="CRBM Training on Phage Display Dataset")
    parser.add_argument('datatype_str', type=str, help="Which Datset? pig, invivo, rod, or cov? Used to Set default config")
    parser.add_argument('dataset', type=str, help="Location of Data File")
    parser.add_argument('epochs', type=int, help="Number of Training Iterations")
    parser.add_argument('gpus', type=int, help="Number of gpus available")
    parser.add_argument('weights', type=str, help="Must provide Weight File or be 'fasta' to use weight in fasta file")
    parser.add_argument('precision', type=str, help="single or double precision", default="double")
    args = parser.parse_args()

    os.environ["SLURM_JOB_NAME"] = "bash"  # server runs crash without this line (yay raytune)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # For debugging of cuda errors

    file = os.path.basename(args.dataset)
    name = file.split(".")[0]   # short specifier of round etc.

    clusternum = name[-1]  # cluster the data belongs to ()
    if clusternum.isalpha() or clusternum.isdigit() and name[-2] != "c":  # No cluster number, set to 1
        clusternum = 1
    elif clusternum.isdigit() and name[-2] == "c": # Cluster specified, get clusternum
        clusternum = int(clusternum)
    else:  # Character is neither a letter nor number
        print(f"Cluster Designation {clusternum} is not supported.")
        exit(-1)

    weights = None
    if args.weights == "fasta" or args.weights in ["True", "TRUE", "yes"]:
        weights = "fasta"  # All weights are already in the processed fasta files
        name += "_f"
    elif args.weights == "" or args.weights is None or args.weights == "None":
        pass
    else:
        ## Assumes weight file to be in same directory as our data files.
        try:
            with open(os.path.dirname(args.dataset)+"/" + args.weights) as f:
                data = json.load(f)
            weights = np.asarray(data["weights"])
            name += f"_{data['extension']}"
        except IOError:
            print(f"Could not load provided weight file {os.path.dirname(args.dataset)+'/'+args.weights}")
            exit(-1)

    model = "crbm"

    try:
        info = get_global_info(args.datatype_str, dir="../datasets/dataset_files/")
    except KeyError:
        print(f"Key {args.datatype_str} not found in get_global_info function in /analysis/analysis_methods.py")
        exit(-1)

    # Set Default config
    try:
        config = crbm_configs.all_configs[info["configkey"][str(clusternum)]]
    except KeyError:
        print(f"Configkey {info['configkey'][str(clusternum)]} Not Supported.")
        print("Please add default config to configs.py under this key. Please add global info about to /analysis/analysis_methods.py")
        exit(-1)

    # Edit config for dataset specific hyperparameters
    config["fasta_file"] = args.dataset
    config["sequence_weights"] = weights
    config["epochs"] = args.epochs

    # Training Code
    crbm = CRBM(config, debug=False)
    logger = TensorBoardLogger('../' + info["server_model_dir"][model], name=name)
    if args.gpus > 1:
        # distributed data parallel, multi-gpus on single machine or across multiple machines
        plt = Trainer(max_epochs=config['epochs'], logger=logger, gpus=args.gpus, accelerator="ddp")  # distributed data-parallel
    else:
        plt = Trainer(max_epochs=config['epochs'], logger=logger, gpus=args.gpus)  # gpus=1,
    plt.fit(crbm)
