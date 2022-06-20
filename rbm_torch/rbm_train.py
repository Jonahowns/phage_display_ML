from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
import argparse
import os
import sys
from rbm import RBM
import rbm_configs
sys.path.insert(1, './analysis/')
from global_info import get_global_info

if __name__ == '__main__':
    # Example Usage
    # rbm_train.py pig ../pig/b3_c1.fasta protein 200 1 False
    parser = argparse.ArgumentParser(description="RBM Training on Phage Display Dataset")
    parser.add_argument('datatype_str', type=str, help="Which Datset? pig, invivo, rod, or cov? Used to Set default config")
    parser.add_argument('dataset', type=str, help="Location of Data File")
    parser.add_argument('epochs', type=int, help="Number of Training Iterations")
    parser.add_argument('gpus', type=int, help="Number of gpus available")
    parser.add_argument('weights', type=str, help="Use sequence count to weight sequences")
    parser.add_argument('precision', type=str, help="single or double precision", default="double")
    args = parser.parse_args()

    os.environ["SLURM_JOB_NAME"] = "bash"  # server runs crash without this line (yay raytune)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # For debugging of cuda errors

    file = os.path.basename(args.dataset)
    name = file.split(".")[0]   # short specifier of round etc.

    clusternum = name[-1]  # cluster the data belongs to
    if clusternum.isalpha() or clusternum.isdigit() and name[-2] != "c":
        clusternum = 1
    elif clusternum.isdigit() and name[-2] == "c":  # Cluster specified
        clusternum = int(clusternum)
    else:  # Character is neither a letter nor number
        print(f"Cluster Designation {clusternum} is not supported.")
        exit(-1)

    weights, w_bool = None, False
    if args.weights in ["True", "true", "TRUE", "yes"]:
        weights = "fasta"  # All weights are already in the processed files
        name += "_w"
        w_bool = True

    model = "rbm"

    try:
        info = get_global_info(args.datatype_str, dir="../datasets/dataset_files/")
    except KeyError:
        print(f"Key {args.datatype_str} not found in get_global_info function in /analysis/analysis_methods.py")
        exit(-1)

    # Set Default config
    try:
        config = rbm_configs.all_configs[info["configkey"][str(clusternum)]]
    except KeyError:
        print(f"Configkey {info['configkey'][str(clusternum)]} Not Supported.")
        print("Please add default config to configs.py under this key. Please add global info about to /analysis/analysis_methods.py")
        exit(-1)

    # Edit config for dataset specific hyperparameters
    config["fasta_file"] = args.dataset
    config["sequence_weights"] = weights
    config["epochs"] = args.epochs

    # Training Code
    rbm = RBM(config, debug=False)
    logger = TensorBoardLogger('../' + info["server_model_dir"][model], name=name)
    if args.gpus > 1:
        # distributed data parallel, multi-gpus on single machine or across multiple machines
        plt = Trainer(max_epochs=config['epochs'], logger=logger, gpus=args.gpus, accelerator="ddp")  # distributed data-parallel
    else:
        plt = Trainer(max_epochs=config['epochs'], logger=logger, gpus=args.gpus)  # gpus=1,
    plt.fit(rbm)
