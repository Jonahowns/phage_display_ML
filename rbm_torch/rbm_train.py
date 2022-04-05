from rbm import RBM
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from numpy.random import randint
import os
import configs
from rbm_torch.analysis.global_info import get_global_info
# from glob import glob


if __name__ == '__main__':
    # Example Usage
    # rbm_train.py pig ../pig_tissue/b3_c1.fasta protein 200 1 False
    parser = argparse.ArgumentParser(description="RBM Training on Phage Display Dataset")
    parser.add_argument('datatype_str', type=str, help="Which Datset? pig, invivo, rod, or cov? Used to Set default config")
    parser.add_argument('dataset', type=str, help="Location of Data File")
    parser.add_argument('epochs', type=int, help="Number of Training Iterations")
    parser.add_argument('gpus', type=int, help="Number of gpus available")
    parser.add_argument('weights', type=str, help="Use sequence count to weight sequences")
    args = parser.parse_args()

    os.environ["SLURM_JOB_NAME"] = "bash"  # server runs crash without this line (yay raytune)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # For debugging of cuda errors

    file = os.path.basename(args.dataset)
    name = file.split(".")[0]   # short specifier of round etc.
    clusternum = int(name[-1])  # cluster the data belongs to ()

    weights, w_bool = None, False
    if args.weights in ["True", "true", "TRUE", "yes"]:
        weights = "fasta"  # All weights are already in the processed files
        name += "_w"
        w_bool = True

    try:
        info = get_global_info(args.datatype_str, cluster=clusternum, weights=w_bool)
    except KeyError:
        print(f"Key {args.datatype_str} not found in get_global_info function in /analysis/analysis_methods.py")
        exit(-1)

    # Set Default config
    try:
        config = configs.all_configs[info["configkey"]]
    except KeyError:
        print(f"Configkey {info['configkey']} Not Supported.")
        print("Please add default config to configs.py under this key. Please add global info about to /analysis/analysis_methods.py")
        exit(-1)

    # Edit config for dataset specific hyperparameters
    config["fasta_file"] = args.dataset
    config["sequence_weights"] = weights
    config["epochs"] = args.epochs

    # Training Code
    rbm = RBM(config, debug=False)
    logger = TensorBoardLogger('../' + info["server_rbm_dir"] + "/trained_rbms", name=name)
    plt = pl.Trainer(max_epochs=config['epochs'], logger=logger, gpus=args.gpus)  # gpus=1,
    plt.fit(rbm)
