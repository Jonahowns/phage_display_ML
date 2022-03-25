from rbm import RBM
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from numpy.random import randint
import os
import configs
# from glob import glob


if __name__ == '__main__':
    # Example Usage
    # rbm_train.py pig ../pig_tissue/b3_c1.fasta protein 200 1 False
    parser = argparse.ArgumentParser(description="RBM Training on Phage Display Dataset")
    parser.add_argument('focus', type=str, help="Which Datset? pig, invivo, rod, or cov? Used to Set default config")
    parser.add_argument('dataset', type=str, help="Location of Data File")
    parser.add_argument('epochs', type=int, help="Number of Training Iterations")
    parser.add_argument('gpus', type=int, help="Number of gpus available")
    parser.add_argument('weights', type=bool, help="Use sequence count to weight sequences")
    args = parser.parse_args()

    weights = None
    if args.weights is True:
        weights = "fasta"  # All weights are already in the processed files

    log_dirs = {"pig": "pig_tissue", "invivo": "invivo", "rod":"rod", "cov":"cov"}

    file = os.path.basename(args.dataset)
    name = file.split(".")[0]   # short specifier of round etc.

    # Weighted RBMS are put into separate tensorboard folders
    if weights is not None:
        name += "_w"

    # Set Default config
    if args.focus == "pig":
        if "c1" in name:
            config = configs.pig_c1_default_config
        elif "c2" in name:
            config = configs.pig_c2_default_config
    elif args.focus == "cov":
        config = configs.cov_default_config
    else:
        print("Focus Not Supported. Please Add focus to this script and add default config to configs.py")
        exit(-1)

    # Edit config for dataset specific hyperparameters
    config["fasta_file"] = args.dataset
    config["sequence_weights"] = weights
    config["epochs"] = args.epochs

    # Training Code
    rbm = RBM(config)
    logger = TensorBoardLogger('../' + log_dirs[args.focus] + "/trained_rbms", name=name)
    plt = pl.Trainer(max_epochs=config['epochs'], logger=logger, gpus=args.gpus)  # gpus=1,
    plt.fit(rbm)
