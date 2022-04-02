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
    parser.add_argument('weights', type=str, help="Use sequence count to weight sequences")
    args = parser.parse_args()

    os.environ["SLURM_JOB_NAME"] = "bash"  # server runs crash without this line (yay raytune)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # For debugging of cuda errors

    weights = None
    if args.weights in ["True", "true", "TRUE", "yes"]:
        weights = "fasta"  # All weights are already in the processed files

    log_dirs = {"pig_ge2": "pig_tissue/gaps_end_2_clusters",
                "pig_gm2": "pig_tissue/gaps_middle_2_clusters",
                "pig_gm4": "pig_tissue/gaps_middle_4_clusters",
                "invivo": "invivo", "rod":"rod", "cov":"cov"}

    file = os.path.basename(args.dataset)
    name = file.split(".")[0]   # short specifier of round etc.

    clusternum = int(name[-1])

    # Weighted RBMS are put into separate tensorboard folders
    if weights is not None:
        name += "_w"

    # Set Default config
    if "pig" in args.focus:
        process_id = args.focus[-3:]
        configkey = "pig_" + f"c{clusternum}_" + process_id
        config = configs.pig_configs[configkey]
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
    rbm = RBM(config, debug=False)
    logger = TensorBoardLogger('../' + log_dirs[args.focus] + "/trained_rbms", name=name)
    plt = pl.Trainer(max_epochs=config['epochs'], logger=logger, gpus=args.gpus)  # gpus=1,
    plt.fit(rbm)
