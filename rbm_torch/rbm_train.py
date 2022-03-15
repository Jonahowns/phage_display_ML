from rbm_test import RBM
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from numpy.random import randint
import os
from glob import glob



if __name__ == '__main__':

    # rbm_trian.py pig ../pig_tissue/b3_c1.fasta protein 22 50 200 2 1

    parser = argparse.ArgumentParser(description="RBM Training on Phage Display Dataset")
    parser.add_argument('focus', type=str, help="Which Datset? pig, invivo, or rod?")
    parser.add_argument('dataset', type=str, help="Location of Data File")
    parser.add_argument('molecule', type=str, help="Must be protein, dna or rna")
    parser.add_argument('visible', type=int, help="Number of Visible Units in RBM")
    parser.add_argument('hidden', type=int, help="Number of Hidden Units in RBM")
    parser.add_argument('epochs', type=int, help="Number of Training Iterations")
    parser.add_argument('gpus', type=int, help="Number of gpus available")
    parser.add_argument('weights', type=bool, help="Use Weights of sequences")
    args = parser.parse_args()

    if args.weights:
        weights = "fasta"
    else:
        weights = None

    molecule_states = {"dna": 5, "rna": 5, "protein": 21}
    log_dirs = {"pig": "pig_tissue", "invivo": "invivo", "rod":"rod"}

    config = {"fasta_file": args.dataset,
              "h_num": args.hidden,  # number of hidden units, can be variable
              "v_num": args.visible,
              "q": molecule_states[args.molecule],
              "molecule": args.molecule,
              "batch_size": 10000,
              "mc_moves": 8,
              "seed": randint(0, 10000, 1)[0],
              "lr": 0.006,
              "lr_final": None,
              "decay_after": 0.75,
              "loss_type": "free_energy",
              "sample_type": "gibbs",
              "sequence_weights": weights,
              "optimizer": "AdamW",
              "epochs": args.epochs,
              "weight_decay": 0.001,  # l2 norm on all parameters
              "l1_2": 0.25,
              "lf": 0.002,
              "data_worker_num": 6
              }

    file = os.path.basename(args.dataset)
    name = file.split(".")[0]

    # Weighted RBMS are put into separate tensorboard folders
    if args.weights:
        name += "_w"


    # Training Code
    rbm = RBM(config)
    logger = TensorBoardLogger('../' + log_dirs[args.focus] + "/trained_rbms", name=name)
    plt = pl.Trainer(max_epochs=config['epochs'], logger=logger, gpus=int(args.gpus))  # gpus=1,
    plt.fit(rbm)
