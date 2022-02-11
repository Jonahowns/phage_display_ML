from rbm_test import RBM
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from numpy.random import randint
import os



if __name__ == '__main__':

    # rbm_trian.py pig ../pig_tissue/b3_c1.fasta protein 22 50 200 2 1

    parser = argparse.ArgumentParser(description="RBM Training on Phage Display Dataset")
    parser.add_argument('focus', type=str, help="Which Datset? pig, invivo, or rod?")
    parser.add_argument('dataset', type=str, help="Location of Data File")
    parser.add_argument('molecule', type=str, help="Must be protein, dna or rna")
    parser.add_argument('visible', type=str, help="Number of Visible Units in RBM")
    parser.add_argument('hidden', type=str, help="Number of Hidden Units in RBM")
    parser.add_argument('epochs', type=str, help="Number of Training Iterations")
    parser.add_argument('gpus', type=str, help="Number of gpus available")
    args = parser.parse_args()

    molecule_states = {"dna": 5, "rna": 5, "protein": 21}
    log_dirs = {"pig": "pig_tissue", "invivo": "invivo", "rod":"rod"}

    config = {"fasta_file": args.dataset,
              "h_num": int(args.hidden),  # number of hidden units, can be variable
              "v_num": int(args.visible),
              "q": molecule_states[args.molecule],
              "batch_size": 10000,
              "mc_moves": 6,
              "seed": randint(0, 10000, 1)[0],
              "lr": 0.01,
              "lr_final": None,
              "decay_after": 0.75,
              "loss_type": "free_energy",
              "sample_type": "gibbs",
              "sequence_weights": None,
              "optimizer": "AdamW",
              "epochs": int(args.epochs),
              "weight_decay": 0.05,  # l2 norm on all parameters
              "l1_2": 0.25,
              "lf": 0.002,
              "raytune": False  # Only for hyperparameter optimization
              }

    file = os.path.basename(args.dataset)
    name = file.split(".")[0]

    # Training Code
    rbm = RBM(config)
    logger = TensorBoardLogger('../' + log_dirs[args.focus] + "/trained_rbms", name=name)
    plt = pl.Trainer(max_epochs=config['epochs'], logger=logger, gpus=int(args.gpus))  # gpus=1,
    plt.fit(rbm)
