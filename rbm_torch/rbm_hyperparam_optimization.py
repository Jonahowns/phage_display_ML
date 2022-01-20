import sklearn
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.suggest.bayesopt import BayesOptSearch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import os
import numpy as np

from rbm_test import RBM

class CustomStopper(tune.Stopper):
    def __init__(self):
        self.should_stop = False

    def __call__(self, trial_id, result):
        max_iter = 1000
        if not self.should_stop and result["psuedolikelihood"] > 0.96:
            self.should_stop = True
        return self.should_stop or result["training_iteration"] >= max_iter

    def stop_all(self):
        return self.should_stop


def pbt_rbm(fasta_file, num_samples=10, num_epochs=10, gpus_per_trial=0, cpus_per_trial=1):

    config = {"fasta_file": fasta_file,
              "h_num": 100,  # number of hidden units, can be variable
              "v_num": 27,
              "q": 21,
              "batch_size": 10000,
              "mc_moves": 6,
              "seed": 38,
              "lr": tune.uniform(1e-5, 1e-1),
              "lr_final": None,  # defaults to lr * 1e-2
              "decay_after": 0.75,  # Number of epochs to decay after
              "loss_type": "free_energy",
              "sample_type": "gibbs",
              "sequence_weights": None,
              "optimizer": "AdamW",
              "epochs": num_epochs,
              "weight_decay": tune.uniform(1e-5, 1e-1),
              "l1_2": tune.uniform(0.15, 0.6),
              "lf": tune.uniform(1e-5, 1e-2),
              "raytune": True}

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=10,
        hyperparam_mutations={
            # distribution for resampling
            "lr": lambda: np.random.uniform(1e-5, 0.1),
            "l1_2": lambda: np.random.uniform(0.15, 0.6),
            "lf": lambda: np.random.uniform(1e-5, 1e-2),
            "weight_decay": lambda: np.random.uniform(1e-5, 1e-1),
            # allow perturbations within this set of categorical values
            # "momentum": [0.8, 0.9, 0.99],
        })

    reporter = CLIReporter(
        parameter_columns=["lr", "l1_2", "lf", "weight_decay"],
        metric_columns=["train_loss", "psuedolikelihood", "training_iteration"])

    stopper = CustomStopper()

    analysis = tune.run(
        tune.with_parameters(
            train_rbm,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial
        },
        metric="psuedolikelihood",
        mode="max",
        local_dir="./ray_results/",
        config=config,
        num_samples=num_samples,
        name="tune_pbt_rbm",
        scheduler=scheduler,
        progress_reporter=reporter,
        verbose=1,
        stop=stopper,
        # export_formats=[ExportFormat.MODEL],
        checkpoint_score_attr="psuedolikelihood",
        keep_checkpoints_num=4)

    print("Best hyperparameters found were: ", analysis.get_best_config(metric="psuedolikelihood", mode="max"))



def train_rbm(config, checkpoint_dir=None, num_epochs=10, num_gpus=0):
    trainer = pl.Trainer(
        # default_root_dir="./checkpoints/",
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=num_gpus,
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        # callbacks=[
        #     TuneReportCheckpointCallback(
        #         metrics={
        #             "train_loss": "train_loss",
        #             "psuedolikelihood": "psuedolikelihood",
        #         },
        #         filename="checkpoint",
        #         on="train_end")
        # ]
    )

    if checkpoint_dir:
        # Currently, this leads to errors:
        # model = LightningMNISTClassifier.load_from_checkpoint(
        #     os.path.join(checkpoint, "checkpoint"))
        # Workaround:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        rbm = RBM.load_from_checkpoint(checkpoint)
    else:
        rbm = RBM(config)

    trainer.fit(rbm)
    
    
    
if __name__ == '__main__':
    # local Test
    # pbt_rbm("/home/jonah/PycharmProjects/ML_for_aptamers/rbm_torch/lattice_proteins_verification/Lattice_Proteins_MSA.fasta", 10, 100, 1, 1)
    # Server Run
    os.environ["SLURM_JOB_NAME"] = "bash"
    pbt_rbm("/scratch/jprocyk/machine_learning/ML_for_aptamers/rbm_torch/lattice_proteins_verification/Lattice_Proteins_MSA.fasta", 10, 100, 1, 2)
