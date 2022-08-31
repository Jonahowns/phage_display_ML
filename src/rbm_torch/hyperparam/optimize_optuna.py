from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import os

# local files
from rbm_torch.models.rbm import RBM
from rbm_torch.models.crbm import CRBM
from rbm_torch.models.crbm_experimental import ExpCRBM, pCRBM
from rbm_torch.models.crbm_net import CRBM_net
from rbm_torch.models.rbm_experimental import ExpRBM


import optuna
from optuna.integration import PyTorchLightningPruningCallback

models = {
    "pcrbm": pCRBM,
    "exp_crbm": ExpCRBM,
    "crbm": CRBM,
    "rbm": RBM,
    "exp_rbm": ExpRBM,
    "net_crbm": CRBM_net
}

metrics = {
    "pcrbm": "ptl/val_pearson_corr",
    "exp_crbm": "ptl/val_free_energy",
    "crbm": "ptl/val_free_energy",
    "rbm": "ptl/val_pseudo_likelihood",
    "exp_rbm": "ptl/val_pseudo_likelihood",
    "net_crbm": "ptl/val_fitness_mse"
}

# maximize of minimize the corresponding metric
directions = {
    "pcrbm": "maximize",
    "exp_crbm": "minimize",
    "crbm": "minimize",
    "rbm": "minimize",
    "exp_rbm": "minimize",
    "net_crbm": "maximize"
}





def objective(trial, hyperparams_of_interest, config, epochs, postfix=None):
    hyper_params = {}
    for key, value in hyperparams_of_interest.items():
        assert key in config.keys()
        assert key not in ["sequence_weights", "seed", "q", "v_num", "raytune", "fasta_file",
                           "molecule"]  # these you can't really change for now
        # This dictionary contains type of hyperparameter it is and the parameters associated with each type
        for subkey, subval in value.items():
            if subkey == "uniform" :
                config[key] = trial.suggest_uniform(key, subval[0], subval[1])
                # config[key] = tune.uniform(subval[0], subval[1])
            elif subkey == "loguniform":
                config[key] = trial.suggest_loguniform(key, subval[0], subval[1])
                # hyper_param_mut[key] = lambda: np.exp(np.random.uniform(subval[0], subval[1]))
            elif subkey == "choice":
                config[key] = trial.suggest_categorical(key, subval)
                # hyper_param_mut[key] = subval
            elif subkey == "grid":
                config[key] = trial.suggest_categorical(key, subval)
                # hyper_param_mut[key] = subval
            hyper_params[key] = config[key]

    model = config["model_type"]
    assert model in models.keys()

    metric = metrics[model]

    if postfix:
        config["model_name"] = config["model_name"] + f"_{postfix}"

    mod = models[model](config, precision=config["precision"])

    # num_gpus = config["gpus"]
    # device_num = [device]
    acc = "cuda"

    trainer = Trainer(
        # default_root_dir="./checkpoints/",
        max_epochs=epochs,
        devices=1,
        accelerator=acc,
        enable_progress_bar=False,
        enable_checkpointing=True,
        logger=TensorBoardLogger(
            save_dir=os.path.join(os.getcwd(), config["server_model_dir"]), name=config["model_name"], version=trial.number),
        callbacks=[PyTorchLightningPruningCallback(trial, monitor=metric)],
    )

    # hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims)
    # trainer.logger.log_hyperparams(hyper_params)
    trainer.fit(mod)

    return trainer.callback_metrics[metric].item()


# Simple implementation for multi gpu optimization
# adapted from https://github.com/optuna/optuna/issues/1365
class Objective:
    def __init__(self, gpu_queue, hyperparams_of_interest, config, epochs, postfix=None):
        # Shared queue to manage GPU IDs.
        self.gpu_queue = gpu_queue
        self.hyperparams_of_interest = hyperparams_of_interest
        self.config = config
        self.epochs = epochs
        self.postfix = postfix

    def __call__(self, trial):
        # Fetch GPU ID for this trial.
        gpu_id = self.gpu_queue.get()

        # Please write actual objective function here
        value = objective(trial, self.hyperparams_of_interest, self.config, self.epochs, postfix=self.postfix)

        # Return GPU ID to the queue.
        self.gpu_queue.put(gpu_id)

        # return metric
        return value
