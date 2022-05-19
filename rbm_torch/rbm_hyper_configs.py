######## Define Hyperparameters Here ########
# All hyperparameters that can be optimized via population based Tuning or ASHA grid search

### If using Grid
# Number of Trials = Number of Samples * Number of Grid Parameter Combinations
### If using Random Choice Only
# Number of Trials = Number of Samples

### All Options that are currently supported
all_hyperparams_options = {
    "h_num": {"grid": [10, 120, 250, 1000]},  # number of hidden units, can be variable
    "batch_size": {"choice": [5000, 10000, 20000]},
    "mc_moves": {"grid": [4, 8]},
    "lr": {"uniform": [1e-5, 1e-1]},
    "decay_after": {"grid": [0.5, 0.75, 0.9]},  # Fraction of epochs to have exponential decay after
    "loss_type": {"choice": ["free_energy", "energy"]},
    "sample_type": {"choice": ["gibbs", "pt"]},
    "optimizer": {"choice": ["AdamW", "SGD", "Adagrad"]},
    "epochs": {"choice": [100, 200, 1000]},
    "weight_decay": {"uniform": [1e-5, 1e-1]},
    "l1_2": {"uniform": [0.15, 0.6]},
    "lf": {"uniform": [1e-5, 1e-2]},
}

# Some of the more pertinent hyperparameters with direct effect on weights, loss, and gradient
reg_opt = {
    "weight_decay": {"uniform": [1e-5, 1e-1]},
    "l1_2": {"uniform": [0.15, 0.6]},
    "lf": {"uniform": [5e-4, 1e-2]},
    "lr": {"uniform": [1e-5, 5e-2]}
}

# hidden unit number grid search with varying of regularization variables
hidden_opt = {
    "h_num": {"grid": [20, 60, 100, 200]},
    # "h_num": {"grid": [5, 10, 15, 20]},
    # "batch_size": {"choice": [10000, 20000]},
    "l1_2": {"uniform": [0.15, 0.6]},
    "lr": {"choice": [1e-3, 6e-3]},
    "lf": {"uniform": [1e-4, 1e-2]}
    # "mc_moves": {"choice": [4, 8]},
}

