from numpy.random import randint

seed = randint(0, 10000, 1)[0]

pig_c1_2_default_config = {"fasta_file": "",
          "h_num": 40,  # number of hidden units, can be variable
          "v_num": 22,
          "q": 21,
          "molecule": "protein",
          "epochs": 200,
          "seed": seed,
          "data_worker_num": 6,
          "batch_size": 60000,
          "mc_moves": 8,
          "lr": 0.006,
          "lr_final": None,
          "decay_after": 0.75,
          "loss_type": "free_energy",
          "sample_type": "gibbs",
          "sequence_weights": None,
          "optimizer": "AdamW",
          "weight_decay": 0.001,  # l2 norm on all parameters
          "l1_2": 0.25,
          "lf": 0.002,
          }

pig_c2_2_default_config = {"fasta_file": "",
          "h_num": 20,  # number of hidden units, can be variable
          "v_num": 45,
          "q": 21,
          "molecule": "protein",
          "epochs": 200,
          "seed": seed,
          "data_worker_num": 6,
          "batch_size": 10000,
          "mc_moves": 8,
          "lr": 0.006,
          "lr_final": None,
          "decay_after": 0.75,
          "loss_type": "free_energy",
          "sample_type": "gibbs",
          "sequence_weights": None,
          "optimizer": "AdamW",
          "weight_decay": 0.001,  # l2 norm on all parameters
          "l1_2": 0.25,
          "lf": 0.002,
          }

pig_c1_4_default_config = {"fasta_file": "",
          "h_num": 20,  # number of hidden units, can be variable
          "v_num": 16,
          "q": 21,
          "molecule": "protein",
          "epochs": 200,
          "seed": seed,
          "data_worker_num": 4,
          "batch_size": 10000,
          "mc_moves": 8,
          "lr": 0.006,
          "lr_final": None,
          "decay_after": 0.75,
          "loss_type": "free_energy",
          "sample_type": "gibbs",
          "sequence_weights": None,
          "optimizer": "AdamW",
          "weight_decay": 0.001,  # l2 norm on all parameters
          "l1_2": 0.25,
          "lf": 0.002,
          }

pig_c2_4_default_config = {"fasta_file": "",
          "h_num": 20,  # number of hidden units, can be variable
          "v_num": 22,
          "q": 21,
          "molecule": "protein",
          "epochs": 200,
          "seed": seed,
          "data_worker_num": 4,
          "batch_size": 10000,
          "mc_moves": 8,
          "lr": 0.006,
          "lr_final": None,
          "decay_after": 0.75,
          "loss_type": "free_energy",
          "sample_type": "gibbs",
          "sequence_weights": None,
          "optimizer": "AdamW",
          "weight_decay": 0.001,  # l2 norm on all parameters
          "l1_2": 0.32,
          "lf": 0.002,
          }

pig_c3_4_default_config = {"fasta_file": "",
          "h_num": 20,  # number of hidden units, can be variable
          "v_num": 39,
          "q": 21,
          "molecule": "protein",
          "epochs": 200,
          "seed": seed,
          "data_worker_num": 4,
          "batch_size": 10000,
          "mc_moves": 8,
          "lr": 0.006,
          "lr_final": None,
          "decay_after": 0.75,
          "loss_type": "free_energy",
          "sample_type": "gibbs",
          "sequence_weights": None,
          "optimizer": "AdamW",
          "weight_decay": 0.001,  # l2 norm on all parameters
          "l1_2": 0.38,
          "lf": 0.002,
          }

pig_c4_4_default_config = {"fasta_file": "",
          "h_num": 20,  # number of hidden units, can be variable
          "v_num": 45,
          "q": 21,
          "molecule": "protein",
          "epochs": 200,
          "seed": seed,
          "data_worker_num": 4,
          "batch_size": 10000,
          "mc_moves": 8,
          "lr": 0.006,
          "lr_final": None,
          "decay_after": 0.75,
          "loss_type": "free_energy",
          "sample_type": "gibbs",
          "sequence_weights": None,
          "optimizer": "AdamW",
          "weight_decay": 0.001,  # l2 norm on all parameters
          "l1_2": 0.45,
          "lf": 0.002,
          }


cov_default_config = {"fasta_file": "",
          "h_num": 30,  # number of hidden units, can be variable
          "v_num": 40,
          "q": 5,
          "molecule": "dna",
          "epochs": 100,
          "seed": seed,
          "data_worker_num": 6,
          "batch_size": 25000,
          "mc_moves": 6,
          "lr": 0.006,
          "lr_final": None,
          "decay_after": 0.75,
          "loss_type": "free_energy",
          "sample_type": "gibbs",
          "sequence_weights": None,
          "optimizer": "AdamW",
          "weight_decay": 0.001,  # l2 norm on all parameters
          "l1_2": 0.475,
          "lf": 0.004,
          }

exo_default_config = {"fasta_file": "",
          "v_num": 38,
          "h_num": 50,
          "q": 5,
          "molecule": "dna",
          "epochs": 100, # get's overwritten by training script anyway
          "seed": seed, # this is defined in the config file
          "batch_size": 500, # can be raised or lowered depending on memory usage
          "mc_moves": 2,
          "lr": 0.006,
          "lr_final": None, # automatically set as lr * 1e-2
          "decay_after": 0.75,
          "loss_type": "free_energy",
          "sample_type": "gibbs",
          "sequence_weights": None,
          "optimizer": "AdamW",
          "weight_decay": 0.001,  # l2 norm on all parameters
          "l1_2": 15.0,
          "lf": 0.004,
          "data_worker_num": 4
          }

exo_sw_default_config = {"fasta_file": "",
          "v_num": 38,
          "h_num": 50,
          "q": 5,
          "molecule": "dna",
          "epochs": 100, # get's overwritten by training script anyway
          "seed": seed, # this is defined in the config file
          "batch_size": 10000, # can be raised or lowered depending on memory usage
          "mc_moves": 4,
          "lr": 0.006,
          "lr_final": None, # automatically set as lr * 1e-2
          "decay_after": 0.75,
          "loss_type": "free_energy",
          "sample_type": "gibbs",
          "sequence_weights": None,
          "optimizer": "AdamW",
          "weight_decay": 0.001,  # l2 norm on all parameters
          "l1_2": 0.35,
          "lf": 0.005,
          "data_worker_num": 4
          }


all_configs = {
   "pig_c1_ge2": pig_c1_2_default_config,
   "pig_c2_ge2": pig_c2_2_default_config,
   "pig_c1_gm2": pig_c1_2_default_config,
   "pig_c2_gm2": pig_c2_2_default_config,
   "pig_c1_gm4": pig_c1_4_default_config,
   "pig_c2_gm4": pig_c2_4_default_config,
   "pig_c3_gm4": pig_c3_4_default_config,
   "pig_c4_gm4": pig_c4_4_default_config,
   "pig_c1_ge4": pig_c1_4_default_config,
   "pig_c2_ge4": pig_c2_4_default_config,
   "pig_c3_ge4": pig_c3_4_default_config,
   "pig_c4_ge4": pig_c4_4_default_config,
   "cov": cov_default_config,
    "cov_nw": cov_default_config,
    "cov_sw": cov_default_config,
   "exo": exo_default_config,
    "exo_sw": exo_sw_default_config,
}




yu_aptamer_default_config = {"fasta_file": "",
          "h_num": 20,  # number of hidden units, can be variable
          "v_num": 40,
          "q": 4,
          "molecule": "dna",
          "epochs": 200,
          "seed": seed,
          "data_worker_num": 6,
          "batch_size": 15000,
          "mc_moves": 6,
          "lr": 0.006,
          "lr_final": None,
          "decay_after": 0.75,
          "loss_type": "free_energy",
          "sample_type": "gibbs",
          "sequence_weights": None,
          "optimizer": "AdamW",
          "weight_decay": 0.001,  # l2 norm on all parameters
          "l1_2": 0.3,
          "lf": 0.004,
          }

lattice_default_config = {"fasta_file": "",
              "molecule": "protein",
              "h_num": 20,  # number of hidden units, can be variable
              "v_num": 27,
              "q": 20,
              "batch_size": 20000,
              "mc_moves": 6,
              "seed": 38,
              "lr": 0.0065,
              "lr_final": None,
              "decay_after": 0.75,
              "loss_type": "free_energy",
              "sample_type": "gibbs",
              "sequence_weights": None,
              "optimizer": "AdamW",
              "epochs": 0,
              "weight_decay": 0.001,  # l2 norm on all parameters
              "l1_2": 0.185,
              "lf": 0.002,
              "data_worker_num": 6
              }




# pig_configs = {"pig_c1_ge2": pig_c1_2_default_config,
#                "pig_c2_ge2": pig_c2_2_default_config,
#                "pig_c1_gm2": pig_c1_2_default_config,
#                "pig_c2_gm2": pig_c2_2_default_config,
#                "pig_c1_gm4": pig_c1_4_default_config,
#                "pig_c2_gm4": pig_c2_4_default_config,
#                "pig_c3_gm4": pig_c3_4_default_config,
#                "pig_c4_gm4": pig_c4_4_default_config,
#                "pig_c1_ge4": pig_c1_4_default_config,
#                "pig_c2_ge4": pig_c2_4_default_config,
#                "pig_c3_ge4": pig_c3_4_default_config,
#                "pig_c4_ge4": pig_c4_4_default_config,
#              }