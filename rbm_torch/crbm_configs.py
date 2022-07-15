from numpy.random import randint

seed = randint(0, 10000, 1)[0]

pig_c1_2_default_config = {"fasta_file": "",
          "v_num": 22,
          "q": 21,
          "molecule": "protein",
          "epochs": 200,
          "seed": seed,
          "batch_size": 4000,
          "mc_moves": 6,
          "lr": 0.006,
          "lr_final": None,
          "decay_after": 0.75,
          "loss_type": "free_energy",
          "sample_type": "gibbs",
          "sequence_weights": None,
          "optimizer": "AdamW",
          "weight_decay": 0.001,  # l2 norm on all parameters
          "l1_2": 15,
          "lf": 10,
          "ld": 1,
          "data_worker_num": 4,
          }

pig_c2_2_default_config = {"fasta_file": "",
          "v_num": 45,
          "q": 21,
          "molecule": "protein",
          "epochs": 200,
          "seed": seed,
          "batch_size": 4000,
          "mc_moves": 6,
          "lr": 0.006,
          "lr_final": None,
          "decay_after": 0.75,
          "loss_type": "free_energy",
          "sample_type": "gibbs",
          "sequence_weights": None,
          "optimizer": "AdamW",
          "weight_decay": 0.001,  # l2 norm on all parameters
          "l1_2": 20,
          "lf": 5,
          "ld": 10,
          "data_worker_num": 4
          }

pig_c1_4_default_config = {"fasta_file": "",
          "v_num": 16,
          "q": 21,
          "molecule": "protein",
          "epochs": 200,
          "seed": seed,
          "batch_size": 10000,
          "mc_moves": 6,
          "lr": 0.006,
          "lr_final": None,
          "decay_after": 0.75,
          "loss_type": "free_energy",
          "sample_type": "gibbs",
          "sequence_weights": None,
          "optimizer": "AdamW",
          "weight_decay": 0.001,  # l2 norm on all parameters
          "l1_2": 20,
          "lf": 5,
          "ld": 10,
          "data_worker_num": 4
          }

pig_c2_4_default_config = {"fasta_file": "",
          "v_num": 22,
          "q": 21,
          "molecule": "protein",
          "epochs": 200,
          "seed": seed,
          "batch_size": 10000,
          "mc_moves": 6,
          "lr": 0.006,
          "lr_final": None,
          "decay_after": 0.75,
          "loss_type": "free_energy",
          "sample_type": "gibbs",
          "sequence_weights": None,
          "optimizer": "AdamW",
          "weight_decay": 0.001,  # l2 norm on all parameters
          "l1_2": 20,
          "lf": 5,
          "ld": 10,
          "data_worker_num": 4
          }

pig_c3_4_default_config = {"fasta_file": "",
          "v_num": 39,
          "q": 21,
          "molecule": "protein",
          "epochs": 200,
          "seed": seed,
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
          "l1_2": 20,
          "lf": 5,
          "ld": 10,
          "data_worker_num": 4
          }

pig_c4_4_default_config = {"fasta_file": "",
          "h_num": 20,  # number of hidden units, can be variable
          "v_num": 45,
          "q": 21,
          "molecule": "protein",
          "epochs": 200,
          "seed": seed,
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
          "l1_2": 20,
          "lf": 5,
          "ld": 10,
          "data_worker_num": 4
          }

# pig_c1_2_default_config["convolution_topology"] = {"hidden1": {"number": 5, "kernel": (11, pig_c1_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
#                                                   "hidden2": {"number": 5, "kernel": (7, pig_c1_2_default_config["q"]), "stride": (3, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
#                                                   "hidden3": {"number": 5, "kernel": (7, pig_c1_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
#                                                   "hidden4": {"number": 5, "kernel": (pig_c1_2_default_config["v_num"], pig_c1_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
#                                                  }
pig_c1_2_default_config["convolution_topology"] = {"hidden3": {"number": 15, "kernel": (3, pig_c1_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
                                                  "hidden7": {"number": 15, "kernel": (7, pig_c1_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
                                                  "hidden9": {"number": 15, "kernel": (9, pig_c1_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
                                                  # "hidden4": {"number": 5, "kernel": (pig_c1_2_default_config["v_num"], pig_c1_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                                 }

# pig_c2_2_default_config["convolution_topology"] = {"hidden1": {"number": 5, "kernel": (15, pig_c2_2_default_config["q"]), "stride": (10, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
#                                                   "hidden2": {"number": 5, "kernel": (8, pig_c2_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (2, 1), "output_padding": (0, 0)},
#                                                   "hidden3": {"number": 5, "kernel": (5, pig_c2_2_default_config["q"]), "stride": (5, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
#                                                   "hidden4": {"number": 5, "kernel": (pig_c2_2_default_config["v_num"], pig_c2_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
#                                                  }
pig_c2_2_default_config["convolution_topology"] = {"hidden1": {"number": 20, "kernel": (11, pig_c2_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
                                                  "hidden2": {"number": 30, "kernel": (19, pig_c2_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
                                                  "hidden3": {"number": 30, "kernel": (32, pig_c2_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
                                                  # "hidden4": {"number": 5, "kernel": (pig_c2_2_default_config["v_num"], pig_c2_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                                 }


pig_c1_4_default_config["convolution_topology"] = {"hidden1": {"number": 20, "kernel": (5, pig_c1_4_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
                                                  "hidden2": {"number": 30, "kernel": (9, pig_c1_4_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
                                                  "hidden3": {"number": 30, "kernel": (13, pig_c1_4_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
                                                  # "hidden4": {"number": 5, "kernel": (pig_c1_4_default_config["v_num"], pig_c1_4_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
                                                 }

pig_c2_4_default_config["convolution_topology"] = {"hidden1": {"number": 20, "kernel": (9, pig_c1_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
                                                  "hidden2": {"number": 30, "kernel": (13, pig_c1_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
                                                  "hidden3": {"number": 30, "kernel": (17, pig_c1_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
                                                  # "hidden4": {"number": 5, "kernel": (pig_c1_2_default_config["v_num"], pig_c1_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                                 }


pig_c3_4_default_config["convolution_topology"] = {"hidden1": {"number": 20, "kernel": (11, pig_c3_4_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
                                                  "hidden2": {"number": 30, "kernel": (18, pig_c3_4_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
                                                  "hidden3": {"number": 30, "kernel": (25, pig_c3_4_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
                                                  # "hidden4": {"number": 5, "kernel": (pig_c3_4_default_config["v_num"], pig_c3_4_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                                 }


pig_c4_4_default_config["convolution_topology"] = {"hidden1": {"number": 20, "kernel": (9, pig_c2_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
                                                  "hidden2": {"number": 30, "kernel": (18, pig_c2_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (2, 1), "output_padding": (0, 0), "weight": 1},
                                                  "hidden3": {"number": 30, "kernel": (29, pig_c2_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
                                                  # "hidden4": {"number": 5, "kernel": (pig_c2_2_default_config["v_num"], pig_c2_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
                                                 }

cov_default_config = {"fasta_file": "",
          "v_num": 40,
          "q": 5,
          "molecule": "dna",
          "epochs": 100,
          "seed": seed,
          "batch_size": 10000,
          "mc_moves": 6,
          "lr": 0.006,
          "lr_final": None,
          "decay_after": 0.75,
          "loss_type": "free_energy",
          "sample_type": "gibbs",
          "sequence_weights": None,
          "optimizer": "AdamW",
          "weight_decay": 0.001,  # l2 norm on all parameters
          "l1_2": 20,
          "lf": 5,
          "ld": 10,
          "data_worker_num": 4
          }

# cov_default_config["convolution_topology"] = {"hidden1": {"number": 5, "kernel": (10, cov_default_config["q"]), "stride": (5, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
#                                             "hidden2": {"number": 5, "kernel": (7, cov_default_config["q"]), "stride": (3, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
#                                             "hidden3": {"number": 5, "kernel": (5, cov_default_config["q"]), "stride": (3, 1), "padding": (0, 0), "dilation": (3, 1), "output_padding": (0, 0)},
#                                             "hidden4": {"number": 5, "kernel": (cov_default_config["v_num"], cov_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
#                                            }

cov_default_config["convolution_topology"] = {
                                            "hidden1": {"number": 10, "kernel": (9, cov_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                            "hidden2": {"number": 10, "kernel": (15, cov_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                            "hidden3": {"number": 20, "kernel": (20, cov_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                            "hidden4": {"number": 20, "kernel": (27, cov_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                            "hidden5": {"number": 20, "kernel": (35, cov_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                             }

cov_sw_default_config = {"fasta_file": "",
          "v_num": 40,
          "q": 5,
          "molecule": "dna",
          "epochs": 100,
          "seed": seed,
          "batch_size": 2000,
          "mc_moves": 6,
          "lr": 0.006,
          "lr_final": None,
          "decay_after": 0.75,
          "loss_type": "free_energy",
          "sample_type": "gibbs",
          "sequence_weights": None,
          "optimizer": "AdamW",
          "weight_decay": 0.001,  # l2 norm on all parameters
          "l1_2": 20,
          "lf": 5,
          "ld": 10,
          "data_worker_num": 4,
          }


cov_sw_default_config["convolution_topology"] = {
                                            "hidden1": {"number": 10, "kernel": (9, cov_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                            "hidden2": {"number": 10, "kernel": (15, cov_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                            "hidden3": {"number": 20, "kernel": (20, cov_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                            "hidden4": {"number": 20, "kernel": (27, cov_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                            "hidden5": {"number": 20, "kernel": (35, cov_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                             }



ribo_default_config = {"fasta_file": "",
          "v_num": 121,
          "q": 5,
          "molecule": "rna",
          "epochs": 100,
          "seed": seed,
          "batch_size": 10000,
          "mc_moves": 4,
          "lr": 0.006,
          "lr_final": None,
          "decay_after": 0.75,
          "loss_type": "free_energy",
          "sample_type": "gibbs",
          "sequence_weights": None,
          "optimizer": "AdamW",
          "weight_decay": 0.001,  # l2 norm on all parameters
          "l1_2": 25.0,
          "lf": 5.0,
          "ld": 10.0,
          "data_worker_num": 4
          }

ribo_default_config["convolution_topology"] = {"hidden10": {"number": 10, "kernel": (11, ribo_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                            "hidden25": {"number": 10, "kernel": (25, ribo_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                            "hidden46": {"number": 10, "kernel": (46, ribo_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                            "hidden86": {"number": 20, "kernel": (86, ribo_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                              "hidden100": {"number": 20, "kernel": (100, ribo_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                                "hidden112": {"number": 20, "kernel": (112, ribo_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                             }

thc_default_config = {"fasta_file": "",
          "v_num": 43,
          "q": 5,
          "molecule": "rna",
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
          "l1_2": 25.0,
          "lf": 5.0,
          "ld": 10.0,
          "data_worker_num": 4
          }

thc_default_config["convolution_topology"] = {"hidden10": {"number": 15, "kernel": (9, thc_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                            "hidden25": {"number": 15, "kernel": (17, thc_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                            "hidden46": {"number": 15, "kernel": (25, thc_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                            "hidden86": {"number": 15, "kernel": (33, thc_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                             }

exo_default_config = {"fasta_file": "",
          "v_num": 38,
          "q": 5,
          "molecule": "dna",
          "epochs": 100, # get's overwritten by training script anyway
          "seed": seed, # this is defined in the config file
          "batch_size": 400, # can be raised or lowered depending on memory usage
          "mc_moves": 4,
          "lr": 0.006,
          "lr_final": None, # automatically set as lr * 1e-2
          "decay_after": 0.75,
          "loss_type": "free_energy",
          "sample_type": "gibbs",
          "sequence_weights": None,
          "optimizer": "AdamW",
          "weight_decay": 0.001,  # l2 norm on all parameters
          "l1_2": 1.0,  # 0.5 for weighted with no denominator, 20.0 for weighted or not with the denominator
          "lf": 5.0,
          "ld": 1.5,  # 0.5 for weighted
          "data_worker_num": 4
          }

exo_default_config["convolution_topology"] = {"hidden7": {"number": 20, "kernel": (8, exo_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                            "hidden13": {"number": 20, "kernel": (13, exo_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                            "hidden19": {"number": 20, "kernel": (19, exo_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                            "hidden25": {"number": 15, "kernel": (25, exo_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0},
                                            "hidden29": {"number": 10, "kernel": (29, exo_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1.0}
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
   "cov_sw": cov_sw_default_config,
   "cov_nw": cov_sw_default_config,
   "ribo": ribo_default_config,
   "thc": thc_default_config,
   "exo": exo_default_config,
   "exo_sw": exo_default_config
}


yu_aptamer_default_config = {"fasta_file": "",
          "v_num": 40,
          "q": 4,
          "molecule": "dna",
          "epochs": 200,
          "seed": seed,
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
          "l1_2": 20,
          "lf": 5,
          "ld": 10,
          "data_worker_num": 4
          }

yu_aptamer_default_config["convolution_topology"] = {"hidden1": {"number": 10, "kernel": (18, cov_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
                                            "hidden2": {"number": 15, "kernel": (13, cov_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
                                            "hidden3": {"number": 15, "kernel": (9, cov_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
                                            # "hidden4": {"number": 5, "kernel": (cov_default_config["v_num"], cov_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                           }


lattice_default_config = {"fasta_file": "",
              "molecule": "protein",
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
              "l1_2": 20,
              "lf": 5,
              "ld": 10,
              }

lattice_default_config["convolution_topology"] = {
        "hidden1": {"number": 10, "kernel": (9, lattice_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
        "hidden2": {"number": 15, "kernel": (13, lattice_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
        "hidden3": {"number": 15, "kernel": (19, lattice_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0), "weight": 1},
        # "hidden4": {"number": 5, "kernel": (lattice_default_config["v_num"], lattice_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
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