from numpy.random import randint

seed = randint(0, 10000, 1)[0]

pig_c1_2_default_config = {"fasta_file": "",
          "v_num": 22,
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
          "ld": 20,
          "data_worker_num": 4
          }

pig_c2_2_default_config = {"fasta_file": "",
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
          "l1_2": 0.9,
          "lf": 0.002,
          "ld": 20.0,
          "data_worker_num": 4
          }

pig_c1_4_default_config = {"fasta_file": "",
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
          "l1_2": 0.9,
          "lf": 0.002,
          "ld": 20.0,
          "data_worker_num": 4
          }

pig_c2_4_default_config = {"fasta_file": "",
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
          "l1_2": 0.9,
          "lf": 0.002,
          "ld": 20.0,
          "data_worker_num": 4
          }

pig_c3_4_default_config = {"fasta_file": "",
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
          "ld": 20.0,
          "data_worker_num": 4
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
          "l1_2": 0.9,
          "lf": 0.002,
          "ld": 20.0,
          "data_worker_num": 4
          }

pig_c1_2_default_config["convolution_topology"] = {"hidden1": {"number": 5, "kernel": (11, pig_c1_2_default_config["q"]), "stride": (11, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                                  "hidden2": {"number": 5, "kernel": (7, pig_c1_2_default_config["q"]), "stride": (5, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                                  "hidden3": {"number": 5, "kernel": (4, pig_c1_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (2, 1), "output_padding": (0, 0)},
                                                  "hidden4": {"number": 5, "kernel": (pig_c1_2_default_config["v_num"], pig_c1_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                                 }

pig_c2_2_default_config["convolution_topology"] = {"hidden1": {"number": 5, "kernel": (15, pig_c2_2_default_config["q"]), "stride": (10, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                                  "hidden2": {"number": 5, "kernel": (8, pig_c2_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (2, 1), "output_padding": (0, 0)},
                                                  "hidden3": {"number": 5, "kernel": (5, pig_c2_2_default_config["q"]), "stride": (5, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                                  "hidden4": {"number": 5, "kernel": (pig_c2_2_default_config["v_num"], pig_c2_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                                 }


pig_c1_4_default_config["convolution_topology"] = {"hidden1": {"number": 5, "kernel": (8, pig_c1_4_default_config["q"]), "stride": (4, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                                  "hidden2": {"number": 5, "kernel": (6, pig_c1_4_default_config["q"]), "stride": (5, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                                  "hidden3": {"number": 5, "kernel": (4, pig_c1_4_default_config["q"]), "stride": (4, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                                  "hidden4": {"number": 5, "kernel": (pig_c1_4_default_config["v_num"], pig_c1_4_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                                 }

pig_c2_4_default_config["convolution_topology"] = {"hidden1": {"number": 5, "kernel": (11, pig_c1_2_default_config["q"]), "stride": (11, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                                  "hidden2": {"number": 5, "kernel": (7, pig_c1_2_default_config["q"]), "stride": (5, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                                  "hidden3": {"number": 5, "kernel": (4, pig_c1_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (2, 1), "output_padding": (0, 0)},
                                                  "hidden4": {"number": 5, "kernel": (pig_c1_2_default_config["v_num"], pig_c1_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                                 }


pig_c3_4_default_config["convolution_topology"] = {"hidden1": {"number": 5, "kernel": (13, pig_c3_4_default_config["q"]), "stride": (2, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                                  "hidden2": {"number": 5, "kernel": (8, pig_c3_4_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (2, 1), "output_padding": (0, 0)},
                                                  "hidden3": {"number": 5, "kernel": (5, pig_c3_4_default_config["q"]), "stride": (2, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                                  "hidden4": {"number": 5, "kernel": (pig_c3_4_default_config["v_num"], pig_c3_4_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                                 }


pig_c4_4_default_config["convolution_topology"] = {"hidden1": {"number": 5, "kernel": (15, pig_c2_2_default_config["q"]), "stride": (10, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                                  "hidden2": {"number": 5, "kernel": (8, pig_c2_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (2, 1), "output_padding": (0, 0)},
                                                  "hidden3": {"number": 5, "kernel": (5, pig_c2_2_default_config["q"]), "stride": (5, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                                  "hidden4": {"number": 5, "kernel": (pig_c2_2_default_config["v_num"], pig_c2_2_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                                 }

cov_default_config = {"fasta_file": "",
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
          "l1_2": 0.9,
          "lf": 0.004,
          "ld": 20.0,
          "data_worker_num": 4
          }

cov_default_config["convolution_topology"] = {"hidden1": {"number": 5, "kernel": (10, cov_default_config["q"]), "stride": (5, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                            "hidden2": {"number": 5, "kernel": (7, cov_default_config["q"]), "stride": (3, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                            "hidden3": {"number": 5, "kernel": (5, cov_default_config["q"]), "stride": (3, 1), "padding": (0, 0), "dilation": (3, 1), "output_padding": (0, 0)},
                                            "hidden4": {"number": 5, "kernel": (cov_default_config["v_num"], cov_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
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
   "cov": cov_default_config
}


yu_aptamer_default_config = {"fasta_file": "",
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
          "l1_2": 0.9,
          "lf": 0.004,
          "ld": 20.0,
          "data_worker_num": 4
          }

yu_aptamer_default_config["convolution_topology"] = {"hidden1": {"number": 5, "kernel": (10, cov_default_config["q"]), "stride": (5, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                            "hidden2": {"number": 5, "kernel": (7, cov_default_config["q"]), "stride": (3, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
                                            "hidden3": {"number": 5, "kernel": (5, cov_default_config["q"]), "stride": (3, 1), "padding": (0, 0), "dilation": (3, 1), "output_padding": (0, 0)},
                                            "hidden4": {"number": 5, "kernel": (cov_default_config["v_num"], cov_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
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
              "l1_2": 0.185,
              "lf": 0.002,
              "ld": 0.8,
              "data_worker_num": 4
              }

lattice_default_config["convolution_topology"] = {
        "hidden1": {"number": 5, "kernel": (9, lattice_default_config["q"]), "stride": (3, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
        "hidden2": {"number": 5, "kernel": (7, lattice_default_config["q"]), "stride": (5, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
        "hidden3": {"number": 5, "kernel": (3, lattice_default_config["q"]), "stride": (2, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
        "hidden4": {"number": 5, "kernel": (lattice_default_config["v_num"], lattice_default_config["q"]), "stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "output_padding": (0, 0)},
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