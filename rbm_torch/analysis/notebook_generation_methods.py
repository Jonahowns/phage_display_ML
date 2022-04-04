
imports = [
""" \
import sys
sys.path.append("../")

from rbm import fasta_read, get_beta_and_W, all_weights, RBM
import analysis_methods as am

from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import subprocess as sp
import matplotlib.image as mpimg
"""
]







log_dirs = {"pig_ge2": "./pig_tissue/gaps_end_2_clusters",
            "pig_gm2": "./pig_tissue/gaps_middle_2_clusters",
            "pig_gm4": "./pig_tissue/gaps_middle_4_clusters",
            "pig_ge4": "./pig_tissue/gaps_end_4_clusters",
            "invivo": "./invivo", "rod": "./rod", "cov": "./cov"}




cells