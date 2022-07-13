# phage_display_ML
Unsupervised Machine Learning for Aptamer Datasets. The overall goal is to identify sequence characteristics responsible for the binding of the aptamer sequences to their respective target. Analysis of the sequence data is performed with two machine learning models: a Restricted Boltzmann Machine and a Convolutional Restricted Boltzmann Machine, both implemented in Pytorch Lightning. Both make use of a categorical representation of the sequence as the visible layer and a continuous valued hidden layer with an applied dReLU activation function. The RBM implementation is a modified version of the RBM from [this repository](https://github.com/jertubiana/PGM).

Both models can be trained on cpu, gpu, or in a distributed setting.


# Repository Structure

### datasets directory

#### dataset_files directory
This directory contains json files for each dataset that specify the neccesary data for training a model: the fasta files belonging to the dataset, the directory of the dataset files (relative to the training script), the locations of the trained models, the type of biomolecule, and the string that specifies the corresponding set of hyperparameters in either rbm_torch/rbm_configs.py or rbm_torch/crbm_configs.py. Automatic generation of these files can be performed with function generate_dataset_file in data_prep.py in parent datasets directory.

Please note some of the information in these files is specific to my particular setup. Some variables in rbm_torch/globabl_info.py need to be changed to work with your particular setup.

#### all other directories
These directories store the aptamer data as specially formatted fasta files with the copy number of each sequence stored in the header of each sequence.

ex. >seq1-10

indicates this sequence has a copy number of 10.

Currently, these directories only contain placeholder files and none of the actual fasta files due to their size. Additionally notebooks analyzing the datasets and trained model's performance will be placed into an analysis subdirectory in the corresponding folder.

#### various notebooks

Dataset specific notebooks that demonstrate how to use the functions in dataset_prep.py to prepare a dataset

### envs

#### exmachina.yml 
The primary environment for running all models and analyses. It does include a cuda version of pytorch as well as several other packages so it can take awhile to install.

To install an conda env from this file you can run the command:

conda env create -n ENVNAME --file exmachina.yml

### rbm_torch directory
