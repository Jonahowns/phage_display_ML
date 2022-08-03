import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

# Project Dependencies
from utils.utils import BatchNorm1D, all_weights

from models.rbm import RBM


class ExpRBM(RBM):
    """ Pytorch Lightning Module of Restricted Boltzmann Machine

    Parameters
    ----------
    config: dictionary,
        sets mandatory variables for model to run
    precision: str, optional, default="double"
        sets default dtype of torch tensors as torch.float32 or torch.float64
    debug: bool, optional, default=False
        sets dataworker number to 1, enables use of debugger during model training
        if False, tensor information won't be visible in debugger

    Notes
    -----
    Description of Mandatory Config Keys and their accepted values
        "fasta_file": str or list of strs,
        "h_num": int, number of hidden nodes
        "v_num": int, number of visible nodes, should be equal to sequence length of data
        "q": int, number of values each visible node can take on, ex. for dna with values {A, C, G, T, -}, q is 5
        "mc_moves": int, number of times each layer is sampled during each step
        "batch_size": int, number of sequences in a batch
        "epochs": int, number of iterations to run the model
        "molecule": str, type of sequence data can be {"dna", "rna", "protein"}
        "sample_type": str, {"gibbs", "pt", "pcd"}
        "loss_type": str, {"free_energy", "energy"}
        "optimizer": str, {"AdamW", "SGD", "Adagrad"}
        "lr": float, learning rate of model
        "l1_2": float, l1^2 penalty on the sum of the absolute value of the weights
        "lf": float, penalty on the sum of the absolute value of the visible biases 'fields'
        "seed": int, controls tensor generating random values for reproducable behavior



    Description of Optional Config Keys and their accepted values
        "data_worker_num": int, default=multiprocessing.cpu_count()
        "sequence_weights": str, torch.tensor, or np.array
            str values: "fasta", use values in fasta file
                        other str, name of a weight file in same directory as provided fasta file

        "lr_final"
    """


    def __init__(self, config, precision="double", debug=False):
        super().__init__(config, precision=precision, debug=debug)
        self.batch_norm = BatchNorm1D(affine=True, momentum=0.1)

    ## Compute Input for Hidden Layer from Visible Potts
    def compute_output_v(self, v):
        """ Calculate input into hidden layer from visible layer
        Parameters
        ----------
        v: torch.tensor
           a visible unit configuration, usually of shape (batch_size, v_num) with values 0 to (self.q-1)
        beta: flaot, optional, default=1
           inverse temperature factor, used in annealed importance sampling

        Returns
        -------
        output: torch.tensor
          returns hidden unit input as shape (batch_size, h_num)

        Notes
        -----
        Calculates and returns:
        .. math:: I_\mu = \sum_{i} w_{i \mu} v_i
        """
        # compute_output of visible potts layer
        vd = v.long()

        # Newest and fastest version, does take a lot of memory though...
        indexTensor = vd.unsqueeze(1).unsqueeze(-1).expand(-1, self.h_num, -1, -1)
        expandedweights = self.W.unsqueeze(0).expand(vd.shape[0], -1, -1, -1)
        output = torch.gather(expandedweights, 3, indexTensor).squeeze(3).sum(2)
        return self.batch_norm(output)
        # return output



# if __name__ == '__main__':
    # data_file = '../invivo/sham2_ipsi_c1.fasta'  # cpu is faster
    # large_data_file = '../invivo/chronic1_spleen_c1.fasta' # gpu is faster
    # lattice_data = '../datasets/lattice_proteins_verification/Lattice_Proteins_MSA.fasta'
    # # b3_c1 = "../pig/b3_c1.fasta"
    # # bivalent_data = "./bivalent_aptamers_verification/s100_8th.fasta"
    #
    # config = rbm_configs.lattice_default_config
    # # Edit config for dataset specific hyperparameters
    # config["fasta_file"] = lattice_data
    # config["sequence_weights"] = None
    # config["epochs"] = 100
    # config["l12"] = 30
    # config["seed"] = 38
    #
    # # Training Code
    # rbm = ExpRBM(config, debug=False)
    # logger = TensorBoardLogger('../tb_logs/', name="lattice_rbm")
    # plt = Trainer(max_epochs=config['epochs'], logger=logger, gpus=1)  # gpus=1,
    # plt.fit(rbm)
    #
    # checkp = "./tb_logs/lattice_rbm/version_12/checkpoints/epoch=99-step=199.ckpt"
    # rbm = ExpRBM.load_from_checkpoint(checkp)
    #
    # all_weights(rbm, name="./tb_logs/lattice_rbm/version_12/affine_batch_norm")

    #
    # # results = gen_data_lowT(rbm, which="marginal")
    # results = gen_data_zeroT(rbm, which="joint")
    # visible, hiddens = results
    #
    # E = rbm.energy(visible, hiddens)
    # print("E", E.shape)





    # import analysis.analysis_methods as am
    # Directory of Stored RBMs
    # mdir = "/mnt/D1/globus/exo_trained_rbms/"
    # rounds = ["exosome_st"]
    # data = ["exosome"]
    #
    # checkp, v_dir = am.get_checkpoint_path(rounds[0], rbmdir=mdir)
    # exo_rbm = RBM.load_from_checkpoint(checkp)
    #
    # exo_rbm.AIS()