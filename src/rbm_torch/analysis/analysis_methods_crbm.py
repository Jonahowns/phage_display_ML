from rbm_torch.utils import utils
from rbm_torch.analysis import analysis_methods as am

import numpy as np

import torch
import matplotlib.pyplot as plt


def dataframe_to_input(dataframe, base_to_id, v_num, q, weights=False):
    seqs = dataframe["sequence"].tolist()
    oh_ten = torch.zeros((len(seqs), v_num, q), dtype=torch.long)
    for iid, seq in enumerate(seqs):
        for n, base in enumerate(seq):
            oh_ten[iid, n, base_to_id[base]] = 1
    if weights:
        weights = dataframe["copy_num"].tolist()
        return oh_ten, weights
    else:
        return oh_ten

# Inputs across the k dimension (separate convolutions, same convolution weight)
# hidden key is name assigned to hidden layer in convolution topology
def data_with_weights_plot(crbm, dataframe, hidden_key, hidden_unit_numbers, kdim="mean", data="cgf", data_range=None):
    if data not in ["cgf", "mean"]:
        print(f"Data Type {data} not supported!")
        exit(-1)

    if hidden_key not in crbm.hidden_convolution_keys:
        print(f"Hidden Convolution Key {hidden_key} not found!")
        exit(-1)

    # Convert Sequences to one hot encoding Format and Compute Hidden Unit Input
    base_to_id = am.int_to_letter_dicts[crbm.molecule]
    data_tensor, weights = dataframe_to_input(dataframe, base_to_id, crbm.v_num, crbm.q, weights=True)
    input_hiddens = crbm.compute_output_v(data_tensor) # List of Iuk matrices

    weight_index = crbm.hidden_convolution_keys.index(hidden_key)  # Get index of Weight for accessing input_hiddens list
    input_W_hiddens = input_hiddens[weight_index] # Get the hidden inputs for this particular weight

    Wdims = crbm.convolution_topology[hidden_key]["weight_dims"]  # Get dimensions of W matrix
    h_num = Wdims[0]
    beta, W = utils.get_beta_and_W(crbm, hidden_key, include_gaps=False)   # Get Beta and sort hidden Units by Frobenius Norms
    order = np.argsort(beta)[::-1]

    ### Reduction of the k dimension, or alternatively view all
    if kdim in ["mean", "sum"]:
        if kdim == "mean":
            input_W_hiddens = input_W_hiddens.mean(2)
        if kdim == "sum":
            input_W_hiddens = input_W_hiddens.sum(2)
        gs_kw = dict(width_ratios=[3, 1], height_ratios=[1 for x in hidden_unit_numbers])
        grid_names = [[f"weight{i}", f"cgf{i}"] for i in range(len(hidden_unit_numbers))]
        # Make Figure
        fig, axd = plt.subplot_mosaic(grid_names, gridspec_kw=gs_kw, figsize=(10, 5*len(hidden_unit_numbers)), constrained_layout=True)
    elif kdim == "full":
        convx = input_W_hiddens.shape[2]
        subcol_num = convx // 2
        if convx - (subcol_num*2) != 0:
            even = False
            # Uneven column sizes, add extra plot which we won't use
            grid_names = [[f"weight{i}", [[f"cgf{i}_{j}" for j in range(0, convx//2 + 1)], [f"cgf{i}_{j}" for j in range(convx//2 + 1, convx + 1)]]] for i in range(len(hidden_unit_numbers))]
        else:
            even = True
            grid_names = [[f"weight{i}", [[f"cgf{i}_{j}" for j in range(0, convx//2)], [f"cgf{i}_{j}" for j in range(convx//2, convx)]]] for i in range(len(hidden_unit_numbers))]
        # Make Figure
        fig, axd = plt.subplot_mosaic(grid_names, figsize=(10, 5*len(hidden_unit_numbers)), constrained_layout=True)
    else:
        print(f"Kdim argument must be mean, sum, or full. {kdim} Not Supported")
        exit(-1)

    # Prepare Line Data to be plot, must be either cgf or mean
    if data == "cgf":
        npoints = 1000  # Number of points for graphing CGF curve
        lims = [(np.sum(np.min(w, axis=1)), np.sum(np.max(w, axis=1))) for w in W]  # Get limits for each hidden unit
        data_range = torch.zeros((npoints, h_num))
        # change this to data_range?
        for i in range(h_num):
            x = lims[i]
            data_range[:, i] = torch.tensor(np.linspace(x[0], x[1], num=npoints).transpose())

        pre_cgf = crbm.cgf_from_inputs_h(data_range, hidden_key)
        line_data = pre_cgf.detach().numpy()
        range_data = data_range.detach().numpy()

    elif data == "mean":
        I = input_W_hiddens
        I_min, inds = I.min(dim=0)
        I_max, inds = I.max(dim=0)
        I_min.unsqueeze_(0)
        I_max.unsqueeze_(0)
        if I.dim() == 3:
            data_range = (I_max-I_min) * torch.arange(0,1+0.01,0.01).unsqueeze(1).unsqueeze(2) + I_min
        elif I.dim() == 2:
            data_range = (I_max-I_min) * torch.arange(0,1+0.01,0.01).unsqueeze(1) + I_min

        mean = crbm.mean_h(data_range, hidden_key=hidden_key)
        line_data = mean.detach().numpy()
        range_data = data_range.detach().numpy()

    # Convewrt to Numpy for Graphing
    input_W_hiddens = input_W_hiddens.detach().numpy()

    for hid, hu_num in enumerate(hidden_unit_numbers):
        ix = order[hu_num]  # get hidden units by frobenius norm order (look at get_beta_and_W)
        # Make Sequence Logo
        utils.Sequence_logo(W[ix], ax=axd[f"weight{hid}"], data_type="weights", ylabel=f"Weight #{hu_num}", ticks_every=5, ticks_labels_size=14, title_size=20, molecule=crbm.molecule)

        if kdim != "full":
            t_x = np.asarray(range_data[:, ix])
            t_y = np.asarray(line_data[:, ix])
            deltay = np.min(t_y)
            counts, bins = np.histogram(input_W_hiddens[:, ix], bins=100, weights=weights)
            factor = np.max(t_y) / np.max(counts)
            # WEIGHTS SHOULD HAVE SAME SIZE AS BINS
            axd[f"cgf{hid}"].hist(bins[:-1], bins, color='grey', label='All sequences', weights=counts*factor,
                       histtype='step', lw=3, fill=True, alpha=0.7, edgecolor='black', linewidth=1)
            axd[f"cgf{hid}"].plot(t_x, t_y - deltay, lw=3, color='C1')
            axd[f"cgf{hid}"].tick_params(axis='both', direction='in', length=6, width=2, colors='k')
            axd[f"cgf{hid}"].tick_params(axis='both', labelsize=16)
            axd[f"cgf{hid}"].yaxis.tick_right()
            axd[f"cgf{hid}"].yaxis.set_label_position("right")

            if data == "cgf":
                axd[f"cgf{hid}"].set_ylabel('CGF', fontsize=18)

        else:
            t_x = np.asarray(range_data[:, ix])
            t_y = np.asarray(line_data[:, ix])

            if data == "cgf":
                axd[f"cgf{hid}_{0}"].set_ylabel('CGF', fontsize=10)
                deltay = np.min(t_y)
            elif data == "mean":
                deltay = np.min(t_y, axis=1)

            axd[f"cgf{hid}_{0}"].yaxis.set_label_position("left")
            axd[f"cgf{hid}_{0}"].tick_params(axis='both', direction='in', length=6, width=2, colors='k')

            for j in range(convx):
                counts, bins = np.histogram(input_W_hiddens[:, ix, j], bins=100, weights=weights)
                factor = np.max(t_y) / np.max(counts)
                axd[f"cgf{hid}_{j}"].hist(bins[:-1], bins, color='grey', label='All sequences', weights=counts*factor,
                           histtype='step', lw=3, fill=True, alpha=0.7, edgecolor='black', linewidth=1)

                if data == "mean":
                    axd[f"cgf{hid}_{j}"].plot(t_x[:, j], t_y[:, j] - deltay[j], lw=3, color='C1')
                else:
                    axd[f"cgf{hid}_{j}"].plot(t_x, t_y - deltay, lw=3, color='C1')

                axd[f"cgf{hid}_{j}"].tick_params(axis='both', labelsize=8)
                axd[f"cgf{hid}_{j}"].yaxis.tick_right()

                if not even and j == convx - 1:  # Last Plot that contains nothing
                    utils.clean_ax(axd[f"cgf{hid}_{j+1}"])
    plt.show()


# Produces flat vector of Inputs of each hidden unit (reduced over the convolution dimension (k) by sum or mean)
def flatten_and_reduce_input(Ih, reduction="sum"):
     # Iuk (Batch, hidden number (u), conv number (k))
    if reduction == "sum":
        return torch.cat([Iuk.sum(2) for Iuk in Ih], 1)
    elif reduction == "mean":
        return torch.cat([Iuk.mean(2) for Iuk in Ih], 1)
    else:
        print(f"Reduction {reduction} not supported")
        exit(-1)


def prepare_input_vector(crbm, dataframe):
    base_to_id = am.int_to_letter_dicts[crbm.molecule]
    data_tensor, weights = dataframe_to_input(dataframe, base_to_id, crbm.v_num, crbm.q, weights=True)
    input_hiddens = crbm.compute_output_v(data_tensor) # List of Iuk matrices
    return flatten_and_reduce_input(input_hiddens).detach().numpy()
