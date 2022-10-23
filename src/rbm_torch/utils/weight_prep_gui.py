import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# VARS CONSTS:
# Upgraded dataSize to global...
_VARS = {'window': False,
         'fig_agg': False,
         'fig_sample': False,
         'fig_mw': False,
         'pltFig': False,
         'pltSample': False,
         'dataSize': 60}

plt.style.use('Solarize_Light2')


# Helper Functions
def remove_gaps_in_dist(dist_vals, bin_size=0.2):
    """returns distribution without gaps in it"""
    min_v = np.min(dist_vals)
    max_v = np.max(dist_vals)

    ng_dist = np.copy(dist_vals)

    current_val = min_v
    gt = ng_dist >= current_val

    while current_val < max_v and np.count_nonzero(gt) != 0.:
        gt = ng_dist >= current_val
        lt = ng_dist < current_val + bin_size
        in_bin = lt & gt

        if np.count_nonzero(in_bin) == 0.:
            ng_dist[gt] -= bin_size
        else:
            current_val = current_val + bin_size

    return ng_dist


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


# \\  -------- PYSIMPLEGUI -------- //

AppFont = 'Any 16'
SliderFont = 'Any 14'
sg.theme('black')

# Get Copy numbers from fasta file, and view the log of them
tab1_input = [[sg.Text("Choose a file: "), sg.FileBrowse(key="-IN-"), sg.Button("Get Copy Number"), sg.Text(text="Molecule (dna, rna, protein):"), sg.InputText(key="mol")],
              [sg.Canvas(key='figCanvas', background_color='#FDF6E3')],
              [sg.Button("Plot Log Copy Number")]]

toggle_btn_off = b'iVBORw0KGgoAAAANSUhEUgAAACgAAAAoCAYAAACM/rhtAAAABmJLR0QA/wD/AP+gvaeTAAAED0lEQVRYCe1WTWwbRRR+M/vnv9hO7BjHpElMKSlpqBp6gRNHxAFVcKM3qgohQSqoqhQ45YAILUUVDRxAor2VAweohMSBG5ciodJUSVqa/iikaePEP4nj2Ovdnd1l3qqJksZGXscVPaylt7Oe/d6bb9/svO8BeD8vA14GvAx4GXiiM0DqsXv3xBcJU5IO+RXpLQvs5yzTijBmhurh3cyLorBGBVokQG9qVe0HgwiXLowdy9aKsY3g8PA5xYiQEUrsk93JTtjd1x3siIZBkSWQudUK4nZO1w3QuOWXV+HuP/fL85klAJuMCUX7zPj4MW1zvC0Ej4yMp/w++K2rM9b70sHBYCjo34x9bPelsgp/XJksZ7KFuwZjr3732YcL64ttEDw6cq5bVuCvgy/sje7rT0sI8PtkSHSEIRIKgCQKOAUGM6G4VoGlwiqoVd2Za9Vl8u87bGJqpqBqZOj86eEHGNch+M7otwHJNq4NDexJD+59RiCEQG8qzslFgN8ibpvZNsBifgXmFvJg459tiOYmOElzYvr2bbmkD509e1ylGEZk1Y+Ssfan18n1p7vgqVh9cuiDxJPxKPT3dfGXcN4Tp3dsg/27hUQs0qMGpRMYjLz38dcxS7Dm3nztlUAb38p0d4JnLozPGrbFfBFm79c8hA3H2AxcXSvDz7/+XtZE1kMN23hjV7LTRnKBh9/cZnAj94mOCOD32gi2EUw4FIRUMm6LGhyiik86nO5NBdGRpxYH14bbjYfJteN/OKR7UiFZVg5T27QHYu0RBxoONV9W8KQ7QVp0iXdE8fANUGZa0QAvfhhXlkQcmjJZbt631oIBnwKmacYoEJvwiuFgWncWnXAtuVBBEAoVVXWCaQZzxmYuut68b631KmoVBEHMUUrJjQLXRAQVSxUcmrKVHfjWWjC3XOT1FW5QrWpc5IJdQhDKVzOigEqS5dKHMVplnNOqrmsXqUSkn+YzWaHE9RW1FeXL7SKZXBFUrXW6jIV6YTEvMAUu0W/G3kcxPXP5ylQZs4fa6marcWvvZfJu36kuHjlc/nMSuXz+/ejxgqPFpuQ/xVude9eu39Jxu27OLvBGoMjrUN04zrNMbgVmOBZ96iPdPZmYntH5Ls76KuxL9NyoLA/brav7n382emDfHqeooXyhQmARVhSnAwNNMx5bu3V1+habun5nWdXhwJZ2C5mirTesyUR738sv7g88UQ0rEkTDlp+1wwe8Pf0klegUenYlgyg7bby75jUTITs2rhCAXXQ2vwxz84vlB0tZ0wL4NEcLX/04OrrltG1s8aOrHhk51SaK0us+n/K2xexBxljcsm1n6x/Fuv1PCWGiKOaoQCY1Vb9gWPov50+fdEqd21ge3suAlwEvA14G/ucM/AuppqNllLGPKwAAAABJRU5ErkJggg=='
toggle_btn_on = b'iVBORw0KGgoAAAANSUhEUgAAACgAAAAoCAYAAACM/rhtAAAABmJLR0QA/wD/AP+gvaeTAAAD+UlEQVRYCe1XzW8bVRCffbvrtbP+2NhOD7GzLm1VoZaPhvwDnKBUKlVyqAQ3/gAkDlWgPeVQEUCtEOIP4AaHSI0CqBWCQyXOdQuRaEFOk3g3IMWO46+tvZ+PeZs6apq4ipON1MNafrvreTPzfvub92bGAOEnZCBkIGQgZOClZoDrh25y5pdjruleEiX+A+rCaQo05bpuvJ/+IHJCSJtwpAHA/e269g8W5RbuzF6o7OVjF8D3Pr4tSSkyjcqfptPDMDKSleW4DKIggIAD5Yf+Oo4DNg6jbUBlvWLUNutAwZu1GnDjzrcXzGcX2AHw/emFUV6Sfk0pqcKpEydkKSo9q3tkz91uF5aWlo1Gs/mYc+i7tz4//19vsW2AU9O381TiioVCQcnlRsWeQhD3bJyH1/MiFLICyBHiuzQsD1arDvypW7DR9nzZmq47q2W95prm+I9fXfqXCX2AF2d+GhI98Y8xVX0lnxvl2UQQg0csb78ag3NjEeD8lXZ7pRTgftmCu4864OGzrq+5ZU0rCa3m+NzXlzvoAoB3+M+SyWQuaHBTEzKMq/3BMbgM+FuFCDBd9kK5XI5PJBKqLSev+POTV29lKB8rT0yMD0WjUSYLZLxzNgZvIHODOHuATP72Vwc6nQ4Uiw8MUeBU4nHS5HA6TYMEl02wPRcZBJuv+ya+UCZOIBaLwfCwQi1Mc4QXhA+PjWRkXyOgC1uIhW5Qd8yG2TK7kSweLcRGKKVnMNExWWBDTQsH9qVmtmzjiThQDs4Qz/OUSGTwcLwIQTLW58i+yOjpXDLqn1tgmDzXzRCk9eDenjo9yhvBmlizrB3V5dDrNTuY0A7opdndStqmaQLPC1WCGfShYRgHdLe32UrV3ntiH9LliuNrsToNlD4kruN8v75eafnSgC6Luo2+B3fGKskilj5muV6pNhk2Qqg5v7lZ51nBZhNBjGrbxfI1+La5t2JCzfD8RF1HTBGJXyDzs1MblONulEqPDVYXgwDIfNx91IUVbAbY837GMur+/k/XZ75UWmJ77ou5mfM1/0x7vP1ls9XQdF2z9uNsPzosXPNFA5m0/EX72TBSiqsWzN8z/GZB08pWq9VeEZ+0bjKb7RTD2i1P4u6r+bwypo5tZUumEcDAmuC3W8ezIqSGfE6g/sTd1W5p5bKjaWubrmWd29Fu9TD0GlYlmTx+8tTJoZeqYe2BZC1/JEU+wQR5TVEUPptJy3Fs+Vkzgf8lemqHumP1AnYoMZSwsVEz6o26i/G9Lgitb+ZmLu/YZtshfn5FZDPBCcJFQRQ+8ih9DctOFvdLIKHH6uUQnq9yhFu0bec7znZ+xpAGmuqef5/wd8hAyEDIQMjAETHwP7nQl2WnYk4yAAAAAElFTkSuQmCC'

tab_modelweights = [[sg.Text("Weight Base"), sg.InputText(key="mw_base")],
                    [sg.Text("Weight Max Exponent"), sg.InputText(key="mw_maxE")],
                    [sg.Text("Weight Min Exponent"), sg.InputText(key="mw_minE")],
                    [sg.Text("Weight Max Cutoff (Log Value)"), sg.InputText(key="mw_maxC")],
                    [sg.Text('Off'), sg.Button(image_data=toggle_btn_off, key='-TOGGLE-GRAPHIC-', button_color=(sg.theme_background_color(), sg.theme_background_color()), border_width=0, metadata=False), sg.Text('On')],
                    [sg.Canvas(key='mwCanvas', background_color='#FDF6E3')]
                    ]

tab2_sampling = [[sg.Text(text="Bin Size:",
                          font=SliderFont,
                          background_color='#FDF6E3',
                          pad=((0, 0), (10, 0)),
                          text_color='Black'),
                  sg.InputText(key="bin_size"),
                  sg.Button('Normalize Histogram',
                            font=AppFont,
                            pad=((4, 0), (10, 0)))],
                 [sg.Canvas(key='sampleCanvas', background_color='#FDF6E3')]]

# tab2_sampling = [[sg.Text(text="Log Separators (separate by space)",
#                           font=SliderFont,
#                           background_color='#FDF6E3',
#                           pad=((0, 0), (10, 0)),
#                           text_color='Black'),
#                   sg.InputText(key="separators"),
#                   sg.Button('Apply Separators',
#                             font=AppFont,
#                             pad=((4, 0), (10, 0)))],
#                  [sg.Canvas(key='sampleCanvas', background_color='#FDF6E3')]]


# tab3_layout = [[sg.Text('This is inside tab 3')]]
# tab4_layout = [[sg.Text('This is inside tab 4')]]


layout = [[sg.TabGroup([[sg.Tab('Tab 1', tab1_input, background_color='darkslateblue', key='-mykey-'),
                         sg.Tab('Tab 2', tab2_sampling, background_color='tan1'),
                         sg.Tab('Model Weights', tab_modelweights, background_color="darkslateblue")]])],
          # pad ((left, right), (top, bottom))
          [sg.Button('Exit', font=AppFont, pad=((540, 0), (0, 0)))]]

_VARS['window'] = sg.Window('Dataset Scaling',
                            layout,
                            finalize=True,
                            resizable=True,
                            location=(100, 100),
                            element_justification="center",
                            background_color='#FDF6E3')


# \\  -------- PYSIMPLEGUI -------- //


# \\  -------- PYPLOT -------- //


# def makeSynthData():
#     xData = np.random.randint(100, size=_VARS['dataSize'])
#     yData = np.linspace(0, _VARS['dataSize'],
#                         num=_VARS['dataSize'], dtype=int)
#     return (xData, yData)


def drawChart(Xdata, weights=None, bins=100):
    # _VARS['pltFig'] = plt.figure()
    if weights is None:
        plt.hist(Xdata, bins=bins)
    else:
        plt.hist(Xdata, bins=bins, weights=weights)
    plt.yscale("log")
    # _VARS['fig_agg'] = draw_figure(_VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])


def drawSep(sep, numbers):
    for sid, s in enumerate(sep):
        plt.axvline(s, c="r", alpha=0.2, lw=0.1)
        if sid != len(sep) - 1:
            plt.text(s + 0.3, 1.0, str(numbers[sid]), rotation=90, va='center')


# def updateChart():
#     _VARS['fig_agg'].get_tk_widget().forget()
#     dataXY = makeSynthData()
#     # plt.cla()
#     plt.clf()
#     plt.plot(dataXY[0], dataXY[1], '.k')
#     _VARS['fig_agg'] = draw_figure(
#         _VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])
#
#
# def updateData(val):
#     _VARS['dataSize'] = val
#     updateChart()

# \\  -------- PYPLOT -------- //


# drawChart()


from rbm_torch.utils.utils import fasta_read
import rbm_torch.utils.data_prep as dp


def assign_groups(distribution, boundaries):
    group_maps = [((distribution > boundaries[i]) & (distribution <= boundaries[i + 1])) for i in range(len(boundaries) - 1)]
    group_numbers = [np.count_nonzero(map) for map in group_maps]
    return group_numbers, group_maps


# _VARS['pltFig'] = plt.figure()
# _VARS['fig_agg'] = draw_figure(_VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])
# plt.close()

# _VARS['pltSample'] = plt.figure()
# _VARS['fig_sample'] = draw_figure(_VARS['window']['sampleCanvas'].TKCanvas, _VARS['pltSample'])
# plt.close()

# MAIN LOOP
while True:
    event, values = _VARS['window'].read(timeout=200)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break

    # Get Data for Input
    elif event == "Get Copy Number":
        assert values["mol"] in ["dna", "rna", "protein"]
        try:
            seqs, copy_num, chars, q = fasta_read(values["-IN-"], values["mol"])
            print(f"Successfuly Loaded {values['-IN-']}")
        except:
            print(f"Could Not Load File {values['-IN-']}")

    # Plot Log of Copy Numbers
    elif event == 'Plot Log Copy Number':
        log_cn = dp.log_scale(copy_num, eps=1.)
        if _VARS['pltFig'] is False:
            _VARS['pltFig'] = plt.figure()
        else:
            _VARS['fig_agg'].get_tk_widget().forget()
            plt.clf()
        drawChart(log_cn)
        _VARS['fig_agg'] = draw_figure(_VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])

    # Normalize the dataset by finding sampling weights so each copy number is just as likely as another
    elif event == "Normalize Histogram":
        bin_size = float(values["bin_size"])
        lcn_min = np.min(log_cn)
        lcn_max = np.max(log_cn)
        boundaries = np.arange(lcn_min - 0.7 * bin_size, lcn_max + 0.7 * bin_size, bin_size)
        gnums, gmps = assign_groups(log_cn, list(boundaries))

        sample_weights = np.full(log_cn.shape, 1.)

        for gid, group_num in enumerate(gnums):
            if group_num > 0:  # make sure the group has members
                sample_weights[gmps[gid]] /= group_num

        if _VARS['pltSample'] is False:
            _VARS['pltSample'] = plt.figure()
        else:
            _VARS['fig_sample'].get_tk_widget().forget()
            plt.clf()

        drawChart(log_cn, weights=sample_weights, bins=len(boundaries))
        drawSep(boundaries, gnums)
        _VARS['fig_sample'] = draw_figure(_VARS['window']['sampleCanvas'].TKCanvas, _VARS['pltSample'])



    elif event == "Apply Separators":
        separators = values["separators"].split()
        boundaries = [0.0, *separators, log_cn.max()]
        boundaries = [float(b) for b in boundaries]
        gnums, gmps = assign_groups(log_cn, boundaries)

        if _VARS['pltSample'] is False:
            _VARS['pltSample'] = plt.figure()
        else:
            _VARS['fig_sample'].get_tk_widget().forget()
            plt.clf()

        drawChart(log_cn)
        drawSep(boundaries, gnums)
        _VARS['fig_sample'] = draw_figure(_VARS['window']['sampleCanvas'].TKCanvas, _VARS['pltSample'])

    elif event == "Assign Sequence Model Weights":
        no_gaps_cns = remove_gaps_in_dist(log_cn, bin_size=0.1)
        model_weights = np.power(float(values["mw_base"]), dp.scale_values_np(np.asarray([x if x < float(values["mw_maxC"]) else float(values["mw_maxC"]) for x in no_gaps_cns]), min=float(values["mw_minE"]), max=float(values["mw_maxE"]))).squeeze(1)

    elif event == '-TOGGLE-GRAPHIC-':  # if the graphical button that changes images
        _VARS['window']['-TOGGLE-GRAPHIC-'].metadata = not _VARS['window']['-TOGGLE-GRAPHIC-'].metadata
        _VARS['window']['-TOGGLE-GRAPHIC-'].update(image_data=toggle_btn_on if _VARS['window']['-TOGGLE-GRAPHIC-'].metadata else toggle_btn_off)

_VARS['window'].close()

#################################################### What we're implementing

# from rbm_torch.utils.utils import fasta_read
# import rbm_torch.utils.data_prep as dp
# import matplotlib.pyplot as plt
# import pandas as pd
#
# # seqs, copy_num, chars, q = fasta_read(f"/home/jonah/PycharmProjects/phage_display_ML/datasets/exo/enriched_raw.fasta", "dna")
# seqs, copy_num, chars, q = fasta_read(f"/home/jonah/PycharmProjects/phage_display_ML/datasets/cov/late_5gcmax_raw.fasta", "dna")
#
# # groups = 2
#
# # Divides data set into bins. User can choose to resample the dataset for more uniform number in each bin or alternatively reweight the dataset.
#
# # Resample allows for more uniform representation to be learned as batches will consist of more uniformly distributed sequences
#
# # Reweight gives weights for each sequence that can be used with a weighted random sampler.
#
#
# # How to space the bins, options: "user" or "log"
# spacing_type = "user"
#
# resample_dist = False
# reweight_dist = True
#
# lazy_resample = "maybe do this later?? or nahhhh???"
#
# assert resample_dist is not reweight_dist
#
# if spacing_type == "user":
#     group_spacing_log = [0, 0.83, 0.845, 7.5]  # cov dataset spacing
#     # max_cutoff = 4
#
#     # group_spacing_log = [0, 0.4, 1.0, 1.2, 2.0, 15]  #exo dataset_spacing
#
#     # Reweight Spacing
#     # group_spacing_log = [0, 1.5, 2.5, 3.5, 4.5, 15]
#     groups = len(group_spacing_log) - 1
#
#     max_cutoff = 6
#     weight_exponent_base = 5
#     weight_exponent_min = -3
#     weight_exponent_max = 0
#     bin_weight_punishment = [1. for x in range(groups)]  # weights in corresponding bin are divided by this number
#
#     stratify_percentiles = [42, 42.5]
#
#     # per_bin_sample_number = "mean"
#
#     # Resample Spacing
#     # group_spacing_log = [0, 2.3, 15]  #exo dataset_spacing
#     # groups = len(group_spacing_log) - 1
#     #
#     # per_bin_sample_number = [25000, 50000]
#     # max_cutoff = 9
#     # weight_exponent_base = 2
#     # weight_exponent_min = -3
#     # weight_exponent_max = 0
#     # weight_power = 3
#     # bin_weight_punishment = [10, 1]  # weights in corresponding bin
#
#     # weight_key = "log_copy_num"
#     weight_power = 1
#     under_weight_key = None
#     over_weight_key = "log_copy_num"
#     per_bin_sample_number = [25000, 50000]
#
#
# elif spacing_type == "log":
#     groups = 10
#
# df = pd.DataFrame({"sequence": seqs, "copy_num": copy_num})
# copy_num = df["copy_num"].tolist()
# log_cn = dp.log_scale(copy_num, eps=1.)
#
# # Keep a copy so we can see the change in the distribution
# raw_dataset_log_cn = copy(log_cn)
#
# df["log_copy_num"] = log_cn
# log_cn = np.asarray(log_cn)  # Variable we are going to use for our resampling/reweighting scheme
#
# if spacing_type == "log":
#     # User Defined Bin Spacing
#     bin_edges, groups = log_spacing(log_cn, groups, min_bin_size=0.1, max_bin_size=0.8)
#
# elif spacing_type == "user":
#     # Using user defined group spacing
#     bin_edges = np.asarray(group_spacing_log)
#
# # Get numbers and bool arrays for sequences within a bin
# group_numbers, group_maps = [], []
# for i in range(groups):
#     gmap = log_cn > bin_edges[i]
#     lmap = log_cn <= bin_edges[i + 1]
#     group_map = gmap & lmap
#     group_number = log_cn[group_map].size
#     group_maps.append(group_map)
#     group_numbers.append(group_number)
#
# if resample_dist:
#     resampled_df, group_number, group_maps = resample(df, group_maps, group_numbers, per_bin_sample_number, weight_power=weight_power, under_weight_key=under_weight_key, over_weight_key=over_weight_key)
#
# elif reweight_dist:
#     df["weights"] = 1.
#     for i in range(groups):
#         df.loc[group_maps[i], "weights"] *= 1. / group_numbers[i]
#
#     resampled_df = df
# else:
#     resampled_df = df
#
# if resample_dist:
#     no_gaps_cns = remove_gaps_in_dist(log_cn, bin_size=0.1)
#     unsupervised_weights = np.power(weight_exponent_base, dp.scale_values_np(np.asarray([x if x < max_cutoff else max_cutoff for x in no_gaps_cns]), min=weight_exponent_min, max=weight_exponent_max).squeeze(1))
#
#     punishment = np.copy(log_cn)
#     for i in range(groups):
#         punishment[group_maps[i]] = bin_weight_punishment[i]
#     # pearson_weights = dp.scale_values_np(np.asarray([x for x in log_cn]))
#
#     unsupervised_weights /= punishment
#
# elif reweight_dist:
#     no_gaps_cns = remove_gaps_in_dist(log_cn, bin_size=0.1)
#     model_weights = np.power(weight_exponent_base, dp.scale_values_np(np.asarray([x if x < max_cutoff else max_cutoff for x in no_gaps_cns]), min=weight_exponent_min, max=weight_exponent_max)).squeeze(1)
#     unsupervised_weights = df["weights"].to_numpy() * model_weights
#
#     punishment = np.copy(log_cn)
#     for i in range(groups):
#         punishment[group_maps[i]] = bin_weight_punishment[i]
#
#     unsupervised_weights /= punishment
#
#     random_sample_df = copy(df)
#     random_sample_df["sample_weights"] = unsupervised_weights
#
#     rand_sample = random_sample_df.sample(10000, replace=True, weights=random_sample_df["sample_weights"])
#
# # if stratify_dist:
# distribution = model_weights
# dist_set = list(set(distribution))
# boundaries = [0.]
# boundaries += [np.percentile(dist_set, x) for x in stratify_percentiles]
# boundaries += [np.max(distribution)]
#
# stratify_group_maps = [((distribution > boundaries[i]) & (distribution <= boundaries[i + 1])) for i in range(len(boundaries) - 1)]
# stratify_group_numbers = [np.count_nonzero(map) for map in stratify_group_maps]
#
# print("Stratify Group Numbers", stratify_group_numbers)
# print("Stratify Seq. Weights Group Spacing", boundaries)
#
# plt.rcParams.update({'font.size': 6})
# fig, axs = plt.subplots(7, 2, sharex="col")
# axs[0, 0].hist(raw_dataset_log_cn, bins=100)
# axs[0, 0].set_yscale("log")
# axs[0, 0].set_ylabel("Raw Log \nSeq. #", multialignment='center')
# # axs[1].hist(log_cn, bins=100)
# axs[1, 0].hist(no_gaps_cns, bins=100)
# axs[1, 0].set_yscale("log")
# axs[1, 0].axvline(max_cutoff, c="g")
# axs[1, 0].set_ylabel("Log Seq. #\nNo Gaps", multialignment='center')
#
# axs[2, 0].hist(log_cn, bins=100, weights=unsupervised_weights)
# axs[2, 0].set_ylabel("Weighted\nDist.", multialignment='center')
#
# axs[3, 0].hist(log_cn, bins=100)
# axs[3, 0].set_yscale("log")
# axs[3, 0].set_ylabel("Log Seq.\n#", multialignment='center')
# # axs.set_yscale("log")
# for b in list(bin_edges):
#     axs[1, 0].axvline(b, c="r")
#     axs[2, 0].axvline(b, c="r")
#
# axs[4, 0].hist(unsupervised_weights)
# axs[4, 0].set_ylabel("Seq. Weights")
# axs[4, 0].set_yscale("log")
#
# for gid, group_number in enumerate(group_numbers):
#     width = bin_edges[gid + 1] - bin_edges[gid]
#
#     prob_seq_in_bin = unsupervised_weights[group_maps[gid]].sum()
#
#     axs[5, 0].bar([bin_edges[gid] + 0.5 * width], [group_number], color="b", width=width - 0.06)
#     axs[6, 0].bar([bin_edges[gid] + 0.5 * width], [prob_seq_in_bin], color="r", width=width - 0.06)
# axs[5, 0].set_yscale("log")
#
# axs[5, 0].set_ylabel("Log Seqs.\nper Bin", multialignment='center')
# axs[6, 0].set_ylabel("Unnormalized\nProb. per Bin", multialignment='center')
# axs[6, 0].set_xlabel("Log Enrichment Value")
#
# # axs[7].hist(rand_sample["log_copy_num"])
#
# axs[0, 1].hist(model_weights)
# axs[1, 1].hist(unsupervised_weights)
# axs[0, 1].set_yscale("log")
# axs[1, 1].set_yscale("log")
#
# # axs[1, 2].hist()
#
#
# # axs[3].hist(log_cn, bins=100, weights=pearson_weights)
# plt.show()
# plt.close()
#
# print(resampled_df.index.__len__())
# print(unsupervised_weights.size)
#
# # Separate into training and validation datasets?
#
# # weights = dp.scale_values_np(log_cn, min=0.01, max=1.0)
#
# # dp.dataframe_to_fasta(resampled_df, "/home/jonah/PycharmProjects/phage_display_ML/datasets/cov/late_5gcmax_resampled.fasta", count_key="copy_num")
#
#
# # dp.dataframe_to_fasta(resampled_df, "/home/jonah/PycharmProjects/phage_display_ML/datasets/exo/enriched_resampled.fasta", count_key="copy_num")
#
# # dp.make_weight_file("enriched_sampling_weights", model_weights.tolist(), dir="/home/jonah/PycharmProjects/phage_display_ML/datasets/exo/", other_data={"sampling_weights": unsupervised_weights.tolist()})
# dp.make_weight_file("late_5gc_sampling_weights", model_weights.tolist(), dir="/home/jonah/PycharmProjects/phage_display_ML/datasets/cov/", other_data={"sampling_weights": unsupervised_weights.tolist()})