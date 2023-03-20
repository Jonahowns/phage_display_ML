import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
from PIL import Image

from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.ticker import FormatStrFormatter



## Implementation inspired from https://stackoverflow.com/questions/42615527/sequence-logos-in-matplotlib-aligning-xticks
## Color choice inspired from: http://weblogo.threeplusone.com/manual.html

def clean_ax(ax):
    ax.axis("off")

def get_ax(ax, i, nrows, ncols):
    if (ncols > 1) & (nrows > 1):
        col = int(i % ncols)
        row = int(i / ncols)
        ax_ = ax[row, col]
    elif (ncols > 1) & (nrows == 1):
        ax_ = ax[i]
    elif (ncols == 1) & (nrows > 1):
        ax_ = ax[i]
    else:
        ax_ = ax
    return ax_


def select_sites(W, window=5, theta_important=0.25):
    n_sites = W.shape[0]
    norm = np.abs(W).sum(-1)
    important = np.nonzero(norm / norm.max() > theta_important)[0]
    selected = []
    for imp in important:
        selected += range(max(0, imp - window), min(imp + window + 1, n_sites))
    selected = np.unique(selected)
    return selected


def ticksAt(selected, ticks_every=10):
    n_selected = len(selected)
    all_ticks = []
    all_ticks_labels = []
    previous_select = selected[0]
    k = 0
    for select in selected:
        if (select - previous_select) > 1:
            k += 1
        if (select % ticks_every == 0) | ((select - previous_select) > 1):
            if not k in all_ticks:
                all_ticks.append(k + 1)
                all_ticks_labels.append(select + 1)
        previous_select = copy.copy(select)
        k += 1
    return np.array(all_ticks), np.array(all_ticks_labels)


def breaksAt(x, maxi_size, ax):
    ax.plot([x, x], [-maxi_size, maxi_size], linewidth=5, c='black', linestyle='--')


def letterAt(letter, x, y, letters, color_scheme, yscale=1, ax=None):
    text = letters[letter]
    t = mpl.transforms.Affine2D().scale(1 * globscale, yscale * globscale) + \
        mpl.transforms.Affine2D().translate(x, y) + ax.transData
    p = PathPatch(text, lw=0, fc=color_scheme[letter], transform=t)
    if ax != None:
        ax.add_artist(p)
    return p


def aa_color(letter):
    if letter in ['C']:
        return 'green'
    elif letter in ['F', 'W', 'Y']:
        return [199 / 256., 182 / 256., 0., 1.]  # 'gold'
    elif letter in ['Q', 'N', 'S', 'T']:
        return 'purple'
    elif letter in ['V', 'L', 'I', 'M']:
        return 'black'
    elif letter in ['K', 'R', 'H']:
        return 'blue'
    elif letter in ['D', 'E']:
        return 'red'
    elif letter in ['A', 'P', 'G']:
        return 'grey'
    elif letter in ['$\\boxminus$']:
        return 'black'
    else:
        return 'black'

def nuc_color(letter):
    if letter in ['C']:
        return 'firebrick'
    elif letter in ['G']:
        return 'tomato'
    elif letter in ['T', "U"]:
        return 'mediumpurple'
    elif letter in ['A']:
        return 'slateblue'
    elif letter in ['$\\boxminus$']:
        return 'black'
    else:
        return 'black'


def build_scores(matrix, base_list, epsilon=1e-4):
    n_sites = matrix.shape[0]
    n_colors = matrix.shape[1]
    all_scores = []
    for site in range(n_sites):
        conservation = np.log2(21) + (np.log2(matrix[site] + epsilon) * matrix[site]).sum()
        liste = []
        order_colors = np.argsort(matrix[site])
        for c in order_colors:
            liste.append((base_list[c], matrix[site, c] * conservation))
        all_scores.append(liste)
    return all_scores


def build_scores2(matrix, base_list):
    n_sites = matrix.shape[0]
    n_colors = matrix.shape[1]
    epsilon = 1e-4
    all_scores = []
    for site in range(n_sites):
        liste = []
        c_pos = np.nonzero(matrix[site] >= 0)[0]
        c_neg = np.nonzero(matrix[site] < 0)[0]

        order_colors_pos = c_pos[np.argsort(matrix[site][c_pos])]
        order_colors_neg = c_neg[np.argsort(-matrix[site][c_neg])]
        for c in order_colors_pos:
            liste.append((base_list[c], matrix[site, c], '+'))
        for c in order_colors_neg:
            liste.append((base_list[c], -matrix[site, c], '-'))
        all_scores.append(liste)
    return all_scores


def build_scores_break(matrix, selected, base_list, epsilon=1e-4):
    has_breaks = (selected[1:] - selected[:-1]) > 1
    has_breaks = np.concatenate((np.zeros(1), has_breaks), axis=0)
    n_sites = len(selected)
    n_colors = matrix.shape[1]

    epsilon = 1e-4
    all_scores = []
    maxi_size = 0
    for site, has_break in zip(selected, has_breaks):
        if has_break:
            all_scores.append([('BREAK', 'BREAK', 'BREAK')])
        #             all_scores.append([('BREAK','BREAK','BREAK')] )
        conservation = np.log2(21) + (np.log2(matrix[site] + epsilon) * matrix[site]).sum()
        liste = []
        order_colors = np.argsort(matrix[site])
        for c in order_colors:
            liste.append((base_list[c], matrix[site, c] * conservation))
        maxi_size = max(maxi_size, conservation)
        all_scores.append(liste)
    return all_scores, maxi_size


def build_scores2_break(matrix, selected, base_list):
    has_breaks = (selected[1:] - selected[:-1]) > 1
    has_breaks = np.concatenate((np.zeros(1), has_breaks), axis=0)
    n_sites = len(selected)
    n_colors = matrix.shape[1]

    epsilon = 1e-4
    all_scores = []
    for site, has_break in zip(selected, has_breaks):
        if has_break:
            all_scores.append([('BREAK', 'BREAK', 'BREAK')])
        liste = []
        c_pos = np.nonzero(matrix[site] >= 0)[0]
        c_neg = np.nonzero(matrix[site] < 0)[0]

        order_colors_pos = c_pos[np.argsort(matrix[site][c_pos])]
        order_colors_neg = c_neg[np.argsort(-matrix[site][c_neg])]
        for c in order_colors_pos:
            liste.append((base_list[c], matrix[site, c], '+'))
        for c in order_colors_neg:
            liste.append((base_list[c], -matrix[site, c], '-'))
        all_scores.append(liste)
    maxi_size = np.abs(matrix).sum(-1).max()
    return all_scores, maxi_size

# Needed to generate Sequence Logos

fp = FontProperties(family="Arial", weight="bold")
globscale = 1.35
aa_to_dna = {'A': 'A', 'C': 'C', 'D': 'G', 'E': 'T', 'F': '$\\boxminus$'}
aa_to_rna = {'A': 'A', 'C': 'C', 'D': 'G', 'E': 'U', 'F': '$\\boxminus$'}

list_dna = ['A', 'C', 'G', 'T', '$\\boxminus$']
list_rna = ['A', 'C', 'G', 'U', '$\\boxminus$']
list_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '$\\boxminus$']
base_list_master = {"dna": list_dna, "rna": list_rna, "protein": list_aa}



LETTERSaa = dict([(letter, TextPath((-0.30, 0), letter, size=1, prop=fp)) for letter in list_aa])
LETTERSdna = dict([(letter, TextPath((-0.30, 0), letter, size=1, prop=fp)) for letter in list_dna])
LETTERSrna = dict([(letter, TextPath((-0.30, 0), letter, size=1, prop=fp)) for letter in list_rna])


COLOR_SCHEME_PROTEIN = dict([(letter, aa_color(letter)) for letter in list_aa])
COLOR_SCHEME_DNA = dict([(letter, nuc_color(letter)) for letter in list_dna])
COLOR_SCHEME_RNA = dict([(letter, nuc_color(letter)) for letter in list_rna])

LETTERS = {"protein": LETTERSaa, "dna": LETTERSdna, "rna": LETTERSrna}
COLOR_SCHEME = {"protein": COLOR_SCHEME_PROTEIN, "dna": COLOR_SCHEME_DNA, "rna": COLOR_SCHEME_RNA}


def Sequence_logo(matrix, ax=None, data_type=None, figsize=None, ylabel=None, title=None, epsilon=1e-4, show=True, ticks_every=1, ticks_labels_size=14, title_size=20, molecule='protein'):
    if data_type is None:
        if matrix.min() >= 0:
            data_type = 'mean'
        else:
            data_type = 'weights'

    base_list = base_list_master[molecule]
    color_scheme = COLOR_SCHEME[molecule]
    letters = LETTERS[molecule]

    if data_type == 'mean':
        all_scores = build_scores(matrix, base_list, epsilon=epsilon)
    elif data_type == 'weights':
        all_scores = build_scores2(matrix, base_list)
    else:
        print('data type not understood')
        return -1

    if ax is not None:
        show = False
        return_fig = False
    else:
        if figsize is None:
            figsize = (max(int(0.3 * matrix.shape[0]), 2), 3)
        fig, ax = plt.subplots(figsize=figsize)
        return_fig = True

    x = 1
    maxi = 0
    mini = 0
    for scores in all_scores:
        if data_type == 'mean':
            y = 0
            for base, score in scores:
                if score > 0.01:
                    letterAt(base, x, y, letters, color_scheme, yscale=score, ax=ax)
                y += score
            x += 1
            maxi = max(maxi, y)


        elif data_type == 'weights':
            y_pos = 0
            y_neg = 0
            for base, score, sign in scores:
                if sign == '+':
                    letterAt(base, x, y_pos, letters, color_scheme, yscale=score, ax=ax)
                    y_pos += score
                else:
                    y_neg += score
                    letterAt(base, x, -y_neg, letters, color_scheme, yscale=score, ax=ax)
            x += 1
            maxi = max(y_pos, maxi)
            mini = min(-y_neg, mini)

    if data_type == 'weights':
        maxi = max(maxi, abs(mini))
        mini = -maxi

    if ticks_every > 1:
        xticks = range(1, x)
        xtickslabels = ['%s' % k if k % ticks_every == 0 else '' for k in xticks]
        ax.set_xticks(xticks, xtickslabels)
    else:
        ax.set_xticks(range(1, x))
    ax.set_xlim((0, x))
    ax.set_ylim((mini, maxi))
    if ylabel is None:
        if data_type == 'mean':
            ylabel = 'Conservation (bits)'
        elif data_type == 'weights':
            ylabel = 'Weights'
    ax.set_ylabel(ylabel, fontsize=title_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', which='major', labelsize=ticks_labels_size)
    ax.tick_params(axis='both', which='minor', labelsize=ticks_labels_size)
    if title is not None:
        ax.set_title(title, fontsize=title_size)
    if return_fig:
        plt.tight_layout()
        if show:
            plt.show()
        return fig


def Sequence_logo_breaks(matrix, data_type=None, selected=None, window=5, theta_important=0.25, figsize=None, nrows=1, ylabel=None, title=None, epsilon=1e-4, show=True, ticks_every=5, ticks_labels_size=14, title_size=20, molecule='protein'):
    if data_type is None:
        if matrix.min() >= 0:
            data_type = 'mean'
        else:
            data_type = 'weights'

    if selected is None:
        if data_type == 'mean':
            'NO SELECTION SUPPORTED FOR MEAN VECTOR'
            return
        else:
            selected = select_sites(matrix, window=window, theta_important=theta_important)
    else:
        selected = np.array(selected)
    print('Number of sites selected: %s' % len(selected))

    xticks, xticks_labels = ticksAt(selected, ticks_every=ticks_every)

    base_list = base_list_master[molecule]
    color_scheme = COLOR_SCHEME[molecule]
    letters = LETTERS[molecule]

    if data_type == 'mean':
        all_scores, maxi_size = build_scores_break(matrix, selected, base_list, epsilon=epsilon)
    elif data_type == 'weights':
        all_scores, maxi_size = build_scores2_break(matrix, selected, base_list)
    else:
        print('data type not understood')
        return -1

    nbreaks = ((selected[1:] - selected[:-1]) > 1).sum()
    width = (len(selected) + nbreaks) / nrows

    if figsize is None:
        figsize = (max(int(0.3 * width), 2), 3 * nrows)

    fig, ax = plt.subplots(figsize=figsize, nrows=nrows)
    if nrows > 1:
        for row in range(nrows):
            ax[row].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    x = 1
    maxi = 0
    mini = 0

    if nrows > 1:
        row = 0
        ax_ = ax[row]
        xmins = np.zeros(nrows)
        xmaxs = np.ones(nrows) * (len(selected) + nbreaks + 1)
    else:
        ax_ = ax

    for scores in all_scores:
        if data_type == 'mean':
            y = 0
            if scores[0][0] == 'BREAK':
                breaksAt(x, maxi_size, ax_)
                if nrows > 1:
                    if x > (1 + row) * width:
                        xmaxs[row] = copy.copy(x) + 1
                        xmins[row + 1] = copy.copy(x)
                        row += 1
                        ax_ = ax[row]
            else:
                for base, score in scores:
                    if score > 0.01:
                        letterAt(base, x, y, letters, color_scheme, yscale=score, ax=ax_)
                    y += score
                x += 1
                maxi = max(maxi, y)


        elif data_type == 'weights':
            y_pos = 0
            y_neg = 0
            if scores[0][0] == 'BREAK':
                breaksAt(x, maxi_size, ax_)
                if nrows > 1:
                    if x > (1 + row) * width:
                        xmaxs[row] = copy.copy(x) + 1
                        xmins[row + 1] = copy.copy(x)
                        row += 1
                        ax_ = ax[row]

            else:
                for base, score, sign in scores:
                    if sign == '+':
                        letterAt(base, x, y_pos, letters, color_scheme, yscale=score, ax=ax_)
                        y_pos += score
                    else:
                        y_neg += score
                        letterAt(base, x, -y_neg, letters, color_scheme, yscale=score, ax=ax_)
            x += 1
            maxi = max(y_pos, maxi)
            mini = min(-y_neg, mini)

    if data_type == 'weights':
        maxi = max(maxi, abs(mini))
        mini = -maxi

    if nrows > 1:
        for row in range(nrows):
            ax[row].set_xlim((xmins[row], xmaxs[row]))
            ax[row].set_ylim((mini, maxi))
            subset = (xticks > xmins[row]) & (xticks < xmaxs[row])
            ax[row].set_xticks(xticks[subset])
            ax[row].set_xticklabels(xticks_labels[subset])
    else:
        plt.xticks(xticks, xticks_labels)
        plt.xlim((0, x))
        plt.ylim((mini, maxi))

    if ylabel is None:
        if data_type == 'mean':
            ylabel = 'Conservation (bits)'
        elif data_type == 'weights':
            ylabel = 'Weights'
    if nrows > 1:
        ax[0].set_ylabel(ylabel, fontsize=title_size)
        for row in range(1, nrows):
            ax[row].set_ylabel('. . .', fontsize=title_size)
    else:
        ax.set_ylabel(ylabel, fontsize=title_size)

    if nrows > 1:
        for k in range(nrows):
            ax[k].spines['right'].set_visible(False)
            ax[k].spines['top'].set_visible(False)
            ax[k].yaxis.set_ticks_position('left')
            ax[k].xaxis.set_ticks_position('bottom')
            ax[k].tick_params(axis='both', which='major', labelsize=ticks_labels_size)
            ax[k].tick_params(axis='both', which='minor', labelsize=ticks_labels_size)

    else:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.tick_params(axis='both', which='major', labelsize=ticks_labels_size)
        ax.tick_params(axis='both', which='minor', labelsize=ticks_labels_size)

    if title is not None:
        if nrows > 1:
            ax[0].set_title(title, fontsize=title_size)
        else:
            ax.set_title(title, fontsize=title_size)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, selected


def Sequence_logo_multiple(matrix, data_type=None, figsize=None, ylabel=None, title=None, epsilon=1e-4, ncols=1, show=True, count_from=0, ticks_every=1, ticks_labels_size=14, title_size=20, molecule='protein'):
    if data_type is None:
        if matrix.min() >= 0:
            data_type = 'mean'
        else:
            data_type = 'weights'

    N_plots = matrix.shape[0]
    nrows = int(np.ceil(N_plots / float(ncols)))

    if figsize is None:
        figsize = (max(int(0.3 * matrix.shape[1]), 2), 3)

    figsize = (figsize[0] * ncols, figsize[1] * nrows)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    if ylabel is None:
        if data_type == 'mean':
            ylabel = 'Conservation (bits)'
        elif data_type == 'weights':
            ylabel = 'Weights'
    if type(ylabel) == str:
        ylabels = [ylabel + ' #%s' % i for i in range(1 + count_from, N_plots + count_from + 1)]
    else:
        ylabels = ylabel

    if title is None:
        title = ''
    if type(title) == str:
        titles = [title for _ in range(N_plots)]
    else:
        titles = title

    for i in range(N_plots):
        ax_ = get_ax(ax, i, nrows, ncols)

        Sequence_logo(matrix[i], ax=ax_, data_type=data_type, ylabel=ylabels[i], title=titles[i],
                      epsilon=epsilon, show=False, ticks_every=ticks_every, ticks_labels_size=ticks_labels_size, title_size=title_size, molecule=molecule)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def Sequence_logo_all(matrix, name='all_Sequence_logo.pdf', nrows=5, ncols=2, data_type=None, figsize=None, ylabel=None, title=None, epsilon=1e-4, ticks_every=5, ticks_labels_size=14, title_size=20, dpi=100, molecule='protein'):
    if data_type is None:
        if matrix.min() >= 0:
            data_type = 'mean'
        else:
            data_type = 'weights'
    n_plots = matrix.shape[0]
    plots_per_page = nrows * ncols
    n_pages = int(np.ceil(n_plots / float(plots_per_page)))
    rng = np.random.randn(1)[0]  # avoid file conflicts in case of multiple threads.
    mini_name = name[:-4]
    images = []
    for i in range(n_pages):
        if type(ylabel) == list:
            ylabel_ = ylabel[i * plots_per_page:min(plots_per_page * (i + 1), n_plots)]
        else:
            ylabel_ = ylabel
        if type(title) == list:
            title_ = title[i * plots_per_page:min(plots_per_page * (i + 1), n_plots)]
        else:
            title_ = title
        fig = Sequence_logo_multiple(matrix[plots_per_page * i:min(plots_per_page * (i + 1), n_plots)], data_type=data_type, figsize=figsize, ylabel=ylabel_, title=title_, epsilon=epsilon, ncols=ncols, show=False, count_from=plots_per_page * i, ticks_every=ticks_every,
                                     ticks_labels_size=ticks_labels_size, title_size=title_size, molecule=molecule)
        file = f"tmp_{rng}_#{i}.jpg"
        fig.savefig(mini_name + file, dpi=dpi)
        fig.clear()
        images.append(Image.open(mini_name + file))
        command = 'rm ' + mini_name + file
        os.system(command)

    images[0].save(name, "PDF", resolution=100.0, save_all=True, append_images=images[1:])
    return 'done'
