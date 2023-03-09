import torch

def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    return X[torch.randperm(num_samples)[:num_clusters]]

def euclidean_distance(x, y):
    # N*1*M
    A = x.unsqueeze(dim=1)

    # 1*N*M
    B = y.unsqueeze(dim=0)

    dis = torch.abs(A - B)
    # return N*N matrix for pairwise distance
    # dis = dis.sum(dim=-1).squeeze()
    return dis

def kmeans(X, num_clusters, tol=1e-4, iter_limit=0):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param seed: (int) seed for kmeans
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :param tqdm_flag: Allows to turn logs on and off
    :param iter_limit: hard limit for max number of iterations
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    X = X.float()

    initial_state = initialize(X, num_clusters)
    if initial_state.shape[0] == 1:
        print("gotcha bitch")
    iteration = 0

    while True:
        dis = euclidean_distance(X, initial_state)
        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze()

            selected = torch.index_select(X, 0, selected)

            # https://github.com/subhadarship/kmeans_pytorch/issues/16
            if selected.shape[0] == 0:
                selected = X[torch.randint(len(X), (1,))]

            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                (initial_state - initial_state_pre) ** 2
            ))

        # increment iteration
        iteration = iteration + 1

        if center_shift ** 2 < tol:
            break
        if iter_limit != 0 and iteration >= iter_limit:
            break

    return choice_cluster, initial_state
