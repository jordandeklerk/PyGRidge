import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, cut_tree


def merge_groups(
    high_dim_data,
    init_groups,
    max_groups,
    method_distance="cityblock",
    method_clust="complete",
):
    """
    Merge initial groups into a specified number of new groups based on their first principal components.

    Args:
    high_dim_data (np.ndarray): High-dimensional data array (features x samples).
    init_groups (dict): Initial grouping of features.
    max_groups (int): Maximum number of new groups to create.
    method_distance (str, optional): Distance metric for clustering. Defaults to "cityblock" (Manhattan).
    method_clust (str, optional): Linkage method for hierarchical clustering. Defaults to "complete".

    Returns:
    dict: A dictionary containing new groups and their members.

    Raises:
    ValueError: If input parameters are invalid.
    """

    # Input validation
    if high_dim_data is None or init_groups is None or max_groups is None:
        raise ValueError(
            "Please provide valid high_dim_data, init_groups, and max_groups."
        )

    if max_groups < 2:
        raise ValueError("The number of new groups must be at least 2.")

    if len(init_groups) < max_groups:
        raise ValueError(
            "The number of new groups cannot be greater than the number of initial"
            " groups."
        )

    clust_names = list(init_groups.keys())
    n_clust = len(clust_names)
    n_samp = high_dim_data.shape[1]

    eigen_clust = np.zeros((n_samp, n_clust))

    for i in range(n_clust):
        id_clust = init_groups[clust_names[i]]
        dat_clust = high_dim_data[id_clust, :].T
        if dat_clust.shape[1] > 1:
            cov_dat = np.cov(dat_clust, rowvar=False)
            eig_vec = np.linalg.eig(cov_dat)[1][:, 0]
            eigen_clust[:, i] = dat_clust @ eig_vec
        else:
            eigen_clust[:, i] = dat_clust.flatten()

    # Hierarchical clustering
    dist_eigen_clust = pdist(eigen_clust.T, metric=method_distance)
    linkage_matrix = linkage(dist_eigen_clust, method=method_clust)
    new_groups = cut_tree(linkage_matrix, n_clusters=max_groups).flatten()

    group_new = {}
    new_clust_members = {}

    for i in range(max_groups):
        id_clust_i = np.where(new_groups == i)[0]
        new_clust_members[i] = [clust_names[j] for j in id_clust_i]
        group_new[i] = list(
            set(
                [
                    item
                    for sublist in [init_groups[clust_names[j]] for j in id_clust_i]
                    for item in sublist
                ]
            )
        )

    return {"new_groups": group_new, "new_group_members": new_clust_members}
