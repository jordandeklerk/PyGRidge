import numpy as np
from scipy import optimize
from typing import List, Union, Optional

def create_variable_groups(values: Union[List, np.ndarray],
                           variable_names: Optional[List[str]] = None,
                           subset: Optional[List] = None,
                           group_size: Optional[int] = None,
                           sort_descending: bool = True,
                           uniform_groups: bool = False,
                           num_groups: int = 10,
                           min_group_size: int = 25) -> dict:
    """
    Create groups of variables based on various input parameters.

    Args:
        values (Union[List, np.ndarray]): The main input vector to be grouped.
        variable_names (Optional[List[str]], optional): Names of variables. Defaults to None.
        subset (Optional[List], optional): Optional subset of variables. Defaults to None.
        group_size (Optional[int], optional): Size of each group. Defaults to None.
        sort_descending (bool, optional): Determine sorting order. Defaults to True.
        uniform_groups (bool, optional): Create uniform group sizes. Defaults to False.
        num_groups (int, optional): Number of groups to create. Defaults to 10.
        min_group_size (int, optional): Minimum size for each group. Defaults to 25.

    Returns:
        dict: A dictionary of groups, where each key is a group name and each value is a list of indices.

    Raises:
        ValueError: If input parameters are invalid or incompatible.
    """
    values = np.array(values)

    if len(values) == 0:
        raise ValueError("Input vector is empty")

    if values.dtype.kind in 'OSU':  # Object, String, or Unicode type
        if variable_names is None:
            raise ValueError("Please specify a character vector for variable_names")
        included_indices = np.where(np.isin(variable_names, values))[0]
        groups = {
            'VarIn': included_indices.tolist(),
            'VarOut': list(set(range(len(variable_names))) - set(included_indices))
        }
    elif np.issubdtype(values.dtype, np.number):
        if uniform_groups or num_groups * min_group_size >= len(values):
            if group_size is None:
                group_size = max(1, len(values) // num_groups)
                print(f"Group size set to: {group_size}")
            else:
                print(f"Group size {group_size}")

            sorted_indices = np.argsort(values)[::-1] if sort_descending else np.argsort(values)
            total_elements = len(values)

            if sort_descending:
                print("Sorting values in descending order, assuming small values are LESS relevant")
            else:
                print("Sorting values in ascending order, assuming small values are MORE relevant")

            actual_num_groups = max(1, total_elements // group_size)
            groups = {f"group{i+1}": sorted_indices[i*group_size:(i+1)*group_size].tolist() if i < actual_num_groups - 1 
                      else sorted_indices[i*group_size:].tolist() for i in range(actual_num_groups)}
        else:
            sorted_indices = np.argsort(values)[::-1] if sort_descending else np.argsort(values)
            total_elements = len(values)

            ratio = total_elements / min_group_size
            growth_factor = ratio ** (1 / num_groups)

            left_term = ratio + 1
            def group_size_function(x):
                return 1 - x**(num_groups+1) - left_term*(1-x)

            root = optimize.brentq(group_size_function, 1.000001, growth_factor)

            group_sizes = np.array([np.floor(min_group_size * root**i) if i == 1 else np.round(min_group_size * root**i) for i in range(1, num_groups+1)])
            total_size = np.sum(group_sizes)
            group_sizes[-1] -= (total_size - total_elements)

            print("Summary of group sizes:")
            print(f"Min: {np.min(group_sizes)}, Max: {np.max(group_sizes)}, Mean: {np.mean(group_sizes):.2f}, Median: {np.median(group_sizes)}")

            cumulative_sizes = np.cumsum(np.concatenate(([0], group_sizes))).astype(int)
            groups = {f"group{i+1}": sorted_indices[cumulative_sizes[i]:cumulative_sizes[i+1]].tolist() for i in range(num_groups)}
    else:
        raise ValueError("Argument 'values' is not correctly specified")

    if not np.issubdtype(values.dtype, np.character) and subset is not None:
        if variable_names is None:
            raise ValueError("variable_names required for subsetting")
        subset_indices = [variable_names.index(s) if s in variable_names else None for s in subset]
        groups = {k: [i for i in v if i in subset_indices] for k, v in groups.items()}

    print("Summary of group sizes:")
    print({k: len(v) for k, v in groups.items()})

    return groups
