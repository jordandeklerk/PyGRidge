import numpy as np


def seagull_group_lasso(
    y: np.ndarray,
    X: np.ndarray,
    feature_weights: np.ndarray,
    groups: np.ndarray,
    beta: np.ndarray,
    index_permutation: np.ndarray,
    epsilon_convergence: float,
    max_iterations: int,
    gamma: float,
    lambda_max: float,
    proportion_xi: float,
    num_intervals: int,
    num_fixed_effects: int,
    trace_progress: bool,
) -> dict:
    """
    Lasso, group lasso, and sparse-group lasso

    Parameters:
    y (np.ndarray): Numeric vector of observations.
    X (np.ndarray): Numeric design matrix relating y to fixed and random effects [X Z].
    feature_weights (np.ndarray): Numeric vector of weights for the vectors of fixed and random effects [b^T, u^T]^T.
    groups (np.ndarray): Integer vector specifying which effect (fixed and random) belongs to which group.
    beta (np.ndarray): Numeric vector whose partitions will be returned.
    index_permutation (np.ndarray): Integer vector that contains information about the original order of the user's input.
    epsilon_convergence (float): Value for relative accuracy of the solution.
    max_iterations (int): Maximum number of iterations for each value of the penalty parameter λ.
    gamma (float): Multiplicative parameter to decrease the step size during backtracking line search.
    lambda_max (float): Maximum value for the penalty parameter.
    proportion_xi (float): Multiplicative parameter to determine the minimum value of λ for the grid search.
    num_intervals (int): Number of lambdas for the grid search.
    num_fixed_effects (int): Number of fixed effects present in the mixed model.
    trace_progress (bool): If True, a message will occur on the screen after each finished loop of the λ grid.

    Returns:
    dict: A dictionary containing the results of the group lasso algorithm.
    """
    n, p = X.shape

    # Initialize variables
    num_groups = np.max(groups)
    index_start = np.zeros(num_groups, dtype=int)
    index_end = np.zeros(num_groups, dtype=int)
    group_sizes = np.zeros(num_groups, dtype=int)
    group_weights = np.zeros(num_groups)

    # Create vectors of group sizes, start indices, end indices, and group weights
    for i in range(num_groups):
        group_mask = groups == (i + 1)
        group_sizes[i] = np.sum(group_mask)
        index_start[i] = np.where(group_mask)[0][0]
        index_end[i] = np.where(group_mask)[0][-1]
        group_weights[i] = np.sqrt(np.sum(feature_weights[group_mask]))

    # Calculate X.T * y
    X_transp_y = X.T @ y

    # Initialize result matrices
    iterations = np.zeros(num_intervals, dtype=int)
    lambdas = np.zeros(num_intervals)
    solution = np.zeros((num_intervals, p))

    for interval in range(num_intervals):
        accuracy_reached = False
        counter = 1
        if num_intervals > 1:
            lambda_val = lambda_max * np.exp(
                (interval / (num_intervals - 1)) * np.log(proportion_xi)
            )
        else:
            lambda_val = lambda_max

        while (not accuracy_reached) and (counter <= max_iterations):
            # Calculate gradient
            gradient = (X.T @ (X @ beta) - X_transp_y) / n

            criterion_fulfilled = False
            time_step = 1.0

            while not criterion_fulfilled:
                beta_new = np.zeros_like(beta)

                # Soft-scaling in groups
                for i in range(num_groups):
                    start, end = index_start[i], index_end[i] + 1
                    temp = beta[start:end] - time_step * gradient[start:end]
                    l2_norm = np.linalg.norm(temp)
                    threshold = time_step * lambda_val * group_weights[i]

                    if l2_norm > threshold:
                        scaling = 1.0 - threshold / l2_norm
                        beta_new[start:end] = scaling * temp

                # Check convergence
                beta_diff = beta - beta_new
                loss_old = 0.5 * np.sum((y - X @ beta) ** 2) / n
                loss_new = 0.5 * np.sum((y - X @ beta_new) ** 2) / n

                if loss_new <= loss_old - np.dot(gradient, beta_diff) + (
                    0.5 / time_step
                ) * np.sum(beta_diff**2):
                    if np.max(
                        np.abs(beta_diff)
                    ) <= epsilon_convergence * np.linalg.norm(beta):
                        accuracy_reached = True
                    beta = beta_new
                    criterion_fulfilled = True
                else:
                    time_step *= gamma

            counter += 1

        if trace_progress:
            print(f"Loop: {interval + 1} of {num_intervals} finished.")

        # Store solution
        solution[interval] = beta[index_permutation - 1]
        iterations[interval] = counter - 1
        lambdas[interval] = lambda_val

    # Prepare results
    if num_fixed_effects == 0:
        return {
            "random_effects": solution,
            "lambda": lambdas,
            "iterations": iterations,
            "rel_acc": epsilon_convergence,
            "max_iter": max_iterations,
            "gamma_bls": gamma,
            "xi": proportion_xi,
            "loops_lambda": num_intervals,
        }
    else:
        return {
            "fixed_effects": solution[:, :num_fixed_effects],
            "random_effects": solution[:, num_fixed_effects:],
            "lambda": lambdas,
            "iterations": iterations,
            "rel_acc": epsilon_convergence,
            "max_iter": max_iterations,
            "gamma_bls": gamma,
            "xi": proportion_xi,
            "loops_lambda": num_intervals,
        }
