#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 2024

@author: lefteris
"""
# =============================================================================
# Interpreting Results
# 
#     Foresight (Oracle):
#         Provides the maximum achievable revenue, serving as the upper bound for comparison.
# 
#     Thompson Sampling:
#         Likely to perform close to the Oracle over time, depending on how well it balances exploration and exploitation.
# 
#     Uniform Pricing:
#         Provides a simple baseline. Expected to perform poorly in non-stationary or complex demand scenarios.
# 
#     Online Learning:
#         Can perform well if the discount factor is properly tuned, though it may struggle to adapt if demand patterns shift too quickly.
# 
# =============================================================================

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from demands import read_and_fit_demand, demand_from_csv
from demands import demand_function_wrapper, plot_demand, DEMAND_TYPE_MAPPING
import inspect



def revenue(price, demand):
    """
    Calculate revenue from price and demand.

    Parameters
    ----------
    price : float
        Price point
    demand : float
        Observed or predicted demand

    Returns
    -------
    float
        Revenue = price × demand
    """
    return price * demand


def foresight(prices, T, sigma, demand_function, demand_params, **kwargs):
    """
    Oracle pricing policy with perfect foresight of demand function.

    This represents the theoretical upper bound on performance. At each time
    step, it evaluates all candidate prices, observes the true demand (with
    noise), and selects the price that maximizes revenue for that specific
    realization. This is unattainable in practice but serves as a benchmark.

    Parameters
    ----------
    prices : array-like of float
        Candidate prices to evaluate, shape (n_prices,)
    T : int
        Number of time periods
    sigma : float
        Standard deviation of demand noise
    demand_function : callable
        Demand function with signature: demand_function(price, sigma, **params)
    demand_params : dict
        Parameters to pass to demand_function
    **kwargs : dict
        Additional arguments (unused)

    Returns
    -------
    revenues : list of float
        Revenue achieved at each time step, length T
    chosen_prices : list of float
        Optimal price chosen at each time step, length T

    Notes
    -----
    For stationary demand, the oracle will consistently choose the same
    optimal price. For time-varying demand, the optimal price changes.

    The oracle observes noisy demand realizations (not expected demand),
    making it a more realistic upper bound than a noise-free oracle.
    """
    revenues = []
    chosen_prices = []
    #Should be the same for stationary demands
    for t in range(T):
        demands = [demand_function_wrapper(p, sigma,demand_function ,time_step=t, **demand_params) for p in prices]
        arg_max = np.argmax([p * d for p, d in zip(prices, demands)])
        best_price = prices[arg_max]
        best_demand = demands[arg_max]
        revenues.append(revenue(best_price, best_demand))
        chosen_prices.append(best_price)
    return revenues, chosen_prices

def thompson_sampling(prices, T, sigma, demand_function, demand_params, **kwargs):
    """
    Thompson Sampling pricing policy using Bayesian linear regression.

    Maintains a posterior distribution over linear demand parameters (a, b)
    where demand ≈ a + b*price. At each time step, samples parameters from
    the posterior, chooses the revenue-maximizing price for that sample, then
    updates the posterior with observed demand.

    This naturally balances exploration (when posterior uncertainty is high)
    and exploitation (when posterior concentrates on true parameters).

    Parameters
    ----------
    prices : array-like of float
        Candidate prices to test, shape (n_prices,)
    T : int
        Number of time periods to run simulation
    sigma : float
        Standard deviation of noise in demand observations
    demand_function : callable
        Demand function to query for observations
    demand_params : dict
        Parameters to pass to demand_function
    **kwargs : dict
        Optional algorithm hyperparameters:
        - prior_mean : list of float, default [0, -1]
            Prior mean for demand parameters [intercept, slope]
        - prior_cov_scale : float, default 10.0
            Scale factor for prior covariance (larger = more uncertainty)
        - return_details : bool, default False
            If True, return detailed behavior data

    Returns
    -------
    revenues : list of float
        Revenue achieved at each time step, length T
    OR (if return_details=True):
    dict with keys:
        - revenues: list of float
        - chosen_prices: list of float (prices chosen at each step)
        - posterior_variance: list of float (trace of posterior covariance)
        - param_estimates: list of [a, b] (posterior mean at each step)

    Notes
    -----
    **Prior**: Gaussian with configurable mean and covariance
        - Default mean [0, -1] assumes demand decreases with price
        - Prior covariance = prior_cov_scale * I
        - Larger prior_cov_scale → more exploration initially

    **Posterior Update**: Uses precision form of Bayesian linear regression
        - Posterior precision = Prior precision + Data precision
        - Closed-form update (no MCMC needed)

    **Limitations**:
        - Assumes linear demand (D = a + b*P + noise)
        - Performance degrades for highly nonlinear demand
        - Prior tuning can significantly impact early performance

    References
    ----------
    .. [1] Agrawal, Shipra, and Navin Goyal. "Thompson sampling for
           contextual bandits with linear payoffs." ICML 2013.
    """
    # Extract algorithm hyperparameters from kwargs
    prior_cov_scale = kwargs.get('prior_cov_scale', 10.0)  # Prior covariance scale
    noise_variance = sigma**2
    return_details = kwargs.get('return_details', False)

    # Auto-initialize prior from demand parameters if not explicitly provided
    # This gives the algorithm a much better starting point
    explicit_prior = kwargs.get('prior_mean')
    if explicit_prior is not None:
        prior_mean = np.array(explicit_prior)
    else:
        # Try to extract (a, b) from demand_params for linear-ish demand
        a_init = demand_params.get('a', 0)
        b_init = demand_params.get('b', -1)
        prior_mean = np.array([float(a_init), float(b_init)])

    revenues = []
    chosen_prices = []
    posterior_variance = []
    param_estimates = []

    # Initialize posterior = prior
    # We use precision form for sequential updates (more numerically stable)
    posterior_mean = prior_mean.copy()
    posterior_precision = np.eye(2) / prior_cov_scale
    posterior_cov = np.eye(2) * prior_cov_scale

    for t in range(T):
        # Sample from the current posterior
        sampled_params = np.random.multivariate_normal(posterior_mean, posterior_cov)
        a_est, b_est = sampled_params

        # Choose price maximizing expected revenue
        expected_revenues = [p * max(a_est + b_est * p, 0) for p in prices]
        chosen_price = prices[np.argmax(expected_revenues)]
        chosen_prices.append(chosen_price)

        # Observe demand
        observed_demand = demand_function_wrapper(chosen_price, sigma, demand_function, time_step=t, **demand_params)
        revenues.append(revenue(chosen_price, observed_demand))

        # Sequential Bayesian linear regression update
        # Update with just the NEW observation (not all observations)
        # This is mathematically equivalent to batch update but cleaner
        #
        # Feature vector for this observation: x = [1, price]
        # Observation: y = demand
        #
        # Update rules:
        #   posterior_precision_new = posterior_precision_old + x x' / σ²
        #   posterior_mean_new = posterior_cov_new @ (posterior_precision_old @ posterior_mean_old + x y / σ²)
        x = np.array([1.0, chosen_price])
        y = observed_demand

        # Update precision (add information from new observation)
        posterior_precision_new = posterior_precision + np.outer(x, x) / noise_variance

        # Update covariance (invert precision)
        posterior_cov_new = np.linalg.inv(posterior_precision_new)

        # Update mean
        posterior_mean_new = posterior_cov_new @ (
            posterior_precision @ posterior_mean + x * y / noise_variance
        )

        # Store for next iteration
        posterior_mean = posterior_mean_new
        posterior_precision = posterior_precision_new
        posterior_cov = posterior_cov_new

        # Track state for detailed output
        posterior_variance.append(np.trace(posterior_cov))
        param_estimates.append(posterior_mean.tolist())

    if return_details:
        return {
            'revenues': revenues,
            'chosen_prices': chosen_prices,
            'posterior_variance': posterior_variance,
            'param_estimates': param_estimates
        }
    return revenues

def epsilon_greedy_price_policy(prices, T, sigma, demand_function, demand_params, **kwargs):
    """
    Epsilon-Greedy pricing policy with fixed exploration rate.

    Simple exploration-exploitation strategy: with probability ε, explore by
    choosing a random price; otherwise exploit by choosing the price with
    best historical performance (highest price × mean_demand).

    Parameters
    ----------
    prices : array-like of float
        Candidate prices to test, shape (n_prices,)
    T : int
        Number of time periods to run simulation
    sigma : float
        Standard deviation of noise in demand observations
    demand_function : callable
        Demand function to query for observations
    demand_params : dict
        Parameters to pass to demand_function
    **kwargs : dict
        Optional algorithm hyperparameters:
        - epsilon : float, default 0.1
            Exploration rate (probability of random price selection)
        - return_details : bool, default False
            If True, return detailed behavior data

    Returns
    -------
    revenues : list of float
        Revenue achieved at each time step, length T
    OR (if return_details=True):
    dict with keys:
        - revenues: list of float
        - chosen_prices: list of float
        - is_exploration: list of bool (True=explored, False=exploited)
        - price_counts: dict mapping price -> count of times chosen

    Notes
    -----
    **Exploration Rate**: Configurable ε (default 0.1 = 10% random exploration)
        - Higher ε → more exploration, slower convergence
        - Lower ε → more exploitation, faster convergence but risk local optima

    **Exploitation Strategy**: Choose price maximizing price × mean(observed demands)
        - Maintains running average of demand for each tested price
        - Untested prices are never selected in exploit phase

    **Limitations**:
        - Fixed ε may be suboptimal (should decrease over time)
        - Prices that are unlucky early on may never be retried
        - No confidence intervals (unlike Thompson Sampling)

    **When to Use**:
        - Need simple, interpretable algorithm
        - Want predictable exploration behavior
        - Computational efficiency is critical
    """
    revenues = []
    chosen_prices = []
    is_exploration = []
    X, Y = [], []  # Observations
    # Extract algorithm hyperparameters from kwargs
    epsilon = kwargs.get('epsilon', 0.1)  # Exploration probability
    return_details = kwargs.get('return_details', False)
    price_demand_mapping = {p: [] for p in prices}
    for t in range(T):
        exploring = (np.random.rand() < epsilon) or (t == 0)
        if exploring:
            # Explore: Choose a random price
            greedy_price = np.random.choice(prices)
        else:
            # Calculate mean demand for each price, then find price maximizing expected revenue
            mean_demands = {p: np.mean(demands) for p, demands in price_demand_mapping.items() if demands}
            # Choose the price with the maximum expected revenue (price × mean_demand)
            greedy_price = max(mean_demands.keys(), key=lambda p: p * mean_demands[p])

        chosen_prices.append(greedy_price)
        is_exploration.append(exploring)

        # Observe demand
        observed_demand = demand_function_wrapper(greedy_price, sigma,demand_function ,time_step=t, **demand_params)
        revenues.append(revenue(greedy_price, observed_demand))

        # Update estimates
        X.append(greedy_price)
        Y.append(observed_demand)
        price_demand_mapping[greedy_price].append(observed_demand)

    if return_details:
        price_counts = {p: len(demands) for p, demands in price_demand_mapping.items()}
        return {
            'revenues': revenues,
            'chosen_prices': chosen_prices,
            'is_exploration': is_exploration,
            'price_counts': price_counts
        }
    return revenues


def uniform_pricing(price, T, sigma, demand_function, demand_params, **kwargs):
    """
    Uniform (fixed) pricing policy - baseline strategy.

    Maintains a single fixed price across all time periods. Useful as a
    simple baseline to compare against adaptive algorithms.

    Parameters
    ----------
    price : float
        Fixed price to use for all time periods
    T : int
        Number of time periods
    sigma : float
        Standard deviation of noise in demand observations
    demand_function : callable
        Demand function to query for observations
    demand_params : dict
        Parameters to pass to demand_function
    **kwargs : dict
        Additional arguments (unused)

    Returns
    -------
    revenues : list of float
        Revenue achieved at each time step, length T

    Notes
    -----
    **No Learning**: Price never changes, regardless of observations

    **Use Case**: Represents real-world scenario where price is fixed
    (e.g., due to contracts, regulations, or operational constraints)

    **Performance**: Optimal only if the fixed price happens to be the
    true revenue-maximizing price. Otherwise, adaptive algorithms will
    outperform over time.
    """
    return [revenue(price, demand_function_wrapper(price, sigma,demand_function ,time_step=t, **demand_params)) for t in range(T)]


def online_learning(prices, T, gamma, sigma, demand_function, demand_params, **kwargs):
    """
    Online learning pricing policy with exponential discounting.

    Estimates linear demand parameters (a, b) using weighted least squares
    where recent observations receive higher weight. The discount factor γ
    controls how quickly old observations are forgotten, enabling adaptation
    to non-stationary demand.

    Parameters
    ----------
    prices : array-like of float
        Candidate prices to test, shape (n_prices,)
    T : int
        Number of time periods
    gamma : float
        Discount factor ∈ (0, 1). Typical value: 0.99
            - γ close to 1: slow forgetting (suitable for stationary demand)
            - γ closer to 0: fast forgetting (suitable for non-stationary demand)
    sigma : float
        Standard deviation of noise in demand observations
    demand_function : callable
        Demand function to query for observations
    demand_params : dict
        Parameters to pass to demand_function
    **kwargs : dict
        Additional arguments:
        - return_details : bool, default False
            If True, return detailed behavior data

    Returns
    -------
    revenues : list of float
        Revenue achieved at each time step, length T
    OR (if return_details=True):
    dict with keys:
        - revenues: list of float
        - chosen_prices: list of float
        - param_estimates: list of [a, b] (demand parameter estimates)

    Notes
    -----
    **Weighting Scheme**: Observation at time t gets weight γ^(n-t-1)
        - Most recent observation: weight = 1.0
        - Observation from n steps ago: weight = γ^n (exponentially smaller)

    **Weighted Least Squares**: Solves (X^T W X)β = X^T W y
        - W = diagonal matrix of weights
        - β = [a, b] are demand parameters
        - Assumes demand ≈ a + b*price

    **Error Handling**: If matrix singular (rare), falls back to lstsq
        - Prevents crashes from numerical issues
        - Maintains previous estimates if all methods fail

    **When to Use**:
        - Non-stationary demand (market conditions changing)
        - Want explicit control over forgetting rate
        - Need point estimates (not full posterior like Thompson Sampling)

    References
    ----------
    .. [1] Sutton & Barto (2018). Reinforcement Learning: An Introduction.
           Chapter on non-stationary problems.
    """
    a_est, b_est = 0, -1  # Initial estimates
    X, Y = [], []
    revenues = []
    chosen_prices = []
    param_estimates = []
    return_details = kwargs.get('return_details', False)

    for t in range(T):
        # Choose price maximizing expected revenue
        expected_revenues = [p * max(a_est + b_est * p, 0) for p in prices]
        chosen_price = prices[np.argmax(expected_revenues)]
        chosen_prices.append(chosen_price)

        # Observe demand
        observed_demand = demand_function_wrapper(chosen_price, sigma,demand_function ,time_step=t, **demand_params)
        revenues.append(revenue(chosen_price, observed_demand))

        # Update estimates
        X.append(chosen_price)
        Y.append(observed_demand)
        weights = np.array([gamma**(len(X) - i - 1) for i in range(len(X))])
        X_design = np.vstack((np.ones(len(X)), X)).T
        W = np.diag(weights)

        try:
            # Weighted least squares using matrix inversion
            beta = np.linalg.inv(X_design.T @ W @ X_design) @ (X_design.T @ W @ Y)
            a_est, b_est = beta[0], beta[1]
        except np.linalg.LinAlgError:
            # Matrix is singular - fall back to least squares
            try:
                result = np.linalg.lstsq(X_design.T @ W @ X_design, X_design.T @ W @ Y, rcond=None)
                beta = result[0]
                if len(beta) >= 2:
                    a_est, b_est = beta[0], beta[1]
                # else: keep previous estimates (a_est, b_est unchanged)
            except:
                # If least squares also fails, keep previous estimates
                pass

        param_estimates.append([a_est, b_est])

    if return_details:
        return {
            'revenues': revenues,
            'chosen_prices': chosen_prices,
            'param_estimates': param_estimates
        }
    return revenues


def simulate_pricing(prices, T, gamma,sigma, demand_params, demand_type, policy_function, **kwargs):
    """
    Simulates a pricing policy over T time steps using the specified demand type.

    Args:
        prices: List of possible prices.
        T: Number of time steps.
        sigma: Noise level in demand.
        demand_params: Dictionary of parameters required for the selected demand function.
        demand_type: String indicating the type of demand function to use.
        policy_function: The pricing policy function to simulate (e.g., Thompson Sampling).
        kwargs: Additional arguments for the policy function.

    Returns:
         revenues over T time steps.
    """
    # Select the demand function
    demand_function = DEMAND_TYPE_MAPPING[demand_type]
    
    
    # Inspect the function's arguments
    func_params = inspect.signature(policy_function).parameters

    # Call the function based on whether it requires `time_step`
    if 'gamma' in func_params:
        # Run the selected policy
        out = policy_function(prices, T,gamma, sigma, demand_function, demand_params, **kwargs)
    else:
        # Run the selected policy
        out = policy_function(prices, T, sigma, demand_function, demand_params, **kwargs)
    
    # Standardize return type - always return list of revenues
    # Some policies (like foresight) return tuple (revenues, chosen_prices)
    if isinstance(out, tuple):
        revenues, chosen_prices = out
        return revenues  # Return only revenues for consistency
    else:
        return out  # Already a list of revenues

def comparison_report(prices, T, sigma, demand_params_for_type, DEMAND_TYPE, num_simulations, algo_params=None):
    """
    Run comparison of all pricing algorithms with configurable hyperparameters.

    Parameters
    ----------
    prices : array-like
        Candidate prices to test
    T : int
        Number of time steps
    sigma : float
        Demand noise standard deviation
    demand_params_for_type : dict
        Parameters for the demand function
    DEMAND_TYPE : str
        Type of demand function
    num_simulations : int
        Number of Monte Carlo simulation runs
    algo_params : dict, optional
        Algorithm hyperparameters with keys:
        - 'ts_prior_mean': [a, b] for Thompson Sampling prior
        - 'ts_prior_cov_scale': float for Thompson Sampling prior covariance
        - 'epsilon': float for Epsilon-Greedy exploration rate
        - 'gamma': float for Online Learning discount factor
        - 'uniform_price': float for Uniform Pricing fixed price (default: median of prices)

    Returns
    -------
    results : dict
        Dictionary with algorithm names as keys, containing:
        - 'mean': mean total revenue
        - 'cumulative': mean cumulative revenue over time
        - 'regret': cumulative regret vs oracle (if applicable)
        - 'all_revenues': list of cumulative revenues for all simulation runs
    best_price : float
        Average optimal price from oracle simulations
    """
    # Default algorithm parameters if not provided
    if algo_params is None:
        algo_params = {
            'ts_prior_mean': [0.0, -1.0],
            'ts_prior_cov_scale': 10.0,
            'epsilon': 0.1,
            'gamma': 0.99
        }

    # Extract algorithm parameters
    gamma = algo_params.get('gamma', 0.99)
    epsilon = algo_params.get('epsilon', 0.1)
    ts_prior_mean = algo_params.get('ts_prior_mean')  # None means auto-initialize from demand_params
    ts_prior_cov_scale = algo_params.get('ts_prior_cov_scale', 10.0)
    uniform_price = algo_params.get('uniform_price', prices[len(prices)//2])  # Default: median price

    # Compare methods
    # Compute the cumulative revenues and regrets for each method

    # Run foresight simulations - foresight returns (revenues, chosen_prices) tuple
    # Handle it specially to extract both revenues and prices
    demand_function = DEMAND_TYPE_MAPPING[DEMAND_TYPE]
    foresight_results_raw = [
        foresight(prices, T, sigma, demand_function, demand_params_for_type)
        for _ in range(num_simulations)
    ]

    # Extract revenues and prices separately
    foresight_revenues_list = np.array([result[0] for result in foresight_results_raw])  # Revenues
    foresight_prices_list = [result[1] for result in foresight_results_raw]  # Chosen prices

    # Extract best price for each simulation (the most frequently chosen optimal price)
    best_prices_per_simulation = []
    for revenues, chosen_prices in foresight_results_raw:
        # Find the index with maximum revenue
        best_idx = np.argmax(revenues)
        best_prices_per_simulation.append(chosen_prices[best_idx])

    # Calculate the average best price across all simulations
    average_best_price = np.mean(best_prices_per_simulation)

    # Run ONE detailed example simulation for each algorithm (for behavior visualization)
    example_runs = {}

    # Thompson Sampling detailed run
    ts_example = thompson_sampling(
        prices, T, sigma, demand_function, demand_params_for_type,
        prior_mean=ts_prior_mean, prior_cov_scale=ts_prior_cov_scale, return_details=True
    )
    example_runs['Thompson Sampling'] = ts_example

    # Epsilon-Greedy detailed run
    eg_example = epsilon_greedy_price_policy(
        prices, T, sigma, demand_function, demand_params_for_type,
        epsilon=epsilon, return_details=True
    )
    example_runs['Epsilon-Greedy'] = eg_example

    # Online Learning detailed run
    ol_example = online_learning(
        prices, T, gamma, sigma, demand_function, demand_params_for_type,
        return_details=True
    )
    example_runs['Online Learning'] = ol_example

    # Oracle example (uses first simulation's prices)
    example_runs['Foresight (Oracle)'] = {
        'chosen_prices': foresight_prices_list[0]
    }

    ts_revenues_list = np.array([
        simulate_pricing(prices, T, gamma, sigma, demand_params_for_type, DEMAND_TYPE, thompson_sampling,
                        prior_mean=ts_prior_mean, prior_cov_scale=ts_prior_cov_scale)
        for _ in range(num_simulations)
    ])

    greedy_revenues_list = np.array([
        simulate_pricing(prices, T, gamma, sigma, demand_params_for_type, DEMAND_TYPE, epsilon_greedy_price_policy,
                        epsilon=epsilon)
        for _ in range(num_simulations)
    ])

    # Uniform pricing uses a user-defined fixed price (default: median of price array)
    uniform_revenues_list = np.array([
        simulate_pricing(uniform_price, T, gamma, sigma, demand_params_for_type, DEMAND_TYPE, uniform_pricing)
        for _ in range(num_simulations)
    ])

    online_revenues_list = np.array([
        simulate_pricing(prices, T, gamma,sigma, demand_params_for_type, DEMAND_TYPE, online_learning)
        for _ in range(num_simulations)
    ])
    
    
    # Calculate mean revenues
    foresight_revenues = np.mean(np.sum(foresight_revenues_list, axis=1))
    ts_revenues = np.mean(np.sum(ts_revenues_list, axis=1))
    greedy_revenues = np.mean(np.sum(greedy_revenues_list, axis=1))
    uniform_revenues = np.mean(np.sum(uniform_revenues_list, axis=1))
    online_revenues = np.mean(np.sum(online_revenues_list, axis=1))
    
    # Calculate mean differences at each time step
    foresight_cumulative_mean = np.mean(np.cumsum(foresight_revenues_list, axis=1), axis=0)
    ts_cumulative_mean = np.mean(np.cumsum(ts_revenues_list, axis=1), axis=0)
    greedy_cumulative_mean = np.mean(np.cumsum(greedy_revenues_list, axis=1), axis=0)
    uniform_cumulative_mean = np.mean(np.cumsum(uniform_revenues_list, axis=1), axis=0)
    online_cumulative_mean = np.mean(np.cumsum(online_revenues_list, axis=1), axis=0)
    

    
    # Regrets for Thompson Sampling and Online Learning
    ts_regret = foresight_cumulative_mean - ts_cumulative_mean
    online_regret = foresight_cumulative_mean - online_cumulative_mean
    greedy_regret = foresight_cumulative_mean - greedy_cumulative_mean
    
    results = {
        "Foresight (Oracle)": {
            'mean': foresight_revenues,
            'cumulative': foresight_cumulative_mean,
            'all_revenues': np.cumsum(foresight_revenues_list, axis=1).tolist()
        },
        "Thompson Sampling": {
            'mean': ts_revenues,
            "cumulative": ts_cumulative_mean,
            'regret': ts_regret,
            'all_revenues': np.cumsum(ts_revenues_list, axis=1).tolist()
        },
        "Epsilon-Greedy": {
            'mean': greedy_revenues,
            "cumulative": greedy_cumulative_mean,
            'regret': greedy_regret,
            'all_revenues': np.cumsum(greedy_revenues_list, axis=1).tolist()
        },
        "Uniform Pricing": {
            'mean': uniform_revenues,
            "cumulative": uniform_cumulative_mean,
            'all_revenues': np.cumsum(uniform_revenues_list, axis=1).tolist()
        },
        "Online Learning": {
            'mean': online_revenues,
            "cumulative": online_cumulative_mean,
            'regret': online_regret,
            'all_revenues': np.cumsum(online_revenues_list, axis=1).tolist()
        },
    }

    # Add example runs for behavior visualization
    results['_example_runs'] = example_runs
    results['_simulation_config'] = {
        'demand_type': DEMAND_TYPE,
        'demand_params': demand_params_for_type,
        'T': T,
        'num_simulations': num_simulations,
        'price_range': [float(prices[0]), float(prices[-1])],
        'num_prices': len(prices),
        'sigma': sigma,
        'algo_params': algo_params
    }

    return results, average_best_price

def comparison_plots(results):
    T = results['Online Learning']['regret'].shape[0]
    # Create cumulative regret plot (fig1)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=list(range(T)), 
        y=results['Thompson Sampling']['regret'], 
        mode='lines', 
        name="Thompson Sampling Regret"
    ))
    fig1.add_trace(go.Scatter(
        x=list(range(T)), 
        y=results['Online Learning']['regret'], 
        mode='lines', 
        name="Online Learning Regret", 
        line=dict(dash="dash")
    ))
    fig1.add_trace(go.Scatter(
        x=list(range(T)), 
        y=results['Epsilon-Greedy']['regret'], 
        mode='lines', 
        name="Epsilon-Greedy Regret", 
        line=dict(dash="dot")
    ))
    fig1.update_layout(
        title="Cumulative Regrets",
        xaxis_title="Time Step",
        yaxis_title="Cumulative Regret",
        legend_title="Algorithm",
        template="plotly_white"
    )
    
    # Create cumulative revenue plot (fig2)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=list(range(T)), 
        y=results['Foresight (Oracle)']['cumulative'], 
        mode='lines', 
        name="Foresight (Oracle)", 
        line=dict(dash="dash")
    ))
    fig2.add_trace(go.Scatter(
        x=list(range(T)), 
        y=results['Thompson Sampling']['cumulative'], 
        mode='lines', 
        name="Thompson Sampling"
    ))
    fig2.add_trace(go.Scatter(
        x=list(range(T)), 
        y=results['Uniform Pricing']['cumulative'], 
        mode='lines', 
        name="Uniform Pricing"
    ))
    fig2.add_trace(go.Scatter(
        x=list(range(T)), 
        y=results['Epsilon-Greedy']['cumulative'], 
        mode='lines', 
        name="Epsilon-Greedy Pricing"
    ))
    fig2.add_trace(go.Scatter(
        x=list(range(T)), 
        y=results['Online Learning']['cumulative'], 
        mode='lines', 
        name="Online Learning"
    ))
    fig2.update_layout(
        title="Cumulative Revenue Comparison",
        xaxis_title="Time Step",
        yaxis_title="Cumulative Revenue",
        legend_title="Algorithm",
        template="plotly_white"
    )
    
    return fig1, fig2

def comparison_plots_mat(results):
    T = results['Online Learning']['regret'].shape[0]
    # Plot cumulative regret (fig1)
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(range(T), results['Thompson Sampling']['regret'], label="Thompson Sampling Regret")
    ax1.plot(range(T), results['Online Learning']['regret'], label="Online Learning Regret", linestyle="--")
    ax1.plot(range(T), results['Epsilon-Greedy']['regret'], label="Epsilon-Greedy Regret", linestyle="dotted")
    ax1.set_title("Cumulative Regrets")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Cumulative Regret")
    ax1.legend()
    ax1.grid(True)

    # Plot cumulative revenues (fig2)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(range(T), results['Foresight (Oracle)']['cumulative'], label="Foresight (Oracle)", linestyle="--")
    ax2.plot(range(T), results['Thompson Sampling']['cumulative'], label="Thompson Sampling")
    ax2.plot(range(T), results['Uniform Pricing']['cumulative'], label="Uniform Pricing")
    ax2.plot(range(T), results['Epsilon-Greedy']['cumulative'], label="Epsilon-Greedy Pricing")
    ax2.plot(range(T), results['Online Learning']['cumulative'], label="Online Learning")
    ax2.set_title("Cumulative Revenue Comparison")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Cumulative Revenue")
    ax2.legend()
    ax2.grid(True)
    
    return fig1, fig2




if __name__ == "__main__":
    
    # Example usage
    DEMAND_TYPE = "linear"  # Change to "linear", "quadratic", "seasonal", etc.
    
    # Demand-specific parameters
    demand_params = {
        "linear": {"a": 100, "b": -2},
        "quadratic": {"a": 50, "b": -1, "c": 0.01},
        "seasonal": {"a": 70, "b": -0.5, "period": 30},
        "shifted": {"a1": 60, "b1": -0.3, "a2": 40, "b2": -0.2, "shift_time": 50},
        "stepwise": {"thresholds": [20, 50, 80], "demands": [100, 50, 20, 5]},
        "combined": {
            "segments": [
                {"type": "quadratic", "params": {"a": 50, "b": -1, "c": 0.01}},
                {"type": "seasonal", "params": {"a": 70, "b": -0.5, "period": 30}},
                {"type": "shifted", "params": {"a1": 60, "b1": -0.3, "a2": 40, "b2": -0.2, "shift_time": 150}}
            ],
            "time_thresholds": [25, 50]
        },
        "csv": {}
    }
    
    
    if DEMAND_TYPE == "csv":
        fitted_demand_function = read_and_fit_demand("data/dummy_demand_data.csv")
        DEMAND_TYPE_MAPPING['csv'] = lambda p, sigma, **kwargs: demand_from_csv(p, sigma, fitted_demand_function)

    
    
    # Parameters
    demand_params_for_type = demand_params[DEMAND_TYPE]
    T = 100  # Number of time steps
    prices = np.linspace(10, 100, 50)  # Price range
    sigma = 5  # Noise level in demand
    gamma = 0.99  # Discount factor for online learning
    num_simulations = 100  # Number of simulations for comparison
    
    
    
    results, best_price = comparison_report(prices, T, sigma, demand_params_for_type, DEMAND_TYPE, num_simulations)
    
    fig1, fig2 = comparison_plots_mat(results)

    # Display the figures
    fig1.show()
    fig2.show()
    
    
    #Plot demand
    plot_demand(DEMAND_TYPE_MAPPING, DEMAND_TYPE,
                    prices,sigma, demand_params_for_type,
                    T,
                    num_simulations=50)
    
    # Print final cumulative revenues
    print("Final Revenues:")
    print(f"  Foresight (Oracle): {results['Foresight (Oracle)']['mean']:.2f}")
    print(f"  Thompson Sampling: {results['Thompson Sampling']['mean']:.2f}")
    print(f"  Epsilon-Greedy: {results['Epsilon-Greedy']['mean']:.2f}")
    print(f"  Uniform Pricing: {results['Uniform Pricing']['mean']:.2f}")
    print(f"  Online Learning: {results['Online Learning']['mean']:.2f}")
