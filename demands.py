#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 2024

@author: lefteris
"""
# =============================================================================
# Explanation of Demand Scenarios
#
#     Linear Demand:
#         The default scenario with a linear relation between price and demand.
#         Parameters: a (intercept), b (slope, typically negative).
#
#     Quadratic Demand:
#         Models scenarios where demand has a nonlinear relationship with price.
#         Can model situations where demand decreases sharply then levels off.
#         Parameters: a, b, c (quadratic coefficient).
#
#     Seasonal Demand:
#         Adds a periodic sinusoidal component to represent seasonal variations
#         (e.g., holiday shopping, summer sales).
#         Parameters: a, b, period.
#
#     Shifted Demand:
#         Models changes in market conditions at a specific point in time,
#         such as new competitors or supply chain disruptions.
#         Uses different demand coefficients before and after a shift.
#         Parameters: a1, b1 (initial demand), a2, b2 (post-shift demand), shift_time.
#
#     Stepwise Demand:
#         Models discrete price-point effects where demand changes at thresholds.
#         Parameters: thresholds (list), demands (list).
#
#     Combined Demand:
#         Combines multiple demand types for longer simulations,
#         switching between scenarios over time.
#
# =============================================================================
        
        
        
import numpy as np
import inspect
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import plotly.graph_objects as go

def plot_demand_and_revenue_streamlit(DEMAND_TYPE_MAPPING, DEMAND_TYPE,
                prices, sigma, demand_params_for_type,
                T,
                num_simulations=100):
    """
    Plot both demand curve and revenue curve side by side.
    Revenue = Price × Demand. The optimal price is where revenue is maximized.

    Returns:
        tuple: (demand_fig, revenue_fig) - two Plotly figures
    """
    from plotly.subplots import make_subplots

    demand_function = DEMAND_TYPE_MAPPING[DEMAND_TYPE]
    func_params = inspect.signature(demand_function).parameters

    if not ('time_step' in func_params):
        # Simulate multiple demand curves
        simulated_demands = np.array([
            [demand_function_wrapper(p, sigma, demand_function, **demand_params_for_type) for p in prices]
            for _ in range(num_simulations)
        ])

        # Calculate average demand and confidence intervals
        mean_demand = simulated_demands.mean(axis=0)
        std_demand = simulated_demands.std(axis=0)
        lower_bound_demand = mean_demand - 1.96 * std_demand / np.sqrt(num_simulations)
        upper_bound_demand = mean_demand + 1.96 * std_demand / np.sqrt(num_simulations)

        # Calculate revenue = price × demand
        simulated_revenues = simulated_demands * prices  # Broadcasting
        mean_revenue = simulated_revenues.mean(axis=0)
        std_revenue = simulated_revenues.std(axis=0)
        lower_bound_revenue = mean_revenue - 1.96 * std_revenue / np.sqrt(num_simulations)
        upper_bound_revenue = mean_revenue + 1.96 * std_revenue / np.sqrt(num_simulations)

        # Find optimal price (where revenue is maximized)
        optimal_idx = np.argmax(mean_revenue)
        optimal_price = prices[optimal_idx]
        optimal_revenue = mean_revenue[optimal_idx]

        # Create demand figure
        demand_fig = go.Figure()
        demand_fig.add_trace(go.Scatter(
            x=prices, y=mean_demand,
            mode='lines', name="Average Demand",
            line=dict(color='blue')
        ))
        demand_fig.add_trace(go.Scatter(
            x=np.concatenate([prices, prices[::-1]]),
            y=np.concatenate([upper_bound_demand, lower_bound_demand[::-1]]),
            fill='toself', fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name="95% CI"
        ))
        demand_fig.update_layout(
            title=f"Demand Curve ({DEMAND_TYPE})",
            xaxis_title="Price (€)",
            yaxis_title="Demand (units)",
            legend=dict(x=0.7, y=0.95),
            template="plotly_white",
            height=350
        )

        # Create revenue figure
        revenue_fig = go.Figure()
        revenue_fig.add_trace(go.Scatter(
            x=prices, y=mean_revenue,
            mode='lines', name="Average Revenue",
            line=dict(color='green')
        ))
        revenue_fig.add_trace(go.Scatter(
            x=np.concatenate([prices, prices[::-1]]),
            y=np.concatenate([upper_bound_revenue, lower_bound_revenue[::-1]]),
            fill='toself', fillcolor='rgba(0, 128, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name="95% CI"
        ))
        # Mark optimal price
        revenue_fig.add_trace(go.Scatter(
            x=[optimal_price], y=[optimal_revenue],
            mode='markers+text',
            marker=dict(color='red', size=12, symbol='star'),
            text=[f"Optimal: {optimal_price:.1f}€"],
            textposition="top center",
            name=f"Optimal Price"
        ))
        revenue_fig.update_layout(
            title="Revenue Curve (Price × Demand)",
            xaxis_title="Price (€)",
            yaxis_title="Revenue (€)",
            legend=dict(x=0.7, y=0.95),
            template="plotly_white",
            height=350
        )

        return demand_fig, revenue_fig, optimal_price

    else:
        # Time-dependent demand
        time_steps = np.sort(np.random.choice(range(T), size=5, replace=False))

        time_step_demands = []
        for t in time_steps:
            simulated_demands_t = np.array([
                [demand_function_wrapper(p, sigma, demand_function, time_step=t, **demand_params_for_type) for p in prices]
                for _ in range(num_simulations)
            ])
            time_step_demands.append(simulated_demands_t)

        mean_demand_time = [demands.mean(axis=0) for demands in time_step_demands]
        std_demand_time = [demands.std(axis=0) for demands in time_step_demands]

        # Calculate revenues for each time step
        mean_revenue_time = [mean * prices for mean in mean_demand_time]

        # Find optimal prices for each time step
        optimal_prices = [prices[np.argmax(rev)] for rev in mean_revenue_time]

        # Demand figure
        demand_fig = go.Figure()
        for t, mean in zip(time_steps, mean_demand_time):
            demand_fig.add_trace(go.Scatter(
                x=prices, y=mean,
                mode='lines', name=f"t={t}"
            ))
        demand_fig.update_layout(
            title=f"Time-Dependent Demand ({DEMAND_TYPE})",
            xaxis_title="Price (€)",
            yaxis_title="Demand (units)",
            legend=dict(x=0.8, y=0.95),
            template="plotly_white",
            height=350
        )

        # Revenue figure
        revenue_fig = go.Figure()
        for t, rev, opt_p in zip(time_steps, mean_revenue_time, optimal_prices):
            revenue_fig.add_trace(go.Scatter(
                x=prices, y=rev,
                mode='lines', name=f"t={t}"
            ))
        # Mark optimal prices
        for t, rev, opt_p in zip(time_steps, mean_revenue_time, optimal_prices):
            opt_idx = np.argmax(rev)
            revenue_fig.add_trace(go.Scatter(
                x=[opt_p], y=[rev[opt_idx]],
                mode='markers',
                marker=dict(size=8, symbol='star'),
                showlegend=False
            ))
        revenue_fig.update_layout(
            title="Time-Dependent Revenue (Price × Demand)",
            xaxis_title="Price (€)",
            yaxis_title="Revenue (€)",
            legend=dict(x=0.8, y=0.95),
            template="plotly_white",
            height=350
        )

        avg_optimal = np.mean(optimal_prices)
        return demand_fig, revenue_fig, avg_optimal


def plot_demand_streamlit(DEMAND_TYPE_MAPPING, DEMAND_TYPE,
                prices,sigma, demand_params_for_type,
                T,
                num_simulations=100):
    #PLOT DEMANDS
    demand_function = DEMAND_TYPE_MAPPING[DEMAND_TYPE]

    # Inspect the function's arguments
    func_params = inspect.signature(demand_function).parameters

    # Call the function based on whether it requires `time_step`
    if not ('time_step' in func_params):
        # Simulate multiple demand curves
        simulated_demands = np.array([
            [demand_function_wrapper(p, sigma, demand_function, **demand_params_for_type) for p in prices]
            for _ in range(num_simulations)
        ])

        # Calculate average and confidence intervals
        mean_demand = simulated_demands.mean(axis=0)
        std_demand = simulated_demands.std(axis=0)
        lower_bound = mean_demand - 1.96 * std_demand / np.sqrt(num_simulations)
        upper_bound = mean_demand + 1.96 * std_demand / np.sqrt(num_simulations)

        fig = go.Figure()

        # Add average demand curve
        fig.add_trace(go.Scatter(
            x=prices,
            y=mean_demand,
            mode='lines',
            name="Average Demand",
            line=dict(color='blue')
        ))

        # Add confidence interval as a shaded area
        fig.add_trace(go.Scatter(
            x=np.concatenate([prices, prices[::-1]]),
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name="95% CI"
        ))

        # Update layout
        fig.update_layout(
            title=f"Demand Curve with Confidence Intervals ({DEMAND_TYPE})",
            xaxis_title="Price",
            yaxis_title="Demand",
            legend=dict(x=0.1, y=0.9),
            template="plotly_white",
        )
    else:
        # Parameters
        time_steps = np.sort(np.random.choice(range(T), size=5, replace=False))  # Range of time steps

        # Simulate demands for each time step
        time_step_demands = []
        for t in time_steps:
            simulated_demands_t = np.array([
                [demand_function_wrapper(p, sigma, demand_function, time_step=t, **demand_params_for_type) for p in prices]
                for _ in range(num_simulations)
            ])
            time_step_demands.append(simulated_demands_t)

        # Calculate mean and confidence intervals for each time step
        mean_demand_time = [demands.mean(axis=0) for demands in time_step_demands]
        std_demand_time = [demands.std(axis=0) for demands in time_step_demands]
        lower_bound_time = [mean - 1.96 * std / np.sqrt(num_simulations) for mean, std in zip(mean_demand_time, std_demand_time)]
        upper_bound_time = [mean + 1.96 * std / np.sqrt(num_simulations) for mean, std in zip(mean_demand_time, std_demand_time)]

        fig = go.Figure()

        # Add a curve for each time step
        for t, mean, lower, upper in zip(time_steps, mean_demand_time, lower_bound_time, upper_bound_time):
            fig.add_trace(go.Scatter(
                x=prices,
                y=mean,
                mode='lines',
                name=f"Time Step {t}",
            ))

            fig.add_trace(go.Scatter(
                x=np.concatenate([prices, prices[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill='toself',
                fillcolor='rgba(0,0,255,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False
            ))

        # Update layout
        fig.update_layout(
            title=f"Time-Dependent Demand Curves with Confidence Intervals ({DEMAND_TYPE})",
            xaxis_title="Price",
            yaxis_title="Demand",
            legend=dict(x=1.05, y=1.05),
            template="plotly_white",
        )

    return fig


def plot_demand(DEMAND_TYPE_MAPPING, DEMAND_TYPE,
                prices,sigma, demand_params_for_type,
                T,
                num_simulations=100):
    #PLOT DEMANDS
    demand_function = DEMAND_TYPE_MAPPING[DEMAND_TYPE]
    
    # Inspect the function's arguments
    func_params = inspect.signature(demand_function).parameters

    # Call the function based on whether it requires `time_step`
    if not ('time_step' in func_params):
        # Simulate multiple demand curves
        simulated_demands = np.array([
            [demand_function_wrapper(p, sigma, demand_function, **demand_params_for_type) for p in prices]
            for _ in range(num_simulations)
        ])
    
        # Calculate average and confidence intervals
        mean_demand = simulated_demands.mean(axis=0)
        std_demand = simulated_demands.std(axis=0)
        lower_bound = mean_demand - 1.96 * std_demand / np.sqrt(num_simulations)
        upper_bound = mean_demand + 1.96 * std_demand / np.sqrt(num_simulations)
    
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(prices, mean_demand, label="Average Demand", color="blue")
        plt.fill_between(prices, lower_bound, upper_bound, color="blue", alpha=0.2, label="95% CI")
        plt.title("Demand Curve with Confidence Intervals")
        plt.xlabel("Price")
        plt.ylabel("Demand" + DEMAND_TYPE)
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        # Parameters
        time_steps = np.sort(np.random.choice(range(T), size=5, replace=False))  # Range of time steps
    
        # Simulate demands for each time step
        time_step_demands = []
        for t in time_steps:
            simulated_demands_t = np.array([
                [demand_function_wrapper(p, sigma, demand_function, time_step=t, **demand_params_for_type) for p in prices]
                for _ in range(num_simulations)
            ])
            time_step_demands.append(simulated_demands_t)
    
        # Calculate mean and confidence intervals for each time step
        mean_demand_time = [demands.mean(axis=0) for demands in time_step_demands]
        std_demand_time = [demands.std(axis=0) for demands in time_step_demands]
        lower_bound_time = [mean - 1.96 * std / np.sqrt(num_simulations) for mean, std in zip(mean_demand_time, std_demand_time)]
        upper_bound_time = [mean + 1.96 * std / np.sqrt(num_simulations) for mean, std in zip(mean_demand_time, std_demand_time)]
    
        # Plot the results
        plt.figure(figsize=(12, 8))
    
        for t, mean, lower, upper in zip(time_steps, mean_demand_time, lower_bound_time, upper_bound_time):
            plt.plot(prices, mean, label=f"Time Step {t}")
            plt.fill_between(prices, lower, upper, alpha=0.1)
    
        plt.title("Time-Dependent Demand Curves with Confidence Intervals")
        plt.xlabel("Price")
        plt.ylabel("Demand" + " " + DEMAND_TYPE)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True)
        plt.tight_layout()
        plt.show()




def demand_function_wrapper(p, sigma,demand_function ,time_step=None, **demand_params):
    """
    Wrapper to handle different types of demand functions dynamically.
    """
    if demand_function is None:
        raise ValueError("No demand function specified in demand_params.")

    # Inspect the function's arguments
    func_params = inspect.signature(demand_function).parameters

    # Call the function based on whether it requires `time_step`
    if 'time_step' in func_params:
        return demand_function(p, sigma=sigma, time_step=time_step, **demand_params)
    else:
        return demand_function(p, sigma=sigma, **demand_params)



# Linear demand curve (default)
def linear_demand(price, a, b, sigma):
    return max(a + b * price + np.random.normal(0, sigma), 0)

# Nonlinear demand curve (e.g., quadratic)
def quadratic_demand(price, a, b, c, sigma):
    # For economically sensible demand, ensure it only decreases with price.
    # Derivative: dD/dp = b + 2*c*p
    # If c > 0, the curve has a minimum at p_vertex = -b/(2c)
    # Beyond this point, demand would increase with price (nonsensical).
    # We return zero demand for prices beyond the vertex.
    if c > 0 and b < 0:
        p_vertex = -b / (2 * c)
        if price > p_vertex:
            return max(np.random.normal(0, sigma), 0)
    return max(a + b * price + c * price**2 + np.random.normal(0, sigma), 0)

# Seasonal demand (e.g., sinusoidal variation over time)
def seasonal_demand(price, a, b, sigma, time_step, period):
    seasonality = np.sin(2 * np.pi * time_step / period)
    return max(a + b * price + seasonality * a / 2 + np.random.normal(0, sigma), 0)

# Shifted demand curve (e.g., market conditions change midway)
def shifted_demand(price, a1, b1, a2, b2, sigma, time_step, shift_time):
    if time_step < shift_time:
        return max(a1 + b1 * price + np.random.normal(0, sigma), 0)
    else:
        return max(a2 + b2 * price + np.random.normal(0, sigma), 0)

def combined_demand(price, time_step, sigma, segments, time_thresholds):
    """
    Combined demand function that uses different sub-functions for different time intervals.

    Parameters:
        price (float): The price.
        time_step (int): The current time step.
        sigma (float): Noise level.
        segments (list): List of segment definitions with sub-functions and parameters.
        time_thresholds (list): List of time thresholds defining the segments.

    Returns:
        float: Simulated demand.

    Note:
        Depends on global DEMAND_TYPE_MAPPING defined at module level.
        Ensure this function is only called after module initialization.
    """
    # Use global DEMAND_TYPE_MAPPING (defined later in this module)
    global DEMAND_TYPE_MAPPING

    # Determine which segment to use based on the time step
    for i, threshold in enumerate(time_thresholds):
        if time_step < threshold:
            segment = segments[i]
            break
    else:
        segment = segments[-1]  # Use the last segment if time_step exceeds all thresholds

    # Extract the demand function and parameters
    demand_type = segment["type"]
    demand_function = DEMAND_TYPE_MAPPING[demand_type]
    demand_params = segment["params"]
    
    # Inspect the function's arguments
    func_params = inspect.signature(demand_function).parameters

    # Call the function based on whether it requires `time_step`
    if 'time_step' in func_params:
        # Call the appropriate demand function
        to_return = demand_function(price, sigma=sigma, time_step=time_step, **demand_params)
    else:
        to_return = demand_function(price, sigma=sigma, **demand_params)
    return to_return


def stepwise_demand(price, thresholds, demands, sigma):
    """
    Stepwise demand curve based on price thresholds.

    Args:
        price (float): The price of the product.
        thresholds (list of float): List of price thresholds that define the steps.
        demands (list of float): List of demand levels corresponding to each threshold.
        sigma (float): Noise level in the demand.

    Returns:
        float: Demand at the given price.
    """
    if len(thresholds) + 1 != len(demands):
        raise ValueError("Number of demands must be one more than the number of thresholds.")

    # Determine the demand level based on the price
    for i, threshold in enumerate(thresholds):
        if price <= threshold:
            demand = demands[i]
            break
    else:
        demand = demands[-1]  # If price exceeds all thresholds, use the last demand level.

    demand += np.random.normal(0, sigma)  # Add noise
    return max(demand, 0)






def read_and_fit_demand(input_data):
    """
    Reads historical demand data and fits a demand function.
    Args:
        csv_file (str): Path to the CSV file with columns 'time', 'price', 'demand'.
    Returns:
        fitted_function: A function representing the estimated demand curve.
    """
    # Load data
    
    # Check if the input is a file path (string)
    if isinstance(input_data, str):
        try:
            data = pd.read_csv(input_data)
        except FileNotFoundError:
            raise ValueError(f"File not found: {input_data}")
    # Check if the input is already a pandas DataFrame
    elif isinstance(input_data, pd.DataFrame):
        data = input_data
    else:
        raise ValueError("Input must be either a file path (string) or a pandas DataFrame.")
            
    # Exponential decay model: naturally monotonic decreasing, always positive
    def demand_model(price, a, b):
        return a * np.exp(-b * price)

    # Fit the model to the data
    prices = data['price'].values
    demands = data['demand'].values

    # Initial guess: a = max demand, b = small positive decay rate
    p0 = [demands.max() if demands.max() > 0 else 100, 0.05]
    # Bounds: a > 0, b > 0 (ensures decreasing curve)
    bounds = ([0, 0], [np.inf, np.inf])
    params, _ = curve_fit(demand_model, prices, demands, p0=p0, bounds=bounds, maxfev=10000)

    # Return the fitted demand function
    def fitted_function(price):
        return demand_model(price, *params)

    return fitted_function



def demand_from_csv(price, sigma, csv_function, time_step=None, **kwargs):
    """
    Wrapper for historical data-based demand estimation.
    Args:
        price (float): The price at which demand is to be estimated.
        sigma (float): Standard deviation for noise (ignored in this case).
        csv_function (callable): Fitted function from CSV data.
    Returns:
        float: Estimated demand for the given price.
    """
    return csv_function(price)



DEMAND_TYPE_MAPPING = {
    "linear": linear_demand,
    "quadratic": quadratic_demand,
    "seasonal": seasonal_demand,
    "shifted": shifted_demand,
    "stepwise": stepwise_demand,
    "combined": combined_demand,
    "csv": None
}


################# Sanity checks #################################
def simulate_demand_and_revenue(prices, demand_function, params, sigma, T=100, num_simulations=10):
    """
    Simulates demand and revenue for the given demand function over time.
    
    Args:
        prices: List of price points.
        demand_function: The demand function to use (e.g., linear_demand).
        params: Dictionary of parameters for the demand function.
        sigma: Noise level in the demand function.
        T: Number of time steps for the simulation.
        num_simulations: Number of simulations to average over.
    
    Returns:
        A dictionary with average cumulative max/min revenue and corresponding prices.
    """
    cumulative_max_revenue = []
    cumulative_min_revenue = []
    
    for _ in range(num_simulations):
        max_revenues = []
        min_revenues = []
        
        for t in range(1, T + 1):
            func_params = inspect.signature(demand_function).parameters
            if 'time_step' in func_params:  # If the function is time-dependent
                params['time_step'] = t
            
            demands = [demand_function(price, **params, sigma=sigma) for price in prices]
            revenues = [price * demand for price, demand in zip(prices, demands)]
            
            max_revenue = max(revenues)
            min_revenue = min(revenues)
            
            max_revenues.append(max_revenue)
            min_revenues.append(min_revenue)
        
        cumulative_max_revenue.append(np.cumsum(max_revenues))
        cumulative_min_revenue.append(np.cumsum(min_revenues))
    
    # Compute averages across simulations
    avg_cumulative_max_revenue = np.mean(cumulative_max_revenue, axis=0)
    avg_cumulative_min_revenue = np.mean(cumulative_min_revenue, axis=0)
    
    return {
        "avg_cumulative_max_revenue": avg_cumulative_max_revenue,
        "avg_cumulative_min_revenue": avg_cumulative_min_revenue,
    }

if __name__ == "__main__":
    # Example Usage:
    prices_A = np.linspace(10, 100, 50)  # Example price range
    prices_B = np.linspace(10, 50, 50)
    prices_C = np.linspace(20, 80, 50)
    prices_D = np.linspace(10, 90, 50)
    prices_E = np.linspace(15, 70, 50)
    prices_F = np.linspace(20, 60, 50)
    prices_G = np.linspace(30, 100, 50)
    T = 100  # Number of time steps
    sigma = 5  # Noise level
    num_simulations = 50  # Number of simulations
    
    # Linear demand example
    params_linear_A = {"a": 100, "b": -2}
    params_linear_D = {"a": 120, "b": -3}
    params_linear_G = {"a": 110, "b": -2.5}
    results_linear_A = simulate_demand_and_revenue(prices_A, linear_demand, params_linear_A, sigma, T, num_simulations)
    results_linear_D = simulate_demand_and_revenue(prices_D, linear_demand, params_linear_D, sigma, T, num_simulations)
    results_linear_G = simulate_demand_and_revenue(prices_G, linear_demand, params_linear_G, sigma, T, num_simulations)

    # Quadratic demand example
    params_quadratic_B = {"a": 50, "b": -1, "c": 0.01}
    params_quadratic_E = {"a": 55, "b": -1.2, "c": 0.02}
    results_quadratic_B = simulate_demand_and_revenue(prices_B, quadratic_demand, params_quadratic_B, sigma, T, num_simulations)
    results_quadratic_E = simulate_demand_and_revenue(prices_E, quadratic_demand, params_quadratic_E, sigma, T, num_simulations)

    # Seasonal demand example
    params_seasonal_C = {"a": 70, "b": -0.5, "period": 30}
    params_seasonal_F = {"a": 65, "b": -0.4, "period": 30}
    results_seasonal_C = simulate_demand_and_revenue(prices_C, seasonal_demand, params_seasonal_C, sigma, T, num_simulations)
    results_seasonal_F = simulate_demand_and_revenue(prices_F, seasonal_demand, params_seasonal_F, sigma, T, num_simulations)









