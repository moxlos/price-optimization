#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 2024

@author: lefteris
"""

import streamlit as st
import numpy as np
import pandas as pd
from demands import plot_demand_streamlit, plot_demand_and_revenue_streamlit, DEMAND_TYPE_MAPPING, read_and_fit_demand, demand_from_csv
from pricing_algorithms import comparison_report, comparison_plots
import param_module
import plotly.graph_objects as go


def compute_ts_prior(demand_type, demand_params, prices):
    """
    Compute meaningful Thompson Sampling priors based on demand type.

    Thompson Sampling uses a linear model D = a + b*p. This function computes
    a linear approximation of the demand curve near the revenue-maximizing price.

    Parameters
    ----------
    demand_type : str
        Type of demand function ('linear', 'quadratic', 'seasonal')
    demand_params : dict
        Parameters for the demand function
    prices : array-like
        Array of candidate prices

    Returns
    -------
    list
        [intercept, slope] for the Thompson Sampling prior
    """
    a = demand_params.get('a', 100)
    b = demand_params.get('b', -1)

    if demand_type == 'linear':
        # D = a + b*p, Revenue R = p*(a + b*p) = a*p + b*p²
        # Optimal price: dR/dp = a + 2b*p = 0 → p* = -a/(2b)
        # Just use the actual linear parameters
        return [a, b]

    elif demand_type == 'quadratic':
        # D = a + b*p + c*p², Revenue R = a*p + b*p² + c*p³
        # dR/dp = a + 2b*p + 3c*p² = 0
        c = demand_params.get('c', 0)
        if c == 0:
            return [a, b]

        # Solve quadratic: 3c*p² + 2b*p + a = 0
        discriminant = 4*b**2 - 12*a*c
        if discriminant < 0:
            # No real solution, use midpoint of price range
            p_opt = (prices[0] + prices[-1]) / 2
        else:
            # Two solutions, pick the one in valid range
            p1 = (-2*b + np.sqrt(discriminant)) / (6*c)
            p2 = (-2*b - np.sqrt(discriminant)) / (6*c)

            # Choose the price that's in the valid range and gives positive demand
            candidates = [p for p in [p1, p2] if prices[0] <= p <= prices[-1]]
            if candidates:
                # Pick the one with higher revenue
                def revenue(p):
                    d = max(a + b*p + c*p**2, 0)
                    return p * d
                p_opt = max(candidates, key=revenue)
            else:
                p_opt = (prices[0] + prices[-1]) / 2

        # Linearize D = a + b*p + c*p² around p_opt
        # D(p) ≈ D(p_opt) + D'(p_opt)*(p - p_opt)
        # D'(p) = b + 2c*p
        slope = b + 2*c*p_opt
        d_at_opt = a + b*p_opt + c*p_opt**2
        intercept = d_at_opt - slope*p_opt

        return [intercept, slope]

    elif demand_type == 'seasonal':
        # D = a + b*p + sin(...)*a/2
        # Base linear component is [a, b], seasonal adds variation
        # Use the base linear parameters
        return [a, b]

    else:
        # Default fallback
        return [a, b]


# Function to apply gradient row by row
def gradient_by_row(row):
    """Apply color gradient to row based on values (higher = greener)."""
    # Handle case where all values are equal (would cause division by zero)
    if row.max() == row.min():
        # All values equal - return neutral background
        return ["background-color: rgba(0, 255, 0, 0.3);" for _ in row]

    # Normalize row values between 0 and 1 for gradient
    normalized = (row - row.min()) / (row.max() - row.min())
    return [f"background-color: rgba(0, 255, 0, {value});" for value in normalized]


def configure_demand(demand_params, DEMAND_TYPE):
    for param, value in demand_params.get(DEMAND_TYPE, {}).items():
        # Handle simple numeric parameters
        if isinstance(value, (int, float)):
            demand_params[DEMAND_TYPE][param] = st.number_input(param, value=value)
        
        # Handle lists (e.g., for stepwise thresholds and demands)
        elif isinstance(value, list) and not isinstance(value[0], dict):
            st.write(f"{param}: {value}")
            updated_list = st.text_input(f"Enter a comma-separated list for {param}", value=",".join(map(str, value)))
            try:
                demand_params[DEMAND_TYPE][param] = list(map(float, updated_list.split(',')))
            except ValueError:
                st.warning(f"Invalid input for {param}. Please enter a comma-separated list of numbers.")
        
        # Handle nested parameters (e.g., for combined segments)
        elif isinstance(value, list) and isinstance(value[0], dict):  # Special case for `combined`
            st.write(f"{param}:")
            for i, segment in enumerate(value):
                st.write(f"Segment {i+1} ({segment['type']}):")
                for seg_param, seg_value in segment["params"].items():
                    if isinstance(seg_value, (int, float)):
                        segment["params"][seg_param] = st.number_input(f"{seg_param} (Segment {i+1})", value=seg_value)
                    elif isinstance(seg_value, list):
                        updated_seg_list = st.text_input(
                            f"Enter a comma-separated list for {seg_param} (Segment {i+1})", 
                            value=",".join(map(str, seg_value))
                        )
                        try:
                            segment["params"][seg_param] = list(map(float, updated_seg_list.split(',')))
                        except ValueError:
                            st.warning(f"Invalid input for {seg_param}. Please enter a comma-separated list of numbers.")
        
        # Special handling for `time_thresholds`
        elif param == "time_thresholds":
            st.write(f"{param}: {value}")
            updated_thresholds = st.text_input(f"Enter a comma-separated list for {param}", value=",".join(map(str, value)))
            try:
                demand_params[DEMAND_TYPE][param] = list(map(float, updated_thresholds.split(',')))
            except ValueError:
                st.warning(f"Invalid input for {param}. Please enter a comma-separated list of numbers.")
    return demand_params


st.set_page_config(layout="wide")

# Initialize session state for persistent data across page reruns
if 'demand_type' not in st.session_state:
    st.session_state.demand_type = "linear"
if 'demand_params' not in st.session_state:
    st.session_state.demand_params = param_module.demand_params.copy()
if 'results' not in st.session_state:
    st.session_state.results = param_module.results.copy()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Configuration", "Results","Multiple Product Simulations"])

# Page: Configuration
if page == "Configuration":
    st.title("Configure Pricing Simulation")
    st.caption("""
    This tool compares different **dynamic pricing algorithms** that learn optimal prices through experimentation.
    Configure a demand model below, then run the simulation to see how each algorithm performs.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.demand_type = st.selectbox("Select Demand Type", list(DEMAND_TYPE_MAPPING.keys()))

        if st.session_state.demand_type == "csv":
            st.write("Upload a CSV with columns: 'time', 'demand', 'price'.")
            uploaded_file = st.file_uploader("Choose a CSV file")
            if uploaded_file:
                try:
                    dt = pd.read_csv(uploaded_file)

                    # Validate CSV structure
                    required_cols = ['time', 'demand', 'price']
                    if not all(col in dt.columns for col in required_cols):
                        st.error(f"CSV must have columns: {required_cols}. Found: {list(dt.columns)}")
                    elif dt.empty:
                        st.error("CSV file is empty. Please provide data.")
                    elif dt.isnull().any().any():
                        st.warning("CSV contains NaN values - they will be dropped.")
                        dt = dt.dropna()
                        if dt.empty:
                            st.error("All rows contained NaN values. Please provide valid data.")
                        else:
                            st.info(f"Dropped NaN values. {len(dt)} rows remaining.")

                    # Validate data ranges
                    if not dt.empty and all(col in dt.columns for col in required_cols):
                        if (dt['demand'] < 0).any():
                            st.error("Demand values must be non-negative.")
                        elif (dt['price'] <= 0).any():
                            st.error("Price values must be positive.")
                        else:
                            # All validations passed - fit the demand function
                            fitted_demand_function = read_and_fit_demand(dt)
                            DEMAND_TYPE_MAPPING['csv'] = lambda p, sigma, **kwargs: demand_from_csv(p, sigma, fitted_demand_function)
                            st.success(f"✓ CSV loaded successfully with {len(dt)} data points.")

                except Exception as e:
                    st.error(f"Failed to load data: {e}")
            else:
                st.warning("No file selected. Please upload a csv.")

        else:
            st.write(f"**{st.session_state.demand_type.capitalize()} Demand**")

        st.write("Adjust demand-specific parameters below:")

        st.session_state.demand_params = configure_demand(st.session_state.demand_params, st.session_state.demand_type)
        st.write("Simulation Parameters")
        T = st.slider("Number of time steps", 50, 200, 100)
        num_simulations = st.slider("Number of simulations", 10, 500, 100)

        min_price = st.sidebar.number_input("Minimum Price", value=10, step=1, min_value=0)
        max_price = st.sidebar.number_input("Maximum Price", value=100, step=1, min_value=1)
        num_prices = st.sidebar.number_input("Number of Price Points", value=50, step=1, min_value=2)

        sigma = st.sidebar.number_input("Sigma (Demand Noise)", value=5.0, step=0.1, min_value=0.0)

        # Algorithm Hyperparameters
        st.sidebar.subheader("Algorithm Parameters")

        with st.sidebar.expander("Thompson Sampling", expanded=False):
            st.caption("Bayesian approach: maintains uncertainty about demand, naturally balances exploration/exploitation")
            ts_auto_prior = st.checkbox(
                "Auto-initialize prior from demand params",
                value=True,
                help="Use the configured demand parameters (a, b) as the prior mean. Recommended for faster convergence."
            )
            if ts_auto_prior:
                # Show what will be used
                demand_p = st.session_state.demand_params.get(st.session_state.demand_type, {})
                auto_a = demand_p.get('a', 0)
                auto_b = demand_p.get('b', -1)
                st.caption(f"Prior will be initialized to: a={auto_a}, b={auto_b}")
                ts_prior_mean_a = None
                ts_prior_mean_b = None
            else:
                ts_prior_mean_a = st.number_input("Prior mean (intercept a)", value=0.0, help="Prior belief for demand intercept parameter")
                ts_prior_mean_b = st.number_input("Prior mean (slope b)", value=-1.0, help="Prior belief for price sensitivity (typically negative)")
            ts_prior_cov_scale = st.number_input("Prior covariance scale", value=10.0, min_value=0.1, help="Uncertainty in prior beliefs (larger = more exploration)")

        with st.sidebar.expander("Epsilon-Greedy", expanded=False):
            st.caption("Simple rule: explore randomly ε% of the time, otherwise pick the best-known price")
            epsilon = st.slider("Exploration rate (ε)", 0.0, 1.0, 0.1, step=0.01, help="Probability of random exploration (0.1 = 10% exploration)")

        with st.sidebar.expander("Online Learning", expanded=False):
            st.caption("Weighted least squares: recent observations matter more (controlled by γ)")
            gamma = st.slider("Discount factor (γ)", 0.80, 1.0, 0.99, step=0.01, help="Weight for recent observations (0.99 = 1% discount per step)")

        with st.sidebar.expander("Uniform Pricing", expanded=False):
            st.caption("Fixed price baseline: uses the same price for all time periods")
            default_uniform_price = (min_price + max_price) / 2
            uniform_price = st.number_input("Fixed price", value=float(default_uniform_price), min_value=float(min_price), max_value=float(max_price), step=1.0, help="The fixed price to use (default: midpoint of price range)")

        # Store algorithm params in session state
        st.session_state.algo_params = {
            'ts_prior_cov_scale': ts_prior_cov_scale,
            'epsilon': epsilon,
            'gamma': gamma,
            'uniform_price': uniform_price
        }
        # Only include ts_prior_mean if NOT auto-initializing
        if not ts_auto_prior and ts_prior_mean_a is not None:
            st.session_state.algo_params['ts_prior_mean'] = [ts_prior_mean_a, ts_prior_mean_b]

        # Validate parameters before creating price array
        validation_errors = []
        if min_price >= max_price:
            validation_errors.append("Minimum price must be less than maximum price.")
        if num_prices < 2:
            validation_errors.append("Need at least 2 price points.")
        if T < 1:
            validation_errors.append("Number of time steps must be at least 1.")
        if sigma < 0:
            validation_errors.append("Sigma must be non-negative.")

        if validation_errors:
            for error in validation_errors:
                st.sidebar.error(error)
            prices = np.array([])  # Empty array to prevent crashes
        else:
            prices = np.linspace(min_price, max_price, int(num_prices))

        # Explain what the simulation does
        st.sidebar.markdown("---")
        st.sidebar.markdown("**What happens when you run:**")
        st.sidebar.caption(f"""
        1. Simulate {num_simulations} runs of {T} time steps each
        2. Each algorithm chooses prices, observes demand
        3. Compare revenues against the Oracle (perfect knowledge)
        """)

        if st.sidebar.button("Run Simulation", type="primary"):
            if validation_errors:
                st.error("Cannot run simulation: Please fix parameter errors in the sidebar.")
            else:
                with st.spinner(f"Running {num_simulations} simulations with {T} time steps each..."):
                    st.session_state.results, best_price = comparison_report(
                        prices, T, sigma,
                        st.session_state.demand_params[st.session_state.demand_type],
                        st.session_state.demand_type,
                        num_simulations,
                        st.session_state.algo_params
                    )
                st.success(f"Simulation completed! Oracle's average best price: {best_price:.2f}€")


    with col2:
        try:
            # Plot demand and revenue curves
            demand_fig, revenue_fig, theoretical_optimal = plot_demand_and_revenue_streamlit(
                DEMAND_TYPE_MAPPING, st.session_state.demand_type,
                prices, sigma, st.session_state.demand_params[st.session_state.demand_type],
                T,
                num_simulations=min(num_simulations, 50)  # Limit for faster preview
            )

            st.markdown("**Understanding the Demand-Revenue Relationship**")
            st.caption("The demand curve shows how many units sell at each price. "
                      "The revenue curve shows total earnings (Price × Demand). "
                      "The optimal price maximizes revenue, not demand.")

            st.plotly_chart(demand_fig, use_container_width=True)
            st.plotly_chart(revenue_fig, use_container_width=True)

            st.info(f"Theoretical optimal price: **{theoretical_optimal:.2f}€** (based on expected demand)")

        except Exception as e:
            st.error(f"Failed to plot: {e}")


# Page: Results
elif page == "Results":
    st.title("Simulation Results")

    # Check if results exist before displaying
    if not st.session_state.results or 'Online Learning' not in st.session_state.results:
        st.warning("No simulation results available. Please run a simulation on the Configuration page first.")
    else:
        # === SIMULATION CONTEXT SUMMARY ===
        config = st.session_state.results.get('_simulation_config', {})
        if config:
            st.subheader("What Was Simulated")
            col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
            with col_cfg1:
                st.markdown(f"""
                **Demand Model:** `{config.get('demand_type', 'N/A')}`
                **Parameters:** {config.get('demand_params', {})}
                """)
            with col_cfg2:
                st.markdown(f"""
                **Time Steps:** {config.get('T', 'N/A')}
                **Simulations:** {config.get('num_simulations', 'N/A')}
                **Noise (σ):** {config.get('sigma', 'N/A')}
                """)
            with col_cfg3:
                price_range = config.get('price_range', [0, 0])
                algo_params = config.get('algo_params', {})
                default_uniform = (price_range[0] + price_range[1]) / 2
                st.markdown(f"""
                **Price Range:** {price_range[0]:.0f} - {price_range[1]:.0f}€
                **Exploration (ε):** {algo_params.get('epsilon', 0.1)}
                **Discount (γ):** {algo_params.get('gamma', 0.99)}
                **Uniform Price:** {algo_params.get('uniform_price', default_uniform):.2f}€
                """)
            st.markdown("---")

        st.subheader("Revenue Comparison")
        st.caption("Each algorithm was run multiple times. Results show average performance across all simulation runs.")

        cola, colb = st.columns(2)
        with cola:
            for method, metrics in st.session_state.results.items():
                if method.startswith('_'):  # Skip internal keys
                    continue
                # Safely get the 'mean' key
                mean_value = metrics.get('mean')

                if mean_value is not None:  # Check if 'mean' exists and is not None
                    st.write(f"{method}: Mean Revenue = {mean_value:.2f}€")
                else:
                    st.write(f"{method}: Mean Revenue = Not Available")
            fig1, fig2 = comparison_plots(st.session_state.results)
        with colb:
            st.write("The *foresight (oracle) policy* serves as an idealized baseline, assuming perfect knowledge of the demand curve in advance—before any observations are made—and selecting the revenue-maximizing price at each time step. While unattainable in practice, it provides an upper bound for performance comparison.")

        col1, col2= st.columns(2)
        with col1:
            st.caption("**Regret** = Oracle revenue - Algorithm revenue. Lower regret means the algorithm is closer to optimal.")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.caption("**Cumulative Revenue** shows total earnings over time. The gap from the Oracle indicates room for improvement.")
            st.plotly_chart(fig2)

        # === ALGORITHM BEHAVIOR INSIGHTS ===
        example_runs = st.session_state.results.get('_example_runs', {})
        if example_runs:
            st.markdown("---")
            st.subheader("Algorithm Behavior (Single Run Example)")
            st.caption("This section shows what happened in ONE example simulation. See how each algorithm chose prices over time.")

            # Price trajectory plot
            fig_prices = go.Figure()
            config = st.session_state.results.get('_simulation_config', {})
            T = config.get('T', 100)

            # Add Oracle prices
            if 'Foresight (Oracle)' in example_runs:
                oracle_prices = example_runs['Foresight (Oracle)'].get('chosen_prices', [])
                if oracle_prices:
                    fig_prices.add_trace(go.Scatter(
                        x=list(range(len(oracle_prices))),
                        y=oracle_prices,
                        mode='lines',
                        name='Oracle (optimal)',
                        line=dict(dash='dash', color='gray'),
                        opacity=0.7
                    ))

            # Add Thompson Sampling prices
            if 'Thompson Sampling' in example_runs:
                ts_data = example_runs['Thompson Sampling']
                ts_prices = ts_data.get('chosen_prices', [])
                if ts_prices:
                    fig_prices.add_trace(go.Scatter(
                        x=list(range(len(ts_prices))),
                        y=ts_prices,
                        mode='lines',
                        name='Thompson Sampling',
                        line=dict(color='#1f77b4')
                    ))

            # Add Epsilon-Greedy prices with explore/exploit coloring
            if 'Epsilon-Greedy' in example_runs:
                eg_data = example_runs['Epsilon-Greedy']
                eg_prices = eg_data.get('chosen_prices', [])
                is_exploration = eg_data.get('is_exploration', [])
                if eg_prices:
                    # Split into explore and exploit segments
                    explore_x, explore_y = [], []
                    exploit_x, exploit_y = [], []
                    for i, (p, is_exp) in enumerate(zip(eg_prices, is_exploration)):
                        if is_exp:
                            explore_x.append(i)
                            explore_y.append(p)
                        else:
                            exploit_x.append(i)
                            exploit_y.append(p)

                    fig_prices.add_trace(go.Scatter(
                        x=exploit_x, y=exploit_y,
                        mode='markers',
                        name='ε-Greedy (exploit)',
                        marker=dict(color='#2ca02c', size=4)
                    ))
                    fig_prices.add_trace(go.Scatter(
                        x=explore_x, y=explore_y,
                        mode='markers',
                        name='ε-Greedy (explore)',
                        marker=dict(color='#ff7f0e', size=6, symbol='x')
                    ))

            # Add Online Learning prices
            if 'Online Learning' in example_runs:
                ol_data = example_runs['Online Learning']
                ol_prices = ol_data.get('chosen_prices', [])
                if ol_prices:
                    fig_prices.add_trace(go.Scatter(
                        x=list(range(len(ol_prices))),
                        y=ol_prices,
                        mode='lines',
                        name='Online Learning',
                        line=dict(color='#d62728')
                    ))

            fig_prices.update_layout(
                title="Price Choices Over Time",
                xaxis_title="Time Step",
                yaxis_title="Price Chosen (€)",
                legend_title="Algorithm",
                template="plotly_white",
                height=400
            )

            st.plotly_chart(fig_prices, use_container_width=True)

            # Algorithm-specific insights in expandable sections
            col_ts, col_eg, col_ol = st.columns(3)

            with col_ts:
                with st.expander("Thompson Sampling Details", expanded=False):
                    if 'Thompson Sampling' in example_runs:
                        ts_data = example_runs['Thompson Sampling']
                        variance = ts_data.get('posterior_variance', [])
                        if variance:
                            st.markdown("**Posterior Uncertainty Over Time**")
                            st.caption("As uncertainty decreases, the algorithm becomes more confident and exploits more.")
                            fig_var = go.Figure()
                            fig_var.add_trace(go.Scatter(
                                x=list(range(len(variance))),
                                y=variance,
                                mode='lines',
                                fill='tozeroy',
                                line=dict(color='#1f77b4')
                            ))
                            fig_var.update_layout(
                                xaxis_title="Time Step",
                                yaxis_title="Uncertainty (trace of covariance)",
                                height=250,
                                margin=dict(l=0, r=0, t=0, b=0)
                            )
                            st.plotly_chart(fig_var, use_container_width=True)

                            # Show convergence info
                            final_var = variance[-1]
                            initial_var = variance[0]
                            reduction = (1 - final_var / initial_var) * 100 if initial_var > 0 else 0
                            st.metric("Uncertainty Reduction", f"{reduction:.1f}%")

            with col_eg:
                with st.expander("Epsilon-Greedy Details", expanded=False):
                    if 'Epsilon-Greedy' in example_runs:
                        eg_data = example_runs['Epsilon-Greedy']
                        is_exploration = eg_data.get('is_exploration', [])
                        price_counts = eg_data.get('price_counts', {})
                        if is_exploration:
                            explore_count = sum(is_exploration)
                            exploit_count = len(is_exploration) - explore_count
                            st.markdown("**Explore vs Exploit Breakdown**")
                            st.caption(f"With ε={config.get('algo_params', {}).get('epsilon', 0.1)}, expect ~{config.get('algo_params', {}).get('epsilon', 0.1)*100:.0f}% exploration.")

                            # Pie chart
                            fig_pie = go.Figure(data=[go.Pie(
                                labels=['Explore (random)', 'Exploit (best known)'],
                                values=[explore_count, exploit_count],
                                marker_colors=['#ff7f0e', '#2ca02c'],
                                hole=0.4
                            )])
                            fig_pie.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
                            st.plotly_chart(fig_pie, use_container_width=True)

                            # Most tried prices
                            if price_counts:
                                top_prices = sorted(price_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                                st.markdown("**Most Tested Prices:**")
                                for p, count in top_prices:
                                    st.write(f"  {p:.2f}€: {count} times")

            with col_ol:
                with st.expander("Online Learning Details", expanded=False):
                    if 'Online Learning' in example_runs:
                        ol_data = example_runs['Online Learning']
                        param_estimates = ol_data.get('param_estimates', [])
                        if param_estimates:
                            st.markdown("**Parameter Estimates Over Time**")
                            st.caption("Shows how the algorithm learns the demand curve (D = a + b*price)")
                            a_vals = [p[0] for p in param_estimates]
                            b_vals = [p[1] for p in param_estimates]

                            fig_params = go.Figure()
                            fig_params.add_trace(go.Scatter(
                                x=list(range(len(a_vals))),
                                y=a_vals,
                                mode='lines',
                                name='Intercept (a)'
                            ))
                            fig_params.add_trace(go.Scatter(
                                x=list(range(len(b_vals))),
                                y=b_vals,
                                mode='lines',
                                name='Slope (b)'
                            ))
                            fig_params.update_layout(
                                xaxis_title="Time Step",
                                yaxis_title="Parameter Value",
                                height=250,
                                margin=dict(l=0, r=0, t=0, b=0)
                            )
                            st.plotly_chart(fig_params, use_container_width=True)

                            # Show final estimates
                            if a_vals and b_vals:
                                st.markdown(f"**Final Estimates:** a={a_vals[-1]:.2f}, b={b_vals[-1]:.2f}")
                                optimal_price = -a_vals[-1] / (2 * b_vals[-1]) if b_vals[-1] != 0 else 0
                                st.markdown(f"**Implied Optimal Price:** {optimal_price:.2f}€")

        # Export Results to CSV
        st.subheader("Export Results")
        if st.session_state.results and any(metrics.get('cumulative') is not None for metrics in st.session_state.results.values()):
            # Build DataFrame from results
            export_data = {'Time_Step': range(len(next(iter(st.session_state.results.values()))['cumulative']))}

            for method, metrics in st.session_state.results.items():
                if 'cumulative' in metrics:
                    export_data[f'{method}_Cumulative_Revenue'] = metrics['cumulative']

            results_df = pd.DataFrame(export_data)
            csv = results_df.to_csv(index=False)

            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="pricing_simulation_results.csv",
                mime="text/csv",
                help="Download cumulative revenue data for all algorithms over time"
            )

            # Show preview
            with st.expander("Preview Export Data"):
                st.dataframe(results_df.head(10))
        else:
            st.info("Run a simulation first to enable data export.")


elif page == "Multiple Product Simulations":
    col1, col2 = st.columns([1.5,2])
    
    st.write("Upload a CSV with columns: product_name,price_range,demand_type,a,b,c,sigma,T")
    uploaded_file = st.file_uploader("Choose a CSV file")
    with col1:
        dt = pd.DataFrame()
        if uploaded_file:
            try:
                dt = pd.read_csv(uploaded_file)
                st.dataframe(dt)
            except Exception as e:
                st.error(f"Failed to load data: {e}")
        else:
            st.warning("No file selected. Please upload a csv.")
    
    if st.sidebar.button("Run Simulation"):
        # Load the product data
        products = dt
        
        # Initialize a dictionary to store results for all products
        all_results = {}
        # Initialize a summary table
        summary_data = []
        # Iterate over each product in the CSV
        for idx, product in products.iterrows():
            product_name = product["product_name"]
            demand_type = product["demand_type"]
            price_range = product["price_range"]
            T = int(product["T"])
            sigma = float(product["sigma"])
            a = float(product["a"])
            b = float(product["b"])
            c = float(product["c"]) if not pd.isna(product["c"]) else None
        
            # Parse price range
            prices = np.linspace(*map(float, price_range.split("-")), 50)
        
            # Create demand parameters based on the demand type
            if demand_type == "linear":
                demand_params = {"a": a, "b": b}
            elif demand_type == "quadratic":
                demand_params = {"a": a, "b": b, "c": c}
            elif demand_type == "seasonal":
                demand_params = {"a": a, "b": b, "period": 30}  # Example: fixed period
            else:
                raise ValueError(f"Unsupported demand type: {demand_type}")
        
            # Run the comparison report
            print(f"Running simulation for {product_name} ({demand_type})...")
            # Use median price as default uniform price
            uniform_price_default = prices[len(prices)//2]
            # Compute meaningful TS prior based on demand type
            ts_prior = compute_ts_prior(demand_type, demand_params, prices)
            results, best_price = comparison_report(
                prices, T, sigma,
                demand_params_for_type=demand_params,
                DEMAND_TYPE=demand_type,
                num_simulations=30,
                algo_params={'gamma': 0.99, 'epsilon': 0.1, 'ts_prior_mean': ts_prior, 'ts_prior_cov_scale': 10.0, 'uniform_price': uniform_price_default}
            )
        
            # Save results to dictionary
            all_results[product_name] = results
            # Extract cumulative revenues and optimal prices
            foresight_revenue = results["Foresight (Oracle)"]["cumulative"][-1]
            thompson_revenue = results["Thompson Sampling"]["cumulative"][-1]
            epsilon_revenue = results["Epsilon-Greedy"]["cumulative"][-1]
            uniform_revenue = results["Uniform Pricing"]["cumulative"][-1]
            online_revenue = results["Online Learning"]["cumulative"][-1]
            optimal_price = best_price
            # Append data
            summary_data.append({
                "Product Name": product_name,
                "Demand Type": demand_type,
                "Optimal Price": optimal_price,
                "Foresight (Oracle)": foresight_revenue,
                "Thompson Sampling": thompson_revenue,
                "Epsilon-Greedy": epsilon_revenue,
                "Uniform Pricing": uniform_revenue,
                "Online Learning": online_revenue
            })
        st.success("Simulation completed successfully!")
        
        revenue_methods = ["Foresight (Oracle)", "Thompson Sampling", "Epsilon-Greedy", "Uniform Pricing", "Online Learning"]

        with col2:
            summary_df = pd.DataFrame(summary_data)
            
        
            # Apply gradient style to the revenue columns by row
            styled_df = summary_df.style.apply(gradient_by_row, subset=revenue_methods, axis=1)
            
            st.dataframe(styled_df)
        
        # Bar Chart: Cumulative Revenues Comparison
        fig1 = go.Figure()
        
        
        for method in revenue_methods:
            fig1.add_trace(
                go.Bar(
                    x=summary_df["Product Name"],
                    y=summary_df[method],
                    name=method
                )
            )
        
        fig1.update_layout(
            title="Cumulative Revenue Comparison Across Products",
            xaxis_title="Products",
            yaxis_title="Cumulative Revenue (€)",
            barmode="group",
            legend_title="Pricing Methods"
        )
        
        col3, col4= st.columns(2)
        with col3:
            st.plotly_chart(fig1, use_container_width=True)
        with col4:
            # Scatter Plot: Optimal Prices vs. Revenues
            scatter_data = []
            
            for method in revenue_methods[1:]:  # Skip Oracle for scatter
                scatter_data.append(
                    go.Scatter(
                        x=summary_df["Optimal Price"],
                        y=summary_df[method],
                        mode="markers",
                        name=method,
                        marker=dict(size=10, opacity=0.8),
                        text=summary_df["Product Name"],  # Add product names for hover
            hovertemplate="<b>%{text}</b><br>Optimal Price: %{x} €<br>Revenue: %{y} €<extra></extra>"
                    )
                )
            
            fig2 = go.Figure(data=scatter_data)
            
            fig2.update_layout(
                title="Optimal Price vs. Revenue",
                xaxis_title="Optimal Price Oracle (€)",
                yaxis_title="Revenue (€)",
                legend_title="Pricing Methods",
                template="plotly_white"
            )
            
            st.plotly_chart(fig2)


