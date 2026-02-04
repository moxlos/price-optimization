# Dynamic Pricing Optimization Platform

An interactive simulation platform for testing and comparing pricing policies across different demand scenarios. Particularly valuable for automating pricing strategies for low-cost items where extensive A/B testing may not be cost-effective.

## Problem Statement

This platform addresses the challenge of optimizing prices for products where traditional promotional campaigns might not justify the cost. By simulating various demand curves and testing multiple pricing algorithms, it enables data-driven pricing decisions without requiring expensive real-world A/B tests.

**Key Use Cases:**

- **Low-cost items**: Products with smaller price points that attract significant customer interest and contribute substantially to overall revenue
- **Price-sensitive markets**: Understanding optimal pricing strategies under different demand conditions
- **Algorithmic comparison**: Evaluating performance of different pricing policies (Thompson Sampling, Epsilon-Greedy, Online Learning) against optimal baselines
- **Non-stationary demand**: Testing how algorithms adapt to seasonal variations or market shifts

<img width="2499" height="1250" alt="1" src="https://github.com/user-attachments/assets/3ee38f8a-c5b1-4066-9590-1bc8e870f484" />
<img width="2499" height="1250" alt="2" src="https://github.com/user-attachments/assets/7c461dff-73b3-4388-a7eb-013f7864fe8f" />
<img width="2499" height="1250" alt="3" src="https://github.com/user-attachments/assets/16a9823f-d793-41e0-b9ea-065ce7476afa" />



## Features

- **5 Pricing Algorithms**
  - **Thompson Sampling**: Bayesian approach with posterior updates for exploration/exploitation balance
  - **Epsilon-Greedy**: Simple exploration strategy with fixed probability
  - **Online Learning**: Weighted least squares with discount factor for adapting to demand changes
  - **Uniform Pricing**: Fixed price baseline
  - **Oracle (Foresight)**: Optimal upper bound with perfect knowledge (theoretical benchmark)

- **6+ Demand Models**
  - **Linear**: Standard price-demand relationship
  - **Quadratic**: Nonlinear price-demand relationship
  - **Seasonal**: Periodic demand variations (e.g., holidays, seasons)
  - **Shifted**: Market condition changes over time (e.g., competitor entry)
  - **Stepwise**: Discrete price-point effects
  - **Combined**: Multiple demand patterns over time
  - **CSV Upload**: Custom demand from historical data

- **Interactive Visualization**
  - Real-time demand curves with 95% confidence intervals
  - Cumulative revenue comparison across algorithms
  - Regret analysis (performance vs. optimal)
  - Multi-product batch analysis

- **Statistical Rigor**
  - Multiple simulation runs for robust results
  - Confidence interval estimation
  - Regret analysis vs. oracle baseline

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone moxlos/price_optimization
cd price_optimization

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run the Streamlit app
streamlit run price_opt_app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage Guide

### 1. Configuration Page

**Select Demand Type:**

- Choose from 8 pre-configured demand models or upload your own CSV data
- Adjust demand-specific parameters (e.g., slope, curvature, seasonality period)

**Set Simulation Parameters:**

- **Time steps (T)**: Number of pricing decisions to simulate (50-200)
- **Number of simulations**: Monte Carlo runs for statistical confidence (10-500)
- **Price range**: Minimum and maximum prices to test
- **Number of price points**: Discretization of price space
- **Sigma**: Demand noise level (standard deviation)

**Run Simulation:**

- Click "Run Simulation" to execute all algorithms
- View real-time demand curve visualization with confidence intervals

### 2. Results Page

**Revenue Comparison:**

- Mean cumulative revenue for each algorithm
- Interactive plots showing:
  - Cumulative revenue over time
  - Cumulative regret vs. oracle

**Interpretation:**

- **Foresight (Oracle)**: Theoretical upper bound - unattainable in practice
- **Thompson Sampling**: Typically performs closest to oracle through adaptive learning
- **Epsilon-Greedy**: Balanced approach with predictable exploration
- **Online Learning**: Adapts to non-stationary demands with discount factor
- **Uniform Pricing**: Simple baseline for comparison

### 3. Multiple Product Simulations

Upload a CSV with columns:

```
product_name,price_range,demand_type,a,b,c,sigma,T
Product A,10-100,linear,100,-2.0,,5,100
Product B,10-50,quadratic,50,-1.0,0.01,5,100
```

- **price_range**: "min-max" format
- **demand_type**: linear, quadratic, or seasonal
- **a, b, c**: Demand function parameters
- **sigma**: Noise level
- **T**: Number of time steps

Results include:

- Comparative bar charts across products
- Optimal price recommendations per product
- Color-coded performance heatmap

### Custom Demand Curves (CSV Upload)

Upload historical data with columns: `time`, `demand`, `price`

The system will:

1. Validate data integrity (no NaN, non-negative values)
2. Fit a nonlinear demand function using least squares
3. Use the fitted model for simulations

**Example CSV:**

```csv
time,demand,price
0,95.2,10
0,85.1,15
1,90.3,12
...
```

## Algorithms Explained

### Thompson Sampling

Uses Bayesian linear regression to maintain a posterior distribution over demand parameters. At each time step:

1. Sample demand parameters from posterior
2. Choose price maximizing expected revenue given sampled parameters
3. Observe actual demand
4. Update posterior using Bayes' rule

**Strengths**: Naturally balances exploration/exploitation, strong theoretical guarantees
**Assumptions**: Linear demand approximation

### Epsilon-Greedy

Simple exploration strategy:

- With probability ε (0.1): explore by choosing random price
- With probability 1-ε: exploit by choosing price with best historical performance

**Strengths**: Simple, predictable
**Weaknesses**: Fixed exploration rate may be suboptimal

### Online Learning

Weighted least squares with exponential discount:

- Recent observations weighted more heavily (weight = γ^(n-t))
- Adapts to non-stationary demands
- Discount factor γ = 0.99

**Strengths**: Handles demand shifts
**Weaknesses**: Requires tuning discount factor

## Interpreting Results

### Cumulative Revenue Plot

- **Higher is better**: Algorithm earning more total revenue
- **Convergence**: Lines stabilizing indicate learning completion
- **Spread**: Distance from oracle shows optimization gap

### Regret Plot

- **Regret = Oracle Revenue - Algorithm Revenue**
- **Lower is better**: Less regret means better performance
- **Sublinear growth**: Indicates effective learning (regret grows slower than time)

### When to Use Each Algorithm

| Algorithm         | Best For                         | Avoid When                      |
| ----------------- | -------------------------------- | ------------------------------- |
| Thompson Sampling | Unknown demand, need exploration | Non-linear demand               |
| Epsilon-Greedy    | Simple needs, interpretability   | Need fast convergence           |
| Online Learning   | Non-stationary demand            | Stationary demand (over-adapts) |
| Uniform Pricing   | Known optimal price              | Learning required               |

## Limitations & Caveats

⚠️ **Important Considerations:**

1. **Simulation vs. Reality**: Results are based on assumed demand models. Always validate with real A/B tests before deployment.

2. **Linear Approximation**: Thompson Sampling assumes linear demand. Performance degrades for highly nonlinear demand curves.

3. **Gaussian Noise**: Assumes normally distributed demand noise. Real demand may have different distributions.

4. **No Inventory Constraints**: Assumes unlimited inventory. Real-world stock constraints not modeled.

5. **Static Competition**: Doesn't model competitor pricing responses.

6. **Ethical Pricing**: Automated pricing must be used responsibly. Avoid exploiting consumer behavior or creating negative experiences.

## Technical Architecture

```
price_opt_app.py          # Streamlit UI and page routing
├── demands.py            # Demand function definitions and plotting
├── pricing_algorithms.py    # Pricing algorithms and comparison logic
├── param_module.py       # Default parameters and state initialization
└── data/                 # Sample datasets
    ├── dummy_demand_data.csv
    └── products_demand_parameters.csv
```

**Key Design Patterns:**

- **Dynamic function inspection**: Uses `inspect.signature()` to handle time-dependent vs. time-independent demands
- **Session state management**: Streamlit `session_state` for persistence across page reruns
- **Wrapper functions**: `demand_function_wrapper()` normalizes different function signatures

## Development

### Running Tests

```bash
# Run pricing algorithm simulation (standalone)
python pricing_algorithms.py

# Run demand curve plotting (standalone)
python demands.py
```

### Adding New Demand Functions

1. Define function in `demands.py`:

```python
def my_demand(price, sigma, param1, param2, time_step=None):
    demand = param1 * price + param2
    return max(demand + np.random.normal(0, sigma), 0)
```

2. Add to `DEMAND_TYPE_MAPPING`:

```python
DEMAND_TYPE_MAPPING["my_demand"] = my_demand
```

3. Add default parameters to `param_module.py`:

```python
demand_params["my_demand"] = {"param1": 100, "param2": -2}
```

### Adding New Pricing Algorithms

1. Implement in `pricing_algorithms.py` with signature:

```python
def my_algorithm(prices, T, sigma, demand_function, demand_params, **kwargs):
    revenues = []
    # ... implementation ...
    return revenues
```

2. Add to `comparison_report()`:

```python
my_revenues_list = np.array([
    simulate_pricing(prices, T, gamma, sigma, demand_params_for_type,
                     DEMAND_TYPE, my_algorithm)
    for _ in range(num_simulations)
])
```

## References

**Algorithms:**

- Agrawal, S., & Goyal, N. (2013). Thompson sampling for contextual bandits with linear payoffs. *ICML*.
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*, 47(2-3), 235-256.

**Demand Modeling:**

- Veblen, T. (1899). *The Theory of the Leisure Class*.
- Perloff, J. M. (2017). *Microeconomics*. Pearson.

## License

GNU General Public License v3.0 see LICENSE file for details

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Disclaimer**: This tool is for simulation and educational purposes. Always validate pricing strategies with real A/B tests before production deployment. Use ethical pricing practices.







