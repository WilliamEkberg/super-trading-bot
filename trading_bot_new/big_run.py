import matplotlib.pyplot as plt
import numpy as np

import run  

DATASETS = ["FirstNorth.csv", "Novotek.csv"]
ACTION_SPACES = ["all_or_nothing", "10%_steps", "all_10%_steps"]
METHODS = ["t-dqn", "double-dqn", "Transformer"]

N_RUNS = 3


def run_multiple_times(dataset, action_space, method, n_runs=N_RUNS):
    all_timelines = []
    for run_index in range(n_runs):
        if dataset == "FirstNorth.csv":
            train_dataset = "FirstNorth_2019_2024.csv"
            stock_name = "first_north"
        else: 
            train_dataset = "Novotek_2019_2024.csv"
            stock_name = "novotech"


        timeline, _ = run.main(
            stock_name,
            action_space,
            method,
        )
    
        all_timelines.append(timeline)
    return all_timelines


def get_daily_profit_from_timeline(timeline):
    daily_profits = [day[3] for day in timeline]  # If 'profit' is day[2]
    return np.array(daily_profits, dtype=np.float32)

def plot_all_results():
    """
    6 figeures
    2 datasets x 3 Action Spaces => 6 combos
    Each figure has 3 subplots (METHODS)
    every subplot has mean daly profit (+- std) over all N runs
    """
    for dataset in DATASETS:
        for action_space in ACTION_SPACES:
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
            
            fig.suptitle(f"Dataset: {dataset} | Action Space: {action_space}")

            for col_idx, method in enumerate(METHODS):
                ax = axes[col_idx]
                
                all_timelines = run_multiple_times(dataset, action_space, method, N_RUNS)
                
                rois = []
                s_ratio = []
                all_profits = []
                for timeline in all_timelines:
                    daily_profit = get_daily_profit_from_timeline(timeline)
                    all_profits.append(daily_profit)
                    rois.append(calculate_period_rois(daily_profit))
                    s_ratio.append(sortino_ratio(daily_profit))

                all_profits = np.array(all_profits)  #(N runs, T)

                mean_profit = all_profits.mean(axis=0)
                std_profit  = all_profits.std(axis=0)
                
                T = len(mean_profit)
                x_axis = np.arange(T)
                ax.plot(x_axis, mean_profit, label=f"{method} (mean)")
                ax.fill_between(
                    x_axis,
                    mean_profit - std_profit,
                    mean_profit + std_profit,
                    alpha=0.2
                )
                
                ax.text(1, 0, f'ROI: {np.mean(rois):.2f} \nSortino: {np.mean(s_ratio):.2f}',
                fontsize=12, ha='right', va='bottom', transform=ax.transAxes)
                
                ax.set_title(method)
                ax.set_xlabel("Day")
                ax.set_ylabel("Profit")
                ax.grid(True)
                ax.legend()
            
            plt.tight_layout()
            plt.savefig(f"{dataset}_{action_space}_profit.png")
            plt.show()

def sortino_ratio(portfolio_values, risk_free_rate=0, target_return=0, periods_per_year=252):
    portfolio_values = np.array(portfolio_values)
    
    # Calculate percentage returns
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Calculate the excess return (annualized)
    mean_return = np.mean(returns)
    excess_return = (mean_return * periods_per_year) - risk_free_rate
    
    # Calculate the downside deviation
    # First, identify returns below the target
    downside_returns = returns[returns < target_return/periods_per_year]
    
    # If there are no downside returns, return infinity or a very large number
    if len(downside_returns) == 0:
        return float('inf')  # or return a very large number like 1000
    
    # Calculate the downside deviation (annualized)
    downside_deviation = np.sqrt(np.mean(np.square(downside_returns))) * np.sqrt(periods_per_year)
    
    # Calculate and return the Sortino ratio
    sortino_ratio = excess_return / downside_deviation
    
    return sortino_ratio

def calculate_period_rois(daily_portfolio_values):
    initial_value = daily_portfolio_values[0]
    final_value = daily_portfolio_values[-1]
    
    # Calculate ROI
    roi = (final_value - initial_value) / initial_value
    
    return roi


if __name__ == "__main__":
    plot_all_results()
