import matplotlib.pyplot as plt
import numpy as np

import run  

DATASETS = ["FirstNorth.csv", "Novotek.csv"]
ACTION_SPACES = ["all_or_nothing", "10%_steps", "all_10%_steps"]
METHODS = ["t-dqn", "double-dqn", "Transformer"]

N_RUNS = 10


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

def get_buy_and_hold_from_timeline(timeline, initial_value=10000):
    closing_prices = [day[0] for day in timeline]
    shares = initial_value / closing_prices[0]
    return shares * np.array(closing_prices)

def plot_all_results():
    """
    3 figeures
    2 datasets x 3 Action Spaces => 6 combos
    Each figure has 3 subplots (METHODS)
    every subplot has mean daly profit (+- std) over all N runs
    """
    colors = ['blue', 'green', 'red']
    for dataset in DATASETS:
        for action_space in ACTION_SPACES:
            #fig, ax = plt.subplots(figsize=(15, 5))
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
            fig.suptitle(f"Dataset: {dataset} | Action Space: {action_space}")
                

            for idx, method in enumerate(METHODS):
                
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
                ax1.plot(x_axis, mean_profit, color=colors[idx], label=f"{method} (mean)")
                # pc = ax1.fill_between(
                #     x_axis,
                #     mean_profit - std_profit,
                #     mean_profit + std_profit,
                #     alpha=0.2,
                #     edgecolor=colors[idx],
                #     linewidth=2,
                #     facecolor=colors[idx],

                #)
                #pc.set_linestyle(':') 
                
                x_offset = 0.98 - idx * 0.3
                ax1.text(x_offset , 0.1, 
                        f'{method}: ROI {np.mean(rois):.2f}\nSortino {np.mean(s_ratio):.2f}', 
                        fontsize=12, 
                        color=colors[idx],
                        ha='right', 
                        va='top', 
                        transform=ax1.transAxes,
                        clip_on=False)
                
                ax2.plot(x_axis, std_profit, color=colors[idx], label=f"{method} Standard Deviation")
                pc2 = ax2.fill_between(
                    x_axis,
                    0,
                    std_profit,
                    alpha=0.2,
                    facecolor=colors[idx],
                    edgecolor=colors[idx],
                    linewidth=2,
                )
                pc2.set_linestyle(':')


                #ax.text(1, 0, f'ROI: {np.mean(rois):.2f} \nSortino: {np.mean(s_ratio):.2f}',
                #fontsize=12, ha='right', va='bottom', transform=ax.transAxes)

            hold_profit = get_buy_and_hold_from_timeline(all_timelines[0], initial_value=10000)
            ax1.plot(x_axis, hold_profit, color='black', label=f"Buy and hold", linestyle='--', linewidth=3)

            ax1.set_title("Portfolio Value")
            ax1.set_ylabel("Portfolio Value")
            ax1.grid(True)
            ax1.legend()

            ax2.set_title("Standard Deviation Between Runs")
            ax2.set_xlabel("Day")
            ax2.set_ylabel("Standard Deviation")
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(f"{dataset}_{action_space}_profit.png")
            plt.savefig(f"{dataset}_{action_space}_{method}_profit.png")
            #plt.show()
            plt.close()

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
