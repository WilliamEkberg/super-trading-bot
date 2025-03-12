import matplotlib.pyplot as plt
import numpy as np

import run  

DATASETS = ["FirstNorth.csv", "Novotek.csv"]
ACTION_SPACES = ["In_or_out", "Ten_action", "Percentage"]
METHODS = ["DQN", "d_DQN", "Transformer"]

N_RUNS = 5 

def run_multiple_times(dataset, action_space, method, n_runs=N_RUNS):
    all_timelines = []
    for run_index in range(n_runs):
        if dataset == "FirstNorth.csv":
            train_dataset = "FirstNorth_2019_2024.csv"
        else: 
            train_dataset = "Novotek_2019_2024.csv"

        timeline, _ = run.main(
            train_dataset = train_dataset,
            val_dataset= dataset,
            action_space=action_space,
            method=method,
        )
    
        all_timelines.append(timeline)
    return all_timelines


def get_daily_profit_from_timeline(timeline):
    daily_profits = [day[2] for day in timeline]  # If 'profit' is day[2]
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
                
                all_profits = []
                for timeline in all_timelines:
                    daily_profit = get_daily_profit_from_timeline(timeline)
                    all_profits.append(daily_profit)
                
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
                

                ax.set_title(method)
                ax.set_xlabel("Day")
                ax.set_ylabel("Profit")
                ax.grid(True)
                ax.legend()
            
            plt.tight_layout()
            plt.show()
            plt.savefig(f"{dataset}_{action_space}_profit.png")


if __name__ == "__main__":
    plot_all_results()
