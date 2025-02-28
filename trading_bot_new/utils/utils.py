import math
import logging
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def sigmoid(x: float) -> float:
    """Computes the sigmoid of x."""
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + str(err))
        return 0.0  # fallback value

def format_position(price):
    # If price is a tensor, convert it to a float
    if isinstance(price, torch.Tensor):
        price = price.item()
    return ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))

def format_currency(price):
    # Similarly, convert price if it's a tensor
    if isinstance(price, torch.Tensor):
        price = price.item()
    return '${0:.2f}'.format(abs(price))


def show_train_result(result, evaluation_position, initial_offset):
    print('Episode {}/{} - Train Position: {}  Val Position: USELESS  Train Loss: {:.4f}'
                     .format(result[0], result[1], format_position(result[2]), result[3]))
    if evaluation_position == initial_offset or evaluation_position == 0.0:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: USELESS  Train Loss: {:.4f}'
                     .format(result[0], result[1], format_position(result[2]), result[3]))
    else:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: {}  Train Loss: {:.4f}'
                     .format(result[0], result[1], format_position(result[2]), format_position(evaluation_position), result[3]))


def show_eval_result(model_name, profit, initial_offset):
    """Displays evaluation results.

    Args:
        model_name (str): The model's name.
        profit (float): The profit value.
        initial_offset (float): The initial offset value.
    """
    if profit == initial_offset or profit == 0.0:
        logging.info('{}: USELESS\n'.format(model_name))
    else:
        logging.info('{}: {}\n'.format(model_name, format_position(profit)))


def get_stock_data(stock_file):

    df = pd.read_csv(stock_file)

    return list(df['Adj Close'])


def get_device():
    """Determines and returns the available device (GPU if available, otherwise CPU).

    Returns:
        torch.device: The selected device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.debug("Using device: {}".format(device))
    return device

def make_plot(df, history, title="Trading Session"):
    if isinstance(history, torch.Tensor):
        history = history.tolist()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract positions and actions
    position = np.array([history[0][0]] + [x[0] for x in history])
    actions = ['HODL'] + [x[1] for x in history]
    df = df.copy()  # Avoid modifying original DataFrame
    df['position'] = position
    df['action'] = actions
    
    # Plot stock positions
    ax.plot(df['date'], df['position'], label='Stock Position', color='green', alpha=0.5)
    
    # Plot BUY and SELL actions
    buy_signals = df[df['action'] == 'Buying']
    sell_signals = df[df['action'] == 'Selling']
    ax.scatter(buy_signals['date'], buy_signals['position'], color='blue', label='BUY', marker='^', s=100)
    ax.scatter(sell_signals['date'], sell_signals['position'], color='red', label='SELL', marker='v', s=100)
    
    # Formatting
    ax.set(title=title, xlabel="Date", ylabel="Price")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.show()

def make_dataframe(stock_name):
    df = pd.read_csv(f"../data/{stock_name}", usecols=['Date', 'Adj Close'])
    df.rename(columns={'Adj Close': 'actual', 'Date': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    return df