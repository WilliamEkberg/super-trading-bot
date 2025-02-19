import os
import math
import logging

import pandas as pd
import numpy as np
import torch  # Used to determine the computing device

# Formats Position
format_position = lambda price: ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))

# Formats Currency
format_currency = lambda price: '${0:.2f}'.format(abs(price))


def show_train_result(result, val_position, initial_offset):
    """Displays training results.

    Args:
        result (tuple): A tuple containing (episode, ep_count, total_profit, avg_loss).
        val_position (float): The evaluation position.
        initial_offset (float): The initial offset value.
    """
    if val_position == initial_offset or val_position == 0.0:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: USELESS  Train Loss: {:.4f}'
                     .format(result[0], result[1], format_position(result[2]), result[3]))
    else:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: {}  Train Loss: {:.4f}'
                     .format(result[0], result[1], format_position(result[2]), format_position(val_position), result[3]))


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
    """Reads stock data from a CSV file.

    Args:
        stock_file (str): Path to the CSV file containing stock data.

    Returns:
        list: List of adjusted close prices.
    """
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
