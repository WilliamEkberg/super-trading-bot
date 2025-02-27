import math
import logging
import pandas as pd
import numpy as np
import torch

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
