import os
import math
import logging

import pandas as pd
import numpy as np
import torch


format_position = lambda price: ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))
format_currency = lambda price: '${0:.2f}'.format(abs(price))


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
