"""
Script for training Stock Trading Bot.

Usage:
  train.py <train-stock> <val-stock> [--strategy=<strategy>]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--pretrained] [--debug]

Options:
  --strategy=<strategy>             Q-learning strategy to use for training the network. Options:
                                      `dqn` i.e. Vanilla DQN,
                                      `t-dqn` i.e. DQN with fixed target distribution,
                                      `double-dqn` i.e. DQN with separate network for value estimation. [default: t-dqn]
  --window-size=<window-size>       Size of the n-day window stock data representation
                                    used as the feature vector. [default: 10]
  --batch-size=<batch-size>         Number of samples to train on in one mini-batch
                                    during training. [default: 32]
  --episode-count=<episode-count>   Number of trading episodes to use for training. [default: 50]
  --model-name=<model-name>         Name of the pretrained model to use. [default: model_debug]
  --pretrained                      Specifies whether to continue training a previously
                                    trained model (reads `model-name`).
  --debug                           Specifies whether to use verbose logs during eval operation.
"""

import coloredlogs
import argparse
from docopt import docopt
from torch.utils.data import DataLoader
from dataset.Dataset import TradingDataset
from utils.utils import show_train_result, get_device, show_eval_result, make_plot, make_dataframe
import pandas as pd

from agent import Agent
from train import Trainer

def get_args():
    parser = argparse.ArgumentParser(description="Stock Trading Bot Training")
    parser.add_argument("--data_dir", type=str, default="../data/",
                        help="Directory where your data files are located")
    parser.add_argument("--train_stock_name", type=str, required=True,
                        help="Name of the training stock data file (e.g. GOOG.csv)")
    parser.add_argument("--val_stock_name", type=str, required=True,
                        help="Name of the validation stock data file (e.g. GOOG_2018.csv)")
    parser.add_argument("--strategy", type=str, default="t-dqn",
                        help="Training strategy: dqn, t-dqn, or double-dqn")
    parser.add_argument("--window_size", type=int, default=50,
                        help="Window size for the n-day state representation")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning Rate")
    parser.add_argument("--episodes", type=int, default=12,
                        help="Number of training episodes")
    parser.add_argument("--model_name", type=str, default="model_debug",
                        help="Name of the model for saving/loading")
    parser.add_argument("--pretrained", action="store_true",
                        help="Whether to continue training a pretrained model")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging during evaluation")
    return parser.parse_args()



def main(data_dir, train_stock_name, val_stock_name, window_size, batch_size, ep_count,
         strategy="t-dqn", model_name="model_debug", pretrained=False, debug=False):
    # The state size is window_size - 1 because TradingDataset.get_state returns a tensor of shape (1, window_size-1)
    state_size = window_size - 1

    # Create the agent with the correct state size and device.
    agent = Agent(state_size, args.lr, strategy=strategy, pretrained=pretrained,
                  model_name=model_name, device=get_device())
    
    # Load the training and validation datasets.
    train_data = TradingDataset(data_dir, train_stock_name, window_size)
    val_data = TradingDataset(data_dir, val_stock_name, window_size)
    
    # Calculate an initial offset for display purposes.
    # (Assumes that the third element in each __getitem__ is a numeric value.)
    initial_offset = val_data[1][2].item() - val_data[0][2].item()
    #initial_offset = 0
    
    # Create DataLoaders from the datasets.
    dataloader_train = DataLoader(train_data, batch_size=1, shuffle=False)
    dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False)
    
    # Create the Trainer instance.
    trainer = Trainer(dataloader_train, dataloader_val, agent, batch_size, window_size)
    
    # Run training over the specified episodes.
    # go_to_gym expects an iterable of episodes and a dataset.
    trainer.go_to_gym(range(1, ep_count + 1), train_data)
    
    # Evaluate the agent on the validation dataset.
    val_profit, timeline = trainer.testing(val_data)
    
    # Display the training and validation results.
    train_result = (ep_count, ep_count, trainer.train_profit, 0.0)
    show_train_result(train_result, val_profit, initial_offset)

    data_frame = make_dataframe(val_stock_name)
    make_plot(data_frame, timeline, title=val_stock_name)

    val_profit, timeline = trainer.testing(train_data)
    data_frame = make_dataframe(train_stock_name)
    make_plot(data_frame, timeline, title=train_stock_name)


if __name__ == "__main__":
    args = get_args()
    #coloredlogs.install(level="DEBUG")
    try:
        main(args.data_dir, args.train_stock_name, args.val_stock_name, args.window_size,
             args.batch_size, args.episodes, strategy=args.strategy, model_name=args.model_name, 
             pretrained=args.pretrained, debug=args.debug)
    except KeyboardInterrupt:
        print("Aborted!")

