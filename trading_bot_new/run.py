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
from dataset.Dataset import TradingDataset, TradingDataset_V3
from utils.utils import show_train_result, get_device, show_eval_result, make_plot, make_dataframe, make_plot_upper
import pandas as pd

from agent import Agent
from train import Trainer #other methods
from train_smaller_steps import Trainer_InSteps #"small_steps"
import os

def get_args():
    parser = argparse.ArgumentParser(description="Stock Trading Bot Training")
    parser.add_argument("--data_dir", type=str, default="../data/",
                        help="Directory where your data files are located")
    parser.add_argument("--stock_name", type=str, required=True,
                        help="Name of the training stock data file (e.g. GOOG.csv)")
    parser.add_argument("--strategy", type=str, default="t-dqn",
                        help="Training strategy: dqn, t-dqn, or double-dqn")
    parser.add_argument("--window_size", type=int, default=50,
                        help="Window size for the n-day state representation")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning Rate")
    parser.add_argument("--episodes", type=int, default=7, #different methods need different periods!!!!!!!
                        help="Number of training episodes")
    parser.add_argument("--model_name", type=str, default="model_debug",
                        help="Name of the model for saving/loading")
    parser.add_argument("--pretrained", action="store_true",
                        help="Whether to continue training a pretrained model")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging during evaluation")
    return parser.parse_args()


def main(stock_name, mdp, strategy):
    mdp = mdp #"10%_steps", "all_10%_steps", "all_or_nothing"
    strategy = strategy #"Transformer", "t-dqn", "double-dqn"

    #hyperparameters:
    ep_count = 0 #This might need to change
    if strategy == "Transformer":
      ep_count = 3 
    elif strategy == "t-dqn":
      if mdp == "10%_steps":
        ep_count = 7 
      if mdp == "all_10%_steps":
        ep_count = 7 
      if mdp == "all_or_nothing":
        ep_count = 7  
    elif strategy == "double-dqn":
      if mdp == "10%_steps":
        ep_count = 5 
      if mdp == "all_10%_steps":
        ep_count = 7 
      if mdp == "all_or_nothing":
         ep_count = 7

    data_dir = "../data" #correct one???
    window_size = 20
    batch_size = 32
    model_name = "test"
    pretrained = False
    debug = False
    default_learningrate = 1e-3

    if mdp == "" or strategy == "": raise ValueError("No value for mdp or strategy is given")
      
    # The state size is window_size - 1 because TradingDataset.get_state returns a tensor of shape (1, window_size-1)
    if mdp == "10%_steps": state_size = 2*(window_size -1) +1
    elif mdp == "all_10%_steps" or mdp == "all_or_nothing":
      state_size = 2*(window_size - 1)
    else: raise ValueError("Wrong mdp")

    # Create the agent with the correct state size and device.
    agent = Agent(state_size, default_learningrate, strategy=strategy, pretrained=pretrained,
                  model_name=model_name, device=get_device(), mdp=mdp)
    
    # Load the training and validation datasets.
    train_data = TradingDataset_V3(data_dir, stock_name, window_size, type='train')
    val_data = TradingDataset_V3(data_dir, stock_name, window_size, type='test')
    
    # Calculate an initial offset for display purposes.
    # (Assumes that the third element in each __getitem__ is a numeric value.)
    initial_offset = val_data[1][2].item() - val_data[0][2].item()
    #initial_offset = 0
    
    # Create DataLoaders from the datasets.
    dataloader_train = DataLoader(train_data, batch_size=1, shuffle=False)
    dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False)
    
    # Create the Trainer instance.
    if mdp == "10%_steps": trainer = Trainer_InSteps(dataloader_train, dataloader_val, agent, batch_size, window_size, mdp=mdp)
    elif mdp == "all_10%_steps" or mdp == "all_or_nothing":  
       trainer = Trainer(dataloader_train, dataloader_val, agent, batch_size, window_size, mdp=mdp)
    else: raise ValueError("Wrong mdp")
    # Run training over the specified episodes.
    # go_to_gym expects an iterable of episodes and a dataset.
    trainer.go_to_gym(range(1, ep_count + 1), train_data)
    
    # Evaluate the agent on the validation dataset.
    val_profit, timeline = trainer.testing(val_data)
    
    # Display the training and validation results.
    train_result = (ep_count, ep_count, trainer.train_profit, 0.0)
    show_train_result(train_result, val_profit, initial_offset)

    data_frame = make_dataframe(os.path.join(data_dir, stock_name, 'combined_test_data.csv'))
    return timeline, data_frame
    #make_plot(data_frame, timeline, title=f"Test {stock_name}")

    #val_profit, timeline = trainer.testing(train_data)
    #data_frame = make_dataframe(os.path.join(data_dir, stock_name, 'combined_train_data.csv'))
    #make_plot(data_frame, timeline, title=f"Train {stock_name}")


if __name__ == "__main__":
    #from the parser:
    args = get_args()

    data_dir = args.data_dir 
    window_size = args.window_size
    batch_size = args.batch_size
    ep_count = args.episodes
    strategy = args.strategy
    model_name = args.model_name 
    pretrained = args.pretrained
    debug = args.debug
    mdp = ""

    #Adjustments
    data_dir = "../data"
    stock_name = "novotech" #first_north
    window_size = 50
    batch_size = 32
    ep_count = 2 #This might need to change
    strategy = "double-dqn" #"Transformer", "t-dqn", "double-dqn"
    model_name = "test"
    pretrained = False
    debug = False
    mdp = "all_or_nothing" #"10%_steps", "all_10%_steps", "all_or_nothing"


    #coloredlogs.install(level="DEBUG")
    try:
        main(stock_name, mdp, strategy)
    except KeyboardInterrupt:
        print("Aborted!")

