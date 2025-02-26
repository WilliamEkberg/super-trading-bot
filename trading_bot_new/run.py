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
from utils.utils import show_train_result, get_device, show_eval_result

from agent import Agent
from train import Trainer

def get_args():
    parser = argparse.ArgumentParser(description="Simple greeting script")

    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--train_stock_name", type=str)
    parser.add_argument("--val_stock_name", type=str)
    parser.add_argument("--strategy", type=str)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--model_name", type=str)


    # train_stock = args["<train-stock>"]
    # val_stock = args["<val-stock>"]
    # strategy = args["--strategy"]
    # window_size = int(args["--window-size"])
    # batch_size = int(args["--batch-size"])
    # ep_count = int(args["--episode-count"])
    # model_name = args["--model-name"]
    # pretrained = args["--pretrained"]
    # debug = args["--debug"]

    args = parser.parse_args()
    return args


def main(train_stock_name, val_stock_name, window_size, batch_size, ep_count,
         strategy="t-dqn", model_name="model_debug", pretrained=False,
         debug=False):
    """Trains the stock trading bot using Deep Q-Learning.
    Please see https://arxiv.org/abs/1312.5602 for more details.
    """
    
    # Create the agent with the selected device (GPU if available, otherwise CPU)
    agent = Agent(window_size, strategy=strategy, pretrained=pretrained,
                  model_name=model_name, device=get_device())
    
    train_data = TradingDataset(train_stock_name)
    val_data = TradingDataset(val_stock_name)
    initial_offset = val_data[1] - val_data[0]

    dataloader_train = DataLoader(train_stock_name, batch_size=1, shuffle=False)
    dataloader_val = DataLoader(train_stock_name, batch_size=1, shuffle=False)
    trainer = Trainer(dataloader_train, dataloader_val, agent, batch_size, window_size)

    for episode in range(1, ep_count + 1):
        train_result = trainer.go_to_gym(agent, episode, train_data,
                                   ep_count=ep_count,
                                   batch_size=batch_size,
                                   window_size=window_size)
        val_result, _ = trainer.testing(agent, val_data, window_size, debug)
        show_train_result(train_result, val_result, initial_offset)


if __name__ == "__main__":
    args = get_args()

    coloredlogs.install(level="DEBUG")
    try:
        main(args.train_stock_name, args.val_stock_name, args.window_size, args.batch_size,
             args.episodes, strategy=args.strategy, model_name=args.model_name, 
             pretrained=args.pretrained, debug=args.debug)
    except KeyboardInterrupt:
        print("Aborted!")
