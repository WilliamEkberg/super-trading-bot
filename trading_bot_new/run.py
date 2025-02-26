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

import logging
import coloredlogs
from docopt import docopt
from torch.utils.data import DataLoader
from dataset.Dataset import TradingDataset

from agent import Agent
from train import Train


def main(train_stock, val_stock, window_size, batch_size, ep_count,
         strategy="t-dqn", model_name="model_debug", pretrained=False,
         debug=False):
    """Trains the stock trading bot using Deep Q-Learning.
    Please see https://arxiv.org/abs/1312.5602 for more details.
    """
    
    # Create the agent with the selected device (GPU if available, otherwise CPU)
    agent = Agent(window_size, strategy=strategy, pretrained=pretrained,
                  model_name=model_name, device=get_device())
    
    train_data = TradingDataset(train_stock)
    val_data = get_stock_data(val_stock)
    initial_offset = val_data[1] - val_data[0]



    dataloader = DataLoader(train_stock, batch_size=batch_size, shuffle=False)
    trainer = Train(train_stock, agent,batch_size, window_size)
    

    trainer

    for episode in range(1, ep_count + 1):
        train_result = go_to_gym(agent, episode, train_data,
                                   ep_count=ep_count,
                                   batch_size=batch_size,
                                   window_size=window_size)
        val_result, _ = testing(agent, val_data, window_size, debug)
        show_train_result(train_result, val_result, initial_offset)


if __name__ == "__main__":
    args = docopt(__doc__)

    train_stock = args["<train-stock>"]
    val_stock = args["<val-stock>"]
    strategy = args["--strategy"]
    window_size = int(args["--window-size"])
    batch_size = int(args["--batch-size"])
    ep_count = int(args["--episode-count"])
    model_name = args["--model-name"]
    pretrained = args["--pretrained"]
    debug = args["--debug"]

    coloredlogs.install(level="DEBUG")
    try:
        main(train_stock, val_stock, window_size, batch_size,
             ep_count, strategy=strategy, model_name=model_name, 
             pretrained=pretrained, debug=debug)
    except KeyboardInterrupt:
        print("Aborted!")
