import argparse
from torch.utils.data import DataLoader
from dataset.Dataset import TradingDataset_V2, TradingDataset
from utils.utils import show_train_result, get_device, show_eval_result, make_plot, make_dataframe

from agent import Agent
from train import Trainer
from transformer_model import Transformer

def get_args():
    parser = argparse.ArgumentParser(description="Stock Trading Bot Training")
    parser.add_argument("--data_dir", type=str, default="../data/", help="Directory where your data files are located")
    parser.add_argument("--train_stock_name", type=str, help="Name of the training stock data file (e.g. GOOG.csv)")
    parser.add_argument("--val_stock_name", type=str,
                        help="Name of the validation stock data file (e.g. GOOG_2018.csv)")
    parser.add_argument("--strategy", type=str, default="t-dqn",
                        help="Training strategy: dqn, t-dqn, or double-dqn")
    parser.add_argument("--window_size", type=int, default=50,
                        help="Window size for the n-day state representation")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Learning Rate")
    parser.add_argument("--model_name", type=str, default="model_debug",
                        help="Name of the model for saving/loading")
    parser.add_argument("--pretrained", action="store_true",
                        help="Whether to continue training a pretrained model")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging during evaluation")
  
    return parser.parse_args()


def main(data_dir, train_stock_name, val_stock_name, window_size, batch_size, ep_count,
         strategy="t-dqn", model_name="model_debug", pretrained=False, debug=False):
    state_size = window_size-1

    agent = Agent(state_size, args.lr, strategy=strategy, pretrained=pretrained, model_type="Transformer",
                  model_name=model_name, device=get_device())
    
    # Load the training and validation datasets.
    train_data = TradingDataset(data_dir, train_stock_name, window_size)
    val_data = TradingDataset(data_dir, val_stock_name, window_size)
    
    # Calculate an initial offset for display purposes.
    # (Assumes that the third element in each __getitem__ is a numeric value.)
    initial_offset = val_data[1][2].item() - val_data[0][2].item()
    
    dataloader_train = DataLoader(train_data, batch_size=1, shuffle=False)
    dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False)
    
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
    print(args)

    try:
        main(args.data_dir, args.train_stock_name, args.val_stock_name, args.window_size,
             args.batch_size, args.episodes, strategy=args.strategy, model_name=args.model_name, 
             pretrained=args.pretrained, debug=args.debug)
    except KeyboardInterrupt:
        print("Aborted!")

